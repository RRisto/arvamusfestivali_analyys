"""Load canonical transcript recordings and choose a diverse sample."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from arvamusfestivali_transcripts.archive import validate_archive

from .types import CanonicalEpisode, Cue


def _publication_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _load_archive(path: Path, year: int) -> CanonicalEpisode:
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("archive must be an object")
        validate_archive(payload, year)
        episode = payload["episode"]
        transcription = payload["transcription"]
        episode_id = episode["id"]
        int(episode_id)
        cues = tuple(
            Cue(cue["start_seconds"], cue["end_seconds"], cue["text"])
            for cue in transcription["cues"]
        )
        return CanonicalEpisode(
            episode_id=episode_id,
            duplicate_episode_ids=(episode_id,),
            title=episode["title"],
            published_at=episode["published_at"],
            audio_sha256=episode["audio_sha256"],
            duration_seconds=episode["duration_seconds"],
            audio_url=episode["audio_url"],
            cues=cues,
            source_paths=(path,),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid transcript archive {path}: {error}") from error


def load_corpus(root: Path, year: int) -> tuple[CanonicalEpisode, ...]:
    """Read a transcript year and retain every SHA duplicate as provenance."""
    grouped: dict[str, list[CanonicalEpisode]] = defaultdict(list)
    for path in sorted((root / str(year)).glob("*.json")):
        episode = _load_archive(path, year)
        grouped[episode.audio_sha256].append(episode)

    canonical: list[CanonicalEpisode] = []
    for duplicates in grouped.values():
        chosen = min(
            duplicates,
            key=lambda item: (
                _publication_time(item.published_at),
                int(item.episode_id),
                item.source_paths[0],
            ),
        )
        canonical.append(
            replace(
                chosen,
                duplicate_episode_ids=tuple(
                    sorted(
                        {item.episode_id for item in duplicates},
                        key=lambda episode_id: (int(episode_id), episode_id),
                    )
                ),
                source_paths=tuple(
                    sorted(path for item in duplicates for path in item.source_paths)
                ),
            )
        )
    return tuple(
        sorted(canonical, key=lambda item: (_publication_time(item.published_at), item.episode_id))
    )


def select_diverse_episodes(
    episodes: Sequence[CanonicalEpisode],
    count: int = 6,
    explicit_ids: Sequence[str] | None = None,
) -> tuple[CanonicalEpisode, ...]:
    """Choose canonical IDs in requested order or by deterministic title diversity."""
    items = tuple(sorted(episodes, key=lambda item: item.episode_id))
    if not items:
        raise ValueError("no canonical episodes available for selection")
    if explicit_ids:
        if len(set(explicit_ids)) != len(explicit_ids):
            raise ValueError("explicit episode IDs must not be repeated")
        by_id = {item.episode_id: item for item in items}
        unknown = sorted(set(explicit_ids) - by_id.keys())
        if unknown:
            raise ValueError(f"unknown canonical episode IDs: {', '.join(unknown)}")
        return tuple(by_id[episode_id] for episode_id in explicit_ids)

    if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= len(items):
        raise ValueError("count must be between one and the number of canonical episodes")
    if count == len(items):
        return items

    titles = [item.title for item in items]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    matrix = vectorizer.fit_transform(titles)
    centroid_similarity = cosine_similarity(matrix, np.asarray(matrix.mean(axis=0))).ravel()
    first_index = min(
        range(len(items)), key=lambda index: (centroid_similarity[index], items[index].episode_id)
    )
    selected = [first_index]
    similarities = cosine_similarity(matrix)
    while len(selected) < count:
        remaining = [index for index in range(len(items)) if index not in selected]
        next_index = min(
            remaining,
            key=lambda index: (
                max(similarities[index, chosen] for chosen in selected),
                items[index].episode_id,
            ),
        )
        selected.append(next_index)
    return tuple(items[index] for index in selected)
