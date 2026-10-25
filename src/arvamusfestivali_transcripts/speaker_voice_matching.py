"""Utilities for cautious cross-recording speaker-voice matching."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def select_speaker_cues(
    cues: Sequence[Mapping[str, Any]],
    *,
    min_seconds: float = 2.5,
    max_per_speaker: int = 6,
    min_speaker_share: float = 0.85,
) -> list[dict[str, Any]]:
    """Choose long, mostly single-speaker cues for stable voice embeddings."""
    if min_seconds <= 0:
        raise ValueError("min_seconds must be positive")
    if max_per_speaker < 1:
        raise ValueError("max_per_speaker must be at least one")
    if not 0 <= min_speaker_share <= 1:
        raise ValueError("min_speaker_share must be between zero and one")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cue in cues:
        try:
            speaker = str(cue["speaker"])
            start = float(cue["start_seconds"])
            end = float(cue["end_seconds"])
            share = float(cue.get("speaker_share", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not speaker or end - start < min_seconds or share < min_speaker_share:
            continue
        grouped[speaker].append(dict(cue))

    selected: list[dict[str, Any]] = []
    for speaker in sorted(grouped):
        candidates = sorted(
            grouped[speaker],
            key=lambda cue: (
                -(float(cue["end_seconds"]) - float(cue["start_seconds"])),
                float(cue["start_seconds"]),
            ),
        )
        speaker_selected: list[dict[str, Any]] = []
        for cue in candidates:
            start = float(cue["start_seconds"])
            end = float(cue["end_seconds"])
            if any(
                min(end, float(other["end_seconds"]))
                > max(start, float(other["start_seconds"]))
                for other in speaker_selected
            ):
                continue
            speaker_selected.append(cue)
            if len(speaker_selected) >= max_per_speaker:
                break
        selected.extend(sorted(speaker_selected, key=lambda cue: cue["start_seconds"]))
    return selected


def normalized_embedding(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a finite, unit-length one-dimensional embedding."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("embedding must contain finite values")
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("embedding must not be zero")
    return array / norm


def rank_named_candidates(
    query: Sequence[float] | np.ndarray,
    references: Mapping[str, Sequence[Sequence[float] | np.ndarray]],
) -> list[dict[str, Any]]:
    """Rank named voice references by cosine similarity to one query voice."""
    query_vector = normalized_embedding(query)
    ranked: list[dict[str, Any]] = []
    for name, vectors in references.items():
        if not vectors:
            continue
        normalized = [normalized_embedding(vector) for vector in vectors]
        if any(vector.shape != query_vector.shape for vector in normalized):
            raise ValueError(f"embedding dimensions do not match for {name}")
        centroid = normalized_embedding(np.mean(normalized, axis=0))
        ranked.append(
            {
                "name": name,
                "similarity": float(np.dot(query_vector, centroid)),
                "reference_count": len(normalized),
            }
        )
    return sorted(ranked, key=lambda item: item["similarity"], reverse=True)
