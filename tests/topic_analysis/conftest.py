"""Small, valid transcript archives for topic-analysis tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from arvamusfestivali_transcripts.archive import write_archive as write_validated_archive


def write_archive(
    path: Path,
    *,
    episode_id: str,
    sha256: str,
    title: str,
    published_at: str,
    cues: Sequence[dict[str, float | str]] | None = None,
) -> Path:
    """Write a compact, complete version-one archive for topic-analysis tests."""
    transcript_cues = list(cues) if cues is not None else [
        {"start_seconds": 0.0, "end_seconds": 120.0, "text": "A public discussion begins."},
        {"start_seconds": 120.0, "end_seconds": 240.0, "text": "The speakers consider solutions."},
    ]
    payload = {
        "schema_version": 1,
        "episode": {
            "id": episode_id,
            "rss_guid": f"test:{episode_id}",
            "title": title,
            "published_at": published_at,
            "apple_collection_id": 1477431807,
            "page_url": f"https://example.test/episodes/{episode_id}",
            "audio_url": f"https://example.test/audio/{episode_id}.mp3",
            "audio_bytes": 1024,
            "audio_sha256": sha256,
            "duration_seconds": 240.0,
        },
        "transcription": {
            "model": {
                "name": "test-model",
                "repository": "example/test-model",
                "revision": "test-revision",
                "precision": "fp32",
            },
            "runtime": {"sherpa_onnx": "test", "decoding_method": "greedy"},
            "text": " ".join(str(cue["text"]) for cue in transcript_cues),
            "cues": transcript_cues,
        },
    }
    return write_validated_archive(path, payload)


@pytest.fixture
def write_topic_archive() -> Callable[..., Path]:
    def create(
        root: Path,
        *,
        episode_id: str = "1",
        sha256: str = "a" * 64,
        title: str = "Public discussion",
        published_at: str = "2026-08-01T12:00:00Z",
        cues: Sequence[dict[str, float | str]] | None = None,
    ) -> Path:
        return write_archive(
            root / "2026" / f"{episode_id}.json",
            episode_id=episode_id,
            sha256=sha256,
            title=title,
            published_at=published_at,
            cues=cues,
        )

    return create


@pytest.fixture
def six_topic_archives(tmp_path: Path, write_topic_archive: Callable[..., Path]) -> Path:
    """Create six distinct canonical recordings for notebook smoke tests."""
    titles = (
        "Teacher workload",
        "Preventive healthcare",
        "Climate adaptation",
        "Industrial investment",
        "Artificial intelligence",
        "Local democracy",
    )
    for index, title in enumerate(titles, start=1):
        write_topic_archive(
            tmp_path,
            episode_id=str(index),
            sha256=f"{index:x}" * 64,
            title=title,
        )
    return tmp_path
