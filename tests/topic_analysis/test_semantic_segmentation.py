"""Behavioral contracts for cue-aligned semantic segmentation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from arvamusfestivali_transcripts.topic_analysis import (
    CanonicalEpisode,
    Cue,
    SemanticBoundary,
    SemanticSegmentationConfig,
    SemanticSegmentationResult,
    build_atomic_blocks,
    build_atomic_blocks_many,
)


def episode_with_cues(
    *,
    count: int,
    seconds_per_cue: float = 10,
    words_per_cue: int = 5,
    episode_id: str = "episode-1",
) -> CanonicalEpisode:
    cues = tuple(
        Cue(
            start_seconds=index * seconds_per_cue,
            end_seconds=(index + 1) * seconds_per_cue,
            text=" ".join([f"word{index}"] * words_per_cue),
        )
        for index in range(count)
    )
    return CanonicalEpisode(
        episode_id=episode_id,
        duplicate_episode_ids=(episode_id,),
        title=f"Discussion {episode_id}",
        published_at="2026-08-01T12:00:00Z",
        audio_sha256=("a" if episode_id == "episode-1" else "b") * 64,
        duration_seconds=count * seconds_per_cue,
        audio_url=f"https://example.test/{episode_id}.mp3",
        cues=cues,
        source_paths=(Path(f"{episode_id}.json"),),
    )


def episode_with_custom_cues(
    rows: list[tuple[float, float, str]], episode_id: str = "episode-1"
) -> CanonicalEpisode:
    cues = tuple(Cue(start, end, text) for start, end, text in rows)
    return CanonicalEpisode(
        episode_id=episode_id,
        duplicate_episode_ids=(episode_id,),
        title="Discussion",
        published_at="2026-08-01T12:00:00Z",
        audio_sha256="a" * 64,
        duration_seconds=max(cue.end_seconds for cue in cues),
        audio_url="https://example.test/audio.mp3",
        cues=cues,
        source_paths=(Path("episode.json"),),
    )


def test_semantic_config_rejects_incompatible_durations_and_quantile() -> None:
    with pytest.raises(ValueError, match="semantic segment durations"):
        SemanticSegmentationConfig(min_segment_seconds=700, max_segment_seconds=600)
    with pytest.raises(ValueError, match="atomic block durations"):
        SemanticSegmentationConfig(atomic_target_seconds=50, atomic_max_seconds=45)
    with pytest.raises(ValueError, match="boundary_quantile"):
        SemanticSegmentationConfig(boundary_quantile=1.1)


def test_atomic_blocks_are_cue_aligned_nonoverlapping_and_deterministic() -> None:
    episode = episode_with_cues(count=8)
    config = SemanticSegmentationConfig(atomic_target_seconds=30, atomic_max_seconds=45)

    blocks = build_atomic_blocks(episode, config)

    assert [(item.start_seconds, item.end_seconds) for item in blocks] == [
        (0, 30),
        (30, 60),
        (60, 80),
    ]
    assert [item.cue_count for item in blocks] == [3, 3, 2]
    assert "word0" in blocks[0].text and "word3" not in blocks[0].text
    assert blocks == build_atomic_blocks(episode, config)
    assert all(left.end_seconds <= right.start_seconds for left, right in zip(blocks, blocks[1:]))


def test_atomic_blocks_close_before_a_new_cue_would_exceed_maximum() -> None:
    episode = episode_with_custom_cues(
        [(0, 20, "first"), (20, 40, "second"), (40, 60, "third")]
    )
    config = SemanticSegmentationConfig(atomic_target_seconds=30, atomic_max_seconds=35)

    blocks = build_atomic_blocks(episode, config)

    assert [(item.start_seconds, item.end_seconds) for item in blocks] == [
        (0, 20),
        (20, 40),
        (40, 60),
    ]


def test_atomic_blocks_keep_an_indivisible_long_cue() -> None:
    episode = episode_with_custom_cues([(0, 120, "long cue"), (120, 130, "tail")])

    blocks = build_atomic_blocks(episode, SemanticSegmentationConfig())

    assert [(item.start_seconds, item.end_seconds) for item in blocks] == [
        (0, 120),
        (120, 130),
    ]


def test_atomic_blocks_many_preserves_episode_order() -> None:
    first = episode_with_cues(count=4, episode_id="episode-1")
    second = episode_with_cues(count=4, episode_id="episode-2")

    blocks = build_atomic_blocks_many((second, first), SemanticSegmentationConfig())

    assert [item.episode_id for item in blocks] == [
        "episode-2",
        "episode-2",
        "episode-1",
        "episode-1",
    ]


def test_semantic_public_records_are_frozen_and_reject_mutable_tuples() -> None:
    config = SemanticSegmentationConfig()
    boundary = SemanticBoundary("episode-1", 30, "left", "right", 0.5)
    passage = build_atomic_blocks(episode_with_cues(count=3), config)[0]
    result = SemanticSegmentationResult((passage,), (boundary,))

    for record, field in ((config, "context_seconds"), (boundary, "score"), (result, "passages")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, 1)
    with pytest.raises(ValueError, match="passages"):
        replace(result, passages=[passage])
    with pytest.raises(ValueError, match="boundaries"):
        replace(result, boundaries=[boundary])


def test_forced_boundary_must_also_be_selected() -> None:
    with pytest.raises(ValueError, match="forced boundary"):
        SemanticBoundary("episode-1", 30, "left", "right", 0.0, forced=True)
