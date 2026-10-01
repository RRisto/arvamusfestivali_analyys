"""Behavioral contracts for cue-aligned semantic segmentation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from arvamusfestivali_transcripts.topic_analysis import (
    CanonicalEpisode,
    Cue,
    EmbeddingResult,
    SemanticBoundary,
    SemanticSegmentationConfig,
    SemanticSegmentationResult,
    build_atomic_blocks,
    build_atomic_blocks_many,
    score_semantic_boundaries,
    segment_episode_semantically,
    segment_episodes_semantically,
    select_semantic_boundaries,
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


def blocks_with_duration(count: int, seconds_per_block: float = 30) -> tuple:
    episode = episode_with_cues(
        count=count,
        seconds_per_cue=seconds_per_block,
        words_per_cue=5,
    )
    config = SemanticSegmentationConfig(
        atomic_target_seconds=30,
        atomic_max_seconds=45,
    )
    return build_atomic_blocks(episode, config)


def test_contextual_score_finds_a_to_b_shift() -> None:
    blocks = blocks_with_duration(6)
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    config = SemanticSegmentationConfig(
        context_seconds=60,
        min_segment_seconds=60,
        max_segment_seconds=600,
        boundary_quantile=0.8,
    )

    scored = score_semantic_boundaries(blocks, vectors, config)
    selected = select_semantic_boundaries(blocks, scored, config)

    assert max(scored, key=lambda item: item.score).timestamp_seconds == 90
    assert [item.timestamp_seconds for item in selected if item.selected] == [90]
    assert not any(item.forced for item in selected)


def test_flat_embeddings_only_receive_required_maximum_cuts() -> None:
    blocks = blocks_with_duration(12, seconds_per_block=60)
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (12, 1))
    config = SemanticSegmentationConfig(
        context_seconds=60,
        min_segment_seconds=90,
        max_segment_seconds=300,
        boundary_quantile=0.85,
    )

    selected = select_semantic_boundaries(
        blocks,
        score_semantic_boundaries(blocks, vectors, config),
        config,
    )

    chosen = [item for item in selected if item.selected]
    assert [item.timestamp_seconds for item in chosen] == [300, 600]
    assert all(item.forced for item in chosen)


def test_minimum_segment_duration_rejects_an_otherwise_strong_cut() -> None:
    blocks = blocks_with_duration(6)
    vectors = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3, dtype=np.float32)
    config = SemanticSegmentationConfig(
        context_seconds=60,
        min_segment_seconds=100,
        max_segment_seconds=600,
        boundary_quantile=0.8,
    )

    selected = select_semantic_boundaries(
        blocks,
        score_semantic_boundaries(blocks, vectors, config),
        config,
    )

    assert not any(item.selected for item in selected)


def test_equal_score_ties_choose_the_earliest_local_maximum() -> None:
    blocks = blocks_with_duration(4, seconds_per_block=60)
    boundaries = tuple(
        SemanticBoundary(
            "episode-1",
            timestamp,
            blocks[index].passage_id,
            blocks[index + 1].passage_id,
            score,
        )
        for index, (timestamp, score) in enumerate(((60, 1.0), (120, 1.0), (180, 0.5)))
    )
    config = SemanticSegmentationConfig(
        min_segment_seconds=60,
        max_segment_seconds=600,
        boundary_quantile=0.5,
    )

    selected = select_semantic_boundaries(blocks, boundaries, config)

    assert [item.timestamp_seconds for item in selected if item.selected] == [60]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.ones((2, 2), dtype=np.float32), "rows"),
        (np.array([[1.0, 0.0], [np.nan, 0.0], [1.0, 0.0]]), "finite"),
        (np.array([[2.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), "normalized"),
    ],
)
def test_boundary_scoring_rejects_invalid_embedding_matrices(
    vectors: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_semantic_boundaries(blocks_with_duration(3), vectors, SemanticSegmentationConfig())


def test_boundary_scoring_rejects_blocks_from_multiple_episodes() -> None:
    first = blocks_with_duration(2)
    other_episode = episode_with_cues(count=2, seconds_per_cue=30, episode_id="episode-2")
    second = build_atomic_blocks(other_episode, SemanticSegmentationConfig())
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))

    with pytest.raises(ValueError, match="one episode"):
        score_semantic_boundaries((*first, *second), vectors, SemanticSegmentationConfig())


def test_indivisible_overlong_block_has_no_legal_forced_boundary() -> None:
    episode = episode_with_custom_cues([(0, 700, "one indivisible cue")])
    config = SemanticSegmentationConfig(max_segment_seconds=600)
    blocks = build_atomic_blocks(episode, config)

    scored = score_semantic_boundaries(blocks, np.array([[1.0, 0.0]]), config)

    assert scored == ()
    assert select_semantic_boundaries(blocks, scored, config) == ()


def shift_config() -> SemanticSegmentationConfig:
    return SemanticSegmentationConfig(
        atomic_target_seconds=30,
        atomic_max_seconds=45,
        context_seconds=60,
        min_segment_seconds=60,
        max_segment_seconds=600,
        boundary_quantile=0.8,
    )


def shift_vectors() -> np.ndarray:
    return np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3, dtype=np.float32)


def test_segment_episode_merges_original_cues_at_selected_cut() -> None:
    episode = episode_with_cues(count=6, seconds_per_cue=30)
    blocks = build_atomic_blocks(episode, shift_config())

    result = segment_episode_semantically(episode, blocks, shift_vectors(), shift_config())

    assert [(item.start_seconds, item.end_seconds) for item in result.passages] == [
        (0, 90),
        (90, 180),
    ]
    assert result.passages[0].text == " ".join(cue.text for cue in episode.cues[:3])
    assert result.passages[1].text == " ".join(cue.text for cue in episode.cues[3:])
    assert all(item.audio_sha256 == episode.audio_sha256 for item in result.passages)
    assert [item.timestamped_audio_url for item in result.passages] == [
        "https://example.test/episode-1.mp3#t=0",
        "https://example.test/episode-1.mp3#t=90",
    ]


def test_segment_episode_with_flat_embeddings_preserves_one_short_passage() -> None:
    episode = episode_with_cues(count=6, seconds_per_cue=30)
    blocks = build_atomic_blocks(episode, shift_config())
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))

    result = segment_episode_semantically(episode, blocks, vectors, shift_config())

    assert len(result.passages) == 1
    assert result.passages[0].text == " ".join(cue.text for cue in episode.cues)
    assert not any(item.selected for item in result.boundaries)


def test_many_episode_orchestrator_rejects_reordered_embedding_ids() -> None:
    episodes = (
        episode_with_cues(count=6, seconds_per_cue=30, episode_id="episode-1"),
        episode_with_cues(count=6, seconds_per_cue=30, episode_id="episode-2"),
    )
    blocks = build_atomic_blocks_many(episodes, shift_config())
    vectors = np.vstack((shift_vectors(), shift_vectors()))
    embedding_result = EmbeddingResult(
        model_key="bge",
        model_id="BAAI/bge-m3",
        model_revision="revision",
        dimension=2,
        passage_ids=tuple(item.passage_id for item in blocks),
        embeddings=vectors,
        cache_hits=0,
    )
    reordered = replace(
        embedding_result,
        passage_ids=tuple(reversed(embedding_result.passage_ids)),
        embeddings=embedding_result.embeddings[::-1],
    )

    with pytest.raises(ValueError, match="IDs and order"):
        segment_episodes_semantically(episodes, blocks, reordered, shift_config())


def test_many_episode_orchestrator_preserves_episode_order() -> None:
    episodes = (
        episode_with_cues(count=6, seconds_per_cue=30, episode_id="episode-2"),
        episode_with_cues(count=6, seconds_per_cue=30, episode_id="episode-1"),
    )
    blocks = build_atomic_blocks_many(episodes, shift_config())
    embedding_result = EmbeddingResult(
        model_key="bge",
        model_id="BAAI/bge-m3",
        model_revision="revision",
        dimension=2,
        passage_ids=tuple(item.passage_id for item in blocks),
        embeddings=np.vstack((shift_vectors(), shift_vectors())),
        cache_hits=0,
    )

    result = segment_episodes_semantically(episodes, blocks, embedding_result, shift_config())

    assert [item.episode_id for item in result.passages] == [
        "episode-2",
        "episode-2",
        "episode-1",
        "episode-1",
    ]
