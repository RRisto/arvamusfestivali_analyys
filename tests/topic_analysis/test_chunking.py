"""Cue-boundary passage construction and audio provenance."""

from __future__ import annotations

from pathlib import Path

from arvamusfestivali_transcripts.topic_analysis import (
    CanonicalEpisode,
    ChunkingConfig,
    Cue,
    chunk_episode,
    chunk_episodes,
)


def episode_with_cues(
    *, count: int, seconds_per_cue: float = 30, words_per_cue: int = 100,
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
        duplicate_episode_ids=(episode_id, f"{episode_id}-duplicate"),
        title="A public discussion",
        published_at="2026-08-01T12:00:00Z",
        audio_sha256="a" * 64,
        duration_seconds=count * seconds_per_cue,
        audio_url=f"https://example.test/audio/{episode_id}.mp3",
        cues=cues,
        source_paths=(Path(f"{episode_id}.json"),),
    )


def test_chunk_episode_preserves_cue_boundaries_and_overlap() -> None:
    episode = episode_with_cues(count=12)
    config = ChunkingConfig(
        min_seconds=60,
        target_seconds=120,
        max_seconds=150,
        min_words=300,
        max_words=500,
        overlap_seconds=30,
        merge_tail_seconds=45,
        max_merged_seconds=180,
    )

    passages = chunk_episode(episode, config)

    assert [(item.start_seconds, item.end_seconds) for item in passages] == [
        (0, 120), (90, 210), (180, 300), (270, 360),
    ]
    assert [item.cue_count for item in passages] == [4, 4, 4, 3]
    assert [item.word_count for item in passages] == [400, 400, 400, 300]
    assert passages[0].text.startswith("word0 ")
    assert passages[1].text.startswith("word3 ")
    assert passages[-1].text.endswith(" word11")
    assert [item.passage_id for item in passages] == [
        "episode-1:000000000000-000000120000",
        "episode-1:000000090000-000000210000",
        "episode-1:000000180000-000000300000",
        "episode-1:000000270000-000000360000",
    ]
    assert all(
        item.timestamped_audio_url
        == f"https://example.test/audio/episode-1.mp3#t={item.start_seconds:g}"
        for item in passages
    )
    assert all(item.duplicate_episode_ids == episode.duplicate_episode_ids for item in passages)


def test_short_final_chunk_merges_without_exceeding_limit() -> None:
    episode = episode_with_cues(count=5)
    config = ChunkingConfig(
        min_seconds=60,
        target_seconds=90,
        max_seconds=120,
        min_words=200,
        max_words=500,
        overlap_seconds=0,
        merge_tail_seconds=60,
        max_merged_seconds=180,
    )

    passages = chunk_episode(episode, config)

    assert len(passages) == 1
    assert (passages[0].start_seconds, passages[0].end_seconds) == (0, 150)
    assert passages[0].cue_count == 5
    assert passages[0].word_count == 500
    assert passages[0].passage_id == "episode-1:000000000000-000000150000"


def test_short_tail_stays_separate_when_merge_would_exceed_limit() -> None:
    episode = episode_with_cues(count=5)
    config = ChunkingConfig(
        min_seconds=60,
        target_seconds=120,
        max_seconds=120,
        min_words=200,
        max_words=500,
        overlap_seconds=0,
        merge_tail_seconds=60,
        max_merged_seconds=120,
    )

    passages = chunk_episode(episode, config)

    assert [(item.start_seconds, item.end_seconds) for item in passages] == [
        (0, 120), (120, 150),
    ]


def test_overlap_never_reemits_only_reused_cues() -> None:
    episode = episode_with_cues(count=3)
    config = ChunkingConfig(
        min_seconds=30,
        target_seconds=30,
        max_seconds=90,
        min_words=100,
        max_words=500,
        overlap_seconds=90,
        merge_tail_seconds=0,
        max_merged_seconds=90,
    )

    passages = chunk_episode(episode, config)

    assert [(item.start_seconds, item.end_seconds) for item in passages] == [
        (0, 30), (0, 60), (0, 90),
    ]


def test_chunk_episodes_preserves_episode_order_and_default_config() -> None:
    first = episode_with_cues(count=1, episode_id="first")
    second = episode_with_cues(count=1, episode_id="second")

    passages = chunk_episodes((second, first))

    assert [item.episode_id for item in passages] == ["second", "first"]
    assert [item.start_seconds for item in passages] == [0, 0]
    assert all(item.cue_count == 1 for item in passages)


def test_dense_speech_waits_for_minimum_duration_before_word_limit_closes() -> None:
    episode = episode_with_cues(count=8, seconds_per_cue=10, words_per_cue=100)
    config = ChunkingConfig(
        min_seconds=30, target_seconds=60, max_seconds=90,
        min_words=100, max_words=200, overlap_seconds=0,
        merge_tail_seconds=0, max_merged_seconds=90,
    )
    passages = chunk_episode(episode, config)
    assert [(p.start_seconds, p.end_seconds) for p in passages] == [
        (0, 30), (30, 60), (60, 80),
    ]
    assert [p.word_count for p in passages] == [300, 300, 200]
