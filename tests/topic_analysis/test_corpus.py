"""Corpus loading and reproducible episode selection."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from arvamusfestivali_transcripts.topic_analysis import load_corpus, select_diverse_episodes
from arvamusfestivali_transcripts.topic_analysis.types import CanonicalEpisode, Cue


def make_episodes(*titles: str) -> tuple[CanonicalEpisode, ...]:
    return tuple(
        CanonicalEpisode(
            episode_id=str(index),
            duplicate_episode_ids=(str(index),),
            title=title,
            published_at="2026-08-01T12:00:00Z",
            audio_sha256=f"{index:x}" * 64,
            duration_seconds=240.0,
            audio_url=f"https://example.test/audio/{index}.mp3",
            cues=(Cue(0.0, 1.0, "Speech"),),
            source_paths=(Path(f"{index}.json"),),
        )
        for index, title in enumerate(titles, start=1)
    )


def test_load_corpus_deduplicates_by_audio_hash_and_retains_provenance(
    tmp_path: Path, write_topic_archive: Callable[..., Path]
) -> None:
    second_path = write_topic_archive(tmp_path, episode_id="2", sha256="a" * 64, title="Same audio")
    first_path = write_topic_archive(tmp_path, episode_id="1", sha256="a" * 64, title="Same audio")
    write_topic_archive(tmp_path, episode_id="3", sha256="b" * 64, title="Other")

    episodes = load_corpus(tmp_path, 2026)

    assert [episode.episode_id for episode in episodes] == ["1", "3"]
    assert episodes[0].duplicate_episode_ids == ("1", "2")
    assert episodes[0].source_paths == tuple(sorted((first_path, second_path)))
    assert episodes[0].audio_url == "https://example.test/audio/1.mp3"
    assert episodes[0].cues == (
        Cue(0.0, 120.0, "A public discussion begins."),
        Cue(120.0, 240.0, "The speakers consider solutions."),
    )


def test_load_corpus_prefers_earliest_published_then_numeric_id(
    tmp_path: Path, write_topic_archive: Callable[..., Path]
) -> None:
    write_topic_archive(
        tmp_path, episode_id="10", sha256="a" * 64, title="Later",
        published_at="2026-08-02T12:00:00Z",
    )
    write_topic_archive(
        tmp_path, episode_id="9", sha256="a" * 64, title="Earlier",
        published_at="2026-08-01T12:00:00Z",
    )
    write_topic_archive(
        tmp_path, episode_id="2", sha256="a" * 64, title="Numeric winner",
        published_at="2026-08-01T12:00:00Z",
    )

    episodes = load_corpus(tmp_path, 2026)

    assert len(episodes) == 1
    assert episodes[0].episode_id == "2"
    assert episodes[0].title == "Numeric winner"
    assert episodes[0].duplicate_episode_ids == ("2", "9", "10")


def test_load_corpus_rejects_bad_archive_with_path_and_field(
    tmp_path: Path, write_topic_archive: Callable[..., Path]
) -> None:
    path = write_topic_archive(tmp_path, episode_id="1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["episode"]["audio_sha256"] = "not-a-sha"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid transcript archive .*1\.json: .*SHA-256"):
        load_corpus(tmp_path, 2026)


def test_load_corpus_rejects_non_object_archive_with_path(
    tmp_path: Path, write_topic_archive: Callable[..., Path]
) -> None:
    path = write_topic_archive(tmp_path, episode_id="1")
    path.write_text("null", encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"invalid transcript archive .*1\.json: archive must be an object"
    ):
        load_corpus(tmp_path, 2026)


def test_select_diverse_episodes_is_deterministic() -> None:
    episodes = make_episodes(
        "Teacher workload",
        "Early childhood education",
        "Preventive healthcare",
        "Artificial intelligence agents",
        "Climate policy",
        "Industrial investment",
        "Another teacher debate",
    )

    first = select_diverse_episodes(episodes)
    second = select_diverse_episodes(tuple(reversed(episodes)))

    assert tuple(item.episode_id for item in first) == tuple(item.episode_id for item in second)
    assert len({item.episode_id for item in first}) == 6


def test_diversity_selection_favors_distinct_titles() -> None:
    episodes = make_episodes("Teacher workload", "Teacher workloads", "Climate policy")

    selected = select_diverse_episodes(episodes, count=2)

    assert "3" in {episode.episode_id for episode in selected}
    assert len(selected) == 2


def test_explicit_selection_preserves_order_and_rejects_repetition() -> None:
    episodes = make_episodes("Health", "Education", "Climate")

    assert tuple(item.episode_id for item in select_diverse_episodes(
        episodes, explicit_ids=["3", "1"]
    )) == ("3", "1")
    with pytest.raises(ValueError, match="repeated"):
        select_diverse_episodes(episodes, explicit_ids=["1", "1"])


def test_explicit_selection_rejects_duplicate_audio_alias() -> None:
    episodes = make_episodes("Health", "Education")
    with pytest.raises(ValueError, match="unknown canonical episode IDs"):
        select_diverse_episodes(episodes, explicit_ids=["duplicate-alias"])


def test_selection_rejects_empty_or_oversized_request() -> None:
    episodes = make_episodes("Health", "Education")
    with pytest.raises(ValueError, match="count"):
        select_diverse_episodes(episodes, count=0)
    with pytest.raises(ValueError, match="count"):
        select_diverse_episodes(episodes, count=3)
    with pytest.raises(ValueError, match="no canonical episodes"):
        select_diverse_episodes(())
