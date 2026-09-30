from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arvamusfestivali_transcripts.topic_analysis.types import (
    CanonicalEpisode,
    ChunkingConfig,
    Cue,
    EmbeddingResult,
    ManualTopicReview,
    Passage,
    TopicRun,
)


def test_passage_is_immutable_and_builds_timestamp_link() -> None:
    passage = Passage(
        passage_id="ep-1:000000120-000000240",
        episode_id="ep-1",
        duplicate_episode_ids=("ep-1", "ep-2"),
        title="Education debate",
        start_seconds=120.0,
        end_seconds=240.0,
        text="Teachers need enough time to prepare lessons.",
        audio_sha256="a" * 64,
        audio_url="https://example.test/audio.mp3",
        word_count=7,
        cue_count=1,
    )

    assert passage.timestamped_audio_url == "https://example.test/audio.mp3#t=120"
    with pytest.raises(FrozenInstanceError):
        passage.title = "Changed"  # type: ignore[misc]


def test_passage_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="passage interval"):
        Passage(
            passage_id="bad",
            episode_id="ep-1",
            duplicate_episode_ids=("ep-1",),
            title="Bad interval",
            start_seconds=10.0,
            end_seconds=10.0,
            text="text",
            audio_sha256="a" * 64,
            audio_url="https://example.test/audio.mp3",
            word_count=1,
            cue_count=1,
        )


def test_timestamped_audio_url_preserves_fractional_seconds() -> None:
    passage = Passage(
        passage_id="ep-1:000007324-000007325",
        episode_id="ep-1",
        duplicate_episode_ids=("ep-1",),
        title="Discussion",
        start_seconds=7324.625,
        end_seconds=7325.0,
        text="Speech",
        audio_sha256="a" * 64,
        audio_url="https://example.test/audio.mp3",
        word_count=1,
        cue_count=1,
    )

    assert passage.timestamped_audio_url == "https://example.test/audio.mp3#t=7324.625"


def test_cue_rejects_nonfinite_interval() -> None:
    with pytest.raises(ValueError, match="cue interval"):
        Cue(float("nan"), 10.0, "Speech")


def test_chunking_config_rejects_unordered_targets() -> None:
    with pytest.raises(ValueError, match="chunk duration"):
        ChunkingConfig(min_seconds=180, target_seconds=120)


def test_embedding_result_rejects_incompatible_rows() -> None:
    with pytest.raises(ValueError, match="embedding rows"):
        EmbeddingResult(
            model_key="qwen",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            model_revision=None,
            dimension=2,
            passage_ids=("p1", "p2"),
            embeddings=np.ones((1, 2)),
            cache_hits=0,
        )


def test_public_types_are_frozen() -> None:
    cue = Cue(0.0, 1.0, "Speech")
    episode = CanonicalEpisode(
        episode_id="ep-1",
        duplicate_episode_ids=("ep-1",),
        title="Discussion",
        published_at="2026-09-01T12:00:00Z",
        audio_sha256="a" * 64,
        duration_seconds=1.0,
        audio_url="https://example.test/audio.mp3",
        cues=(cue,),
        source_paths=(Path("ep-1.json"),),
    )
    run = TopicRun(
        model_key="qwen",
        topics=np.array([0]),
        probabilities=np.array([[1.0]]),
        reduced_embeddings=np.ones((1, 2)),
        topic_info=pd.DataFrame({"Topic": [0]}),
        representative_passages={0: ("p1",)},
        cluster_persistence=(1.0,),
        passage_ids=("p1",),
    )
    review = ManualTopicReview("qwen", 0, "coherent")

    for item, field in (
        (cue, "text"),
        (episode, "title"),
        (ChunkingConfig(), "min_seconds"),
        (run, "model_key"),
        (review, "model_key"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(item, field, "changed")


def test_tuple_fields_reject_mutable_lists() -> None:
    episode = CanonicalEpisode(
        episode_id="ep-1",
        duplicate_episode_ids=("ep-1",),
        title="Discussion",
        published_at="2026-09-01T12:00:00Z",
        audio_sha256="a" * 64,
        duration_seconds=1.0,
        audio_url="https://example.test/audio.mp3",
        cues=(Cue(0.0, 1.0, "Speech"),),
        source_paths=(Path("ep-1.json"),),
    )
    passage = Passage(
        passage_id="p1",
        episode_id="ep-1",
        duplicate_episode_ids=("ep-1",),
        title="Discussion",
        start_seconds=0.0,
        end_seconds=1.0,
        text="Speech",
        audio_sha256="a" * 64,
        audio_url="https://example.test/audio.mp3",
        word_count=1,
        cue_count=1,
    )
    embedding = EmbeddingResult(
        model_key="qwen",
        model_id="qwen",
        model_revision=None,
        dimension=2,
        passage_ids=("p1",),
        embeddings=np.ones((1, 2)),
        cache_hits=0,
    )
    run = TopicRun(
        model_key="qwen",
        topics=np.array([0]),
        probabilities=None,
        reduced_embeddings=np.ones((1, 2)),
        topic_info=pd.DataFrame({"Topic": [0]}),
        representative_passages={0: ("p1",)},
        cluster_persistence=(1.0,),
        passage_ids=("p1",),
    )

    for record, field in (
        (episode, "duplicate_episode_ids"),
        (episode, "cues"),
        (episode, "source_paths"),
        (passage, "duplicate_episode_ids"),
        (embedding, "passage_ids"),
        (run, "cluster_persistence"),
    ):
        with pytest.raises(ValueError, match=field):
            replace(record, **{field: list(getattr(record, field))})

    with pytest.raises(ValueError, match="representative_passages"):
        replace(run, representative_passages={0: ["p1"]})


def test_representative_passage_mapping_cannot_be_mutated_after_construction() -> None:
    representatives = {0: ("p1",)}
    run = TopicRun(
        model_key="qwen",
        topics=np.array([0]),
        probabilities=None,
        reduced_embeddings=np.ones((1, 2)),
        topic_info=pd.DataFrame({"Topic": [0]}),
        representative_passages=representatives,
        cluster_persistence=(1.0,),
        passage_ids=("p1",),
    )

    representatives[0] = ("p2",)
    assert run.representative_passages[0] == ("p1",)
    with pytest.raises(TypeError):
        run.representative_passages[0] = ("p3",)  # type: ignore[index]
