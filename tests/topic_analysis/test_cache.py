"""Stable and recoverable per-passage embedding cache behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arvamusfestivali_transcripts.topic_analysis import (
    CanonicalEpisode,
    Cue,
    DeterministicHashEmbedder,
    SemanticSegmentationConfig,
    build_atomic_blocks,
    embed_passages,
)
from arvamusfestivali_transcripts.topic_analysis import cache as cache_module
from arvamusfestivali_transcripts.topic_analysis.cache import CacheIdentity, EmbeddingCache
from arvamusfestivali_transcripts.topic_analysis.types import ChunkingConfig, Passage


def make_identity(*, dimension: int = 3) -> CacheIdentity:
    return CacheIdentity(
        model_id="model",
        model_revision="rev",
        dimension=dimension,
        instruction="cluster Estonian debates",
        chunking=ChunkingConfig(),
        adapter_version=1,
    )


def make_passage(*, text: str = "A public discussion.") -> Passage:
    return Passage(
        passage_id="episode-1:0-120",
        episode_id="episode-1",
        duplicate_episode_ids=("episode-1",),
        title="Discussion",
        start_seconds=0.0,
        end_seconds=120.0,
        text=text,
        audio_sha256="a" * 64,
        audio_url="https://example.test/audio.mp3",
        word_count=len(text.split()),
        cue_count=1,
    )


def test_cache_identity_changes_with_instruction_or_chunking() -> None:
    base = CacheIdentity(
        model_id="model",
        model_revision="rev",
        dimension=3,
        instruction="cluster Estonian debates",
        chunking=ChunkingConfig(),
        adapter_version=1,
    )

    assert base.digest != replace(base, instruction="search Estonian debates").digest
    assert base.digest != replace(base, chunking=replace(base.chunking, max_words=900)).digest
    assert base.digest != replace(base, model_id="other-model").digest
    assert base.digest != replace(base, model_revision="next-rev").digest
    assert base.digest != replace(base, dimension=4).digest
    assert base.digest != replace(base, adapter_version=2).digest


def test_cache_round_trip_and_corruption_rejection(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    identity = make_identity()
    passage = make_passage()
    vector = np.array([0.0, 0.6, 0.8], dtype=np.float32)

    path = cache.put(identity, passage, vector)
    assert path.parent == tmp_path / identity.digest
    assert np.allclose(cache.get(identity, passage), vector)

    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt embedding cache") as error:
        cache.get(identity, passage)
    assert str(path) in str(error.value)


def test_assemble_preserves_passage_order_and_reports_missing_entries(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    identity = make_identity()
    first = make_passage(text="First passage")
    second = make_passage(text="Second passage")
    vector = np.array([0.0, 0.6, 0.8], dtype=np.float32)
    cache.put(identity, second, vector)

    assembled = cache.assemble(identity, (first, second))

    assert assembled[0] is None
    assert np.array_equal(assembled[1], vector)


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([[0.0, 0.6, 0.8]], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([np.nan, 0.6, 0.8], dtype=np.float32),
        np.array([0.0, 0.3, 0.4], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.int32),
    ],
)
def test_get_rejects_invalid_stored_vectors_with_path(
    tmp_path: Path, bad_vector: np.ndarray
) -> None:
    cache = EmbeddingCache(tmp_path)
    identity = make_identity()
    passage = make_passage()
    path = cache.put(identity, passage, np.array([0.0, 0.6, 0.8], dtype=np.float32))
    with path.open("wb") as destination:
        np.save(destination, bad_vector)

    with pytest.raises(ValueError, match="corrupt embedding cache") as error:
        cache.get(identity, passage)
    assert str(path) in str(error.value)


def test_put_rejects_invalid_vector_without_creating_cache_file(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    identity = make_identity()
    passage = make_passage()

    with pytest.raises(ValueError, match="corrupt embedding cache") as error:
        cache.put(identity, passage, np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert str(tmp_path) in str(error.value)
    assert list(tmp_path.rglob("*.npy")) == []


def test_passage_path_uses_normalized_text_sha_and_timestamps(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    identity = make_identity()
    passage = make_passage(text="Public  discussion")
    vector = np.array([0.0, 0.6, 0.8], dtype=np.float32)

    path = cache.put(identity, passage, vector)

    assert cache.get(identity, replace(passage, text="Public discussion")) is not None
    assert cache.get(identity, replace(passage, text="Different discussion")) is None
    assert cache.get(identity, replace(passage, audio_sha256="b" * 64)) is None
    assert cache.get(identity, replace(passage, start_seconds=1.0)) is None
    assert cache.get(identity, replace(passage, end_seconds=121.0)) is None
    assert path.suffix == ".npy"


def test_atomic_replacement_keeps_previous_file_if_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = EmbeddingCache(tmp_path)
    identity = make_identity()
    passage = make_passage()
    original = np.array([0.0, 0.6, 0.8], dtype=np.float32)
    replacement = np.array([0.0, 0.8, 0.6], dtype=np.float32)
    path = cache.put(identity, passage, original)

    def fail_replace(source: Path, destination: Path) -> None:
        assert source.parent == destination.parent == path.parent
        assert np.array_equal(np.load(source), replacement)
        assert np.array_equal(np.load(destination), original)
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interrupted replacement"):
        cache.put(identity, passage, replacement)

    assert np.array_equal(cache.get(identity, passage), original)
    assert list(path.parent.iterdir()) == [path]


def test_atomic_block_embeddings_are_reused_from_cache(tmp_path: Path) -> None:
    config = SemanticSegmentationConfig()
    cues = tuple(
        Cue(index * 10, (index + 1) * 10, f"Cue number {index}")
        for index in range(9)
    )
    episode = CanonicalEpisode(
        episode_id="episode-1",
        duplicate_episode_ids=("episode-1",),
        title="Discussion",
        published_at="2026-08-01T12:00:00Z",
        audio_sha256="a" * 64,
        duration_seconds=90,
        audio_url="https://example.test/audio.mp3",
        cues=cues,
        source_paths=(Path("episode.json"),),
    )
    blocks = build_atomic_blocks(episode, config)
    cache = EmbeddingCache(tmp_path)
    adapter = DeterministicHashEmbedder(dimension=8)

    first = embed_passages(
        adapter,
        blocks,
        cache,
        batch_size=2,
        chunking=config.atomic_chunking,
    )
    second = embed_passages(
        adapter,
        blocks,
        cache,
        batch_size=2,
        chunking=config.atomic_chunking,
    )

    assert first.cache_hits == 0
    assert second.cache_hits == len(blocks)
    assert np.array_equal(first.embeddings, second.embeddings)
