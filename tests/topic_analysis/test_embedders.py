"""Offline contract tests for embedding adapters and cache orchestration."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arvamusfestivali_transcripts.topic_analysis.cache import EmbeddingCache
from arvamusfestivali_transcripts.topic_analysis.embedders import (
    BgeM3Embedder,
    DeterministicHashEmbedder,
    GeminiEmbedder,
    QwenEmbedder,
    available_embedders,
    embed_passages,
)
from arvamusfestivali_transcripts.topic_analysis.types import ChunkingConfig, Passage


def make_passage(number: int) -> Passage:
    text = f"Estonian discussion {number}"
    return Passage(
        passage_id=f"episode-{number}:0-120",
        episode_id=f"episode-{number}",
        duplicate_episode_ids=(f"episode-{number}",),
        title="Discussion",
        start_seconds=0.0,
        end_seconds=120.0,
        text=text,
        audio_sha256=f"{number:x}" * 64,
        audio_url="https://example.test/audio.mp3",
        word_count=len(text.split()),
        cue_count=1,
    )


class FakeAdapter:
    key = "fake"
    model_id = "fake/model"
    model_revision = "revision-1"
    dimension = 3
    instruction = "cluster"

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def embed_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls.append((list(texts), batch_size))
        return np.array([[len(text), 1.0, 2.0] for text in texts], dtype=np.float32)


def test_embed_passages_uses_cache_incrementally_and_preserves_order(tmp_path: Path) -> None:
    first_passage, second_passage = make_passage(1), make_passage(2)
    cache = EmbeddingCache(tmp_path)
    adapter = FakeAdapter()

    first = embed_passages(adapter, (first_passage, second_passage), cache, batch_size=2)
    second = embed_passages(adapter, (second_passage, first_passage), cache, batch_size=2)

    assert first.cache_hits == 0
    assert second.cache_hits == 2
    assert second.passage_ids == (second_passage.passage_id, first_passage.passage_id)
    assert adapter.calls == [([first_passage.text, second_passage.text], 2)]
    assert np.allclose(second.embeddings, first.embeddings[::-1])
    assert np.allclose(np.linalg.norm(first.embeddings, axis=1), 1.0)


def test_embed_passages_invalidates_changed_chunking(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    adapter = FakeAdapter()
    passage = make_passage(1)
    default = ChunkingConfig()

    embed_passages(adapter, (passage,), cache, batch_size=1, chunking=default)
    changed = embed_passages(
        adapter, (passage,), cache, batch_size=1,
        chunking=replace(default, max_words=900),
    )

    assert changed.cache_hits == 0
    assert len(adapter.calls) == 2


def test_embed_passages_keeps_valid_row_when_later_row_is_invalid(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    passages = (make_passage(1), make_passage(2))
    adapter = FakeAdapter()

    def mixed_rows(texts: list[str], batch_size: int) -> np.ndarray:
        adapter.calls.append((list(texts), batch_size))
        if len(texts) == 2:
            return np.array([[3.0, 4.0, 0.0], [np.nan, 1.0, 2.0]])
        return np.array([[0.0, 3.0, 4.0]])

    adapter.embed_texts = mixed_rows

    with pytest.raises(ValueError, match="fake"):
        embed_passages(adapter, passages, cache, batch_size=2)

    resumed = embed_passages(adapter, passages, cache, batch_size=2)

    assert resumed.cache_hits == 1
    assert adapter.calls == [
        ([passages[0].text, passages[1].text], 2),
        ([passages[1].text], 2),
    ]
    assert np.allclose(resumed.embeddings[0], [0.6, 0.8, 0.0])
    assert np.allclose(resumed.embeddings[1], [0.0, 0.6, 0.8])


def test_embed_passages_embeds_identical_cache_keys_once(tmp_path: Path) -> None:
    first = make_passage(1)
    duplicate = replace(first, passage_id="same-audio-and-text-alias")
    second = make_passage(2)
    adapter = FakeAdapter()
    adapter.key = "gemini"

    result = embed_passages(
        adapter, (first, duplicate, second), EmbeddingCache(tmp_path), batch_size=4
    )

    assert result.passage_ids == (first.passage_id, duplicate.passage_id, second.passage_id)
    assert result.cache_hits == 0
    assert np.array_equal(result.embeddings[0], result.embeddings[1])
    assert adapter.calls == [([first.text], 1), ([second.text], 1)]
    assert len(list(tmp_path.rglob("*.npy"))) == 2


@pytest.mark.parametrize("bad", [np.array([[1.0, 2.0]]), np.array([[np.nan, 1.0, 2.0]])])
def test_embed_passages_rejects_bad_adapter_rows(
    tmp_path: Path, bad: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(adapter, "embed_texts", lambda texts, batch_size: bad)

    with pytest.raises(ValueError, match="fake"):
        embed_passages(adapter, (make_passage(1),), EmbeddingCache(tmp_path), batch_size=1)
    assert list(tmp_path.rglob("*.npy")) == []


def test_available_embedders_keys_gate_gemini() -> None:
    assert set(available_embedders(env={})) == {"qwen", "bge"}
    assert set(available_embedders(env={"GEMINI_API_KEY": "secret"})) == {
        "qwen", "bge", "gemini",
    }


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    adapter = DeterministicHashEmbedder(dimension=32)
    first = adapter.embed_texts(["sama tekst"], batch_size=1)
    second = adapter.embed_texts(["sama tekst"], batch_size=1)

    assert np.array_equal(first, second)
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)


def test_local_adapters_encode_lazily_with_clustering_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[tuple[str, object]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            records.append(("init", (model_id, kwargs)))

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            records.append(("encode", (texts, kwargs)))
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    qwen = QwenEmbedder(device="cpu")
    bge = BgeM3Embedder(device="cpu")
    assert records == []

    qwen.embed_texts(["Tere"], batch_size=2)
    bge.embed_texts(["Tere"], batch_size=3)

    qwen_options = records[1][1][1]
    bge_options = records[3][1][1]
    assert records[0][1][0] == "Qwen/Qwen3-Embedding-0.6B"
    assert records[2][1][0] == "BAAI/bge-m3"
    assert qwen_options["prompt"].startswith("Represent this Estonian")
    assert qwen_options["normalize_embeddings"] is True
    assert qwen_options["batch_size"] == 2
    assert bge_options.get("prompt") is None
    assert bge_options["normalize_embeddings"] is True
    assert bge_options["batch_size"] == 3


def test_local_adapter_recommends_cpu_for_cuda_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSentenceTransformer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def encode(self, *_args: object, **_kwargs: object) -> np.ndarray:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FailingSentenceTransformer),
    )
    with pytest.raises(RuntimeError, match='device="cpu"'):
        QwenEmbedder(device="cuda").embed_texts(["Tere"], batch_size=4)


def test_gemini_prepares_clustering_instruction_and_embeds_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, object]] = []

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "secret"
            self.models = self

        def embed_content(self, *, model: str, contents: str, config: object) -> object:
            calls.append((model, contents, config))
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0] + [0.0] * 767)]
            )

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(
        Client=FakeClient,
        types=SimpleNamespace(EmbedContentConfig=lambda **kwargs: kwargs),
    )))
    monkeypatch.setitem(sys.modules, "google.genai", sys.modules["google"].genai)
    adapter = GeminiEmbedder(api_key="secret")
    assert calls == []

    vectors = adapter.embed_texts(["Tere", "Maailm"], batch_size=4)

    assert GeminiEmbedder.prepare_text("Tere maailm") == (
        "task: clustering | query: Tere maailm"
    )
    assert vectors.shape == (2, 768)
    assert [call[1] for call in calls] == [
        "task: clustering | query: Tere", "task: clustering | query: Maailm",
    ]
    assert all(call[0] == "gemini-embedding-2" for call in calls)
    assert all(call[2]["output_dimensionality"] == 768 for call in calls)


def test_gemini_failure_names_passage_and_preserves_prior_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = self

        def embed_content(self, **_kwargs: object) -> object:
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise ConnectionError("network down")
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0] + [0.0] * 767)])

    genai = SimpleNamespace(
        Client=FakeClient,
        types=SimpleNamespace(EmbedContentConfig=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    cache = EmbeddingCache(tmp_path)
    passages = (make_passage(1), make_passage(2))

    with pytest.raises(RuntimeError, match="Gemini.*episode-2:0-120"):
        embed_passages(GeminiEmbedder(api_key="secret"), passages, cache, batch_size=4)

    assert len(list(tmp_path.rglob("*.npy"))) == 1
