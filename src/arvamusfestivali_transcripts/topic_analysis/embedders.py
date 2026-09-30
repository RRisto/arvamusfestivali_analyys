"""Lazy embedding adapters and incremental, content-addressed orchestration."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from .cache import CacheIdentity, EmbeddingCache
from .types import ChunkingConfig, EmbeddingResult, Passage

QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
BGE_MODEL_ID = "BAAI/bge-m3"
QWEN_INSTRUCTION = (
    "Represent this Estonian public-discussion passage for semantic topic clustering: "
)
_ADAPTER_VERSION = 1


class EmbeddingAdapter(Protocol):
    """Common shape expected by the cache orchestration."""

    key: str
    model_id: str
    model_revision: str | None
    dimension: int
    instruction: str

    def embed_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray: ...


def _normalize_rows(rows: object, count: int, dimension: int, model_key: str) -> np.ndarray:
    try:
        matrix = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{model_key} returned invalid embedding values") from error
    if matrix.shape != (count, dimension) or not np.isfinite(matrix).all():
        raise ValueError(f"{model_key} returned invalid embedding shape or non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"{model_key} returned zero or invalid embedding vectors")
    return (matrix / norms[:, None]).astype(np.float32)


def embed_passages(
    adapter: EmbeddingAdapter,
    passages: Sequence[Passage],
    cache: EmbeddingCache,
    batch_size: int,
    *,
    chunking: ChunkingConfig = ChunkingConfig(),
) -> EmbeddingResult:
    """Embed cache misses, persist each valid row, and retain passage order."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not passages:
        raise ValueError("passages must be nonempty")
    identity = CacheIdentity(
        model_id=adapter.model_id,
        model_revision=adapter.model_revision,
        dimension=adapter.dimension,
        instruction=adapter.instruction,
        chunking=chunking,
        adapter_version=_ADAPTER_VERSION,
    )
    rows = list(cache.assemble(identity, passages))
    cache_hits = sum(row is not None for row in rows)
    missing = [index for index, row in enumerate(rows) if row is None]
    effective_batch_size = 1 if adapter.key == "gemini" else batch_size
    for offset in range(0, len(missing), effective_batch_size):
        indices = missing[offset:offset + effective_batch_size]
        texts = [passages[index].text for index in indices]
        try:
            generated = adapter.embed_texts(texts, batch_size=effective_batch_size)
        except Exception as error:
            if adapter.key == "gemini":
                passage_id = passages[indices[0]].passage_id
                raise RuntimeError(
                    f"Gemini embedding failed for passage {passage_id}: {error}"
                ) from error
            raise
        normalized = _normalize_rows(generated, len(indices), adapter.dimension, adapter.key)
        for index, vector in zip(indices, normalized, strict=True):
            cache.put(identity, passages[index], vector)
            rows[index] = vector
    return EmbeddingResult(
        model_key=adapter.key,
        model_id=adapter.model_id,
        model_revision=adapter.model_revision,
        dimension=adapter.dimension,
        passage_ids=tuple(passage.passage_id for passage in passages),
        embeddings=np.stack(rows).astype(np.float32),
        cache_hits=cache_hits,
    )


class DeterministicHashEmbedder:
    """Stable synthetic vectors for offline integration tests and notebook dry runs."""

    key = "fake"
    model_id = "deterministic-hash"
    model_revision = None
    instruction = "deterministic test embedding"

    def __init__(self, dimension: int = 32) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            row = np.random.default_rng(seed).standard_normal(self.dimension).astype(np.float32)
            rows.append(row)
        return _normalize_rows(rows, len(texts), self.dimension, self.key)


class _LocalEmbedder:
    """Shared lazy SentenceTransformer loading and CUDA memory guidance."""

    key: str
    model_id: str
    dimension = 1024
    instruction: str

    def __init__(self, *, device: str = "cpu", model_revision: str | None = None) -> None:
        self.device = device
        self.model_revision = model_revision
        self._model: object | None = None

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            options: dict[str, str] = {"device": self.device}
            if self.model_revision is not None:
                options["revision"] = self.model_revision
            self._model = SentenceTransformer(self.model_id, **options)
        return self._model

    def embed_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        options: dict[str, object] = {
            "batch_size": batch_size,
            "device": self.device,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if self.instruction:
            options["prompt"] = self.instruction
        try:
            model = self._load_model()
            return np.asarray(model.encode(list(texts), **options), dtype=np.float32)
        except RuntimeError as error:
            message = str(error).lower()
            if "cuda" in message and "out of memory" in message:
                raise RuntimeError(
                    f"{self.key} CUDA out of memory; retry with device=\"cpu\" "
                    "or a smaller batch size"
                ) from error
            raise


class QwenEmbedder(_LocalEmbedder):
    key = "qwen"
    model_id = QWEN_MODEL_ID
    instruction = QWEN_INSTRUCTION


class BgeM3Embedder(_LocalEmbedder):
    key = "bge"
    model_id = BGE_MODEL_ID
    instruction = ""


class GeminiEmbedder:
    """Optional cloud adapter; callers decide which passages may leave the machine."""

    key = "gemini"
    model_id = "gemini-embedding-2"
    model_revision = None
    dimension = 768
    instruction = "task: clustering | query: "

    def __init__(self, *, api_key: str) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.api_key = api_key
        self._client: object | None = None

    @staticmethod
    def prepare_text(text: str) -> str:
        return f"task: clustering | query: {text}"

    def embed_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        from google import genai

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        rows = []
        for text in texts:
            response = self._client.models.embed_content(
                model=self.model_id,
                contents=self.prepare_text(text),
                config=genai.types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None or len(embeddings) != 1:
                raise ValueError("Gemini returned no single embedding vector")
            rows.append(embeddings[0].values)
        return _normalize_rows(rows, len(texts), self.dimension, self.key)


def available_embedders(env: Mapping[str, str] = os.environ) -> Mapping[str, EmbeddingAdapter]:
    """Provide local models and include Gemini only when a key is configured."""
    adapters: dict[str, EmbeddingAdapter] = {
        "qwen": QwenEmbedder(),
        "bge": BgeM3Embedder(),
    }
    key = env.get("GEMINI_API_KEY", "").strip()
    if key:
        adapters["gemini"] = GeminiEmbedder(api_key=key)
    return adapters
