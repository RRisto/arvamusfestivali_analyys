"""Content-addressed cache for individual normalized passage embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .types import ChunkingConfig, Passage


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """Configuration that can change the meaning or shape of an embedding."""

    model_id: str
    model_revision: str | None
    dimension: int
    instruction: str
    chunking: ChunkingConfig
    adapter_version: int

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _passage_digest(passage: Passage) -> str:
    normalized_text = unicodedata.normalize("NFC", " ".join(passage.text.split()))
    payload = json.dumps(
        {
            "audio_sha256": passage.audio_sha256,
            "start_seconds": passage.start_seconds,
            "end_seconds": passage.end_seconds,
            "text": normalized_text,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_vector(vector: np.ndarray, dimension: int, path: Path) -> None:
    if (
        not isinstance(vector, np.ndarray)
        or vector.shape != (dimension,)
        or not np.issubdtype(vector.dtype, np.floating)
        or not np.isfinite(vector).all()
        or abs(float(np.linalg.norm(vector.astype(np.float64))) - 1.0) > 1e-4
    ):
        raise ValueError(f"corrupt embedding cache at {path}: invalid vector")


class EmbeddingCache:
    """Read and atomically write a separate NumPy file for each passage."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, identity: CacheIdentity, passage: Passage) -> Path:
        return self.root / identity.digest / f"{_passage_digest(passage)}.npy"

    def get(self, identity: CacheIdentity, passage: Passage) -> np.ndarray | None:
        path = self._path(identity, passage)
        try:
            with path.open("rb") as source:
                vector = np.load(source, allow_pickle=False)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError) as error:
            raise ValueError(f"corrupt embedding cache at {path}: {error}") from error
        _validate_vector(vector, identity.dimension, path)
        return vector

    def put(self, identity: CacheIdentity, passage: Passage, vector: np.ndarray) -> Path:
        path = self._path(identity, passage)
        _validate_vector(vector, identity.dimension, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.stem}.", suffix=".npy", dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.save(temporary, vector, allow_pickle=False)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def assemble(
        self, identity: CacheIdentity, passages: Sequence[Passage]
    ) -> tuple[np.ndarray | None, ...]:
        """Return cached vectors in passage order, with None for absent files."""
        return tuple(self.get(identity, passage) for passage in passages)
