"""Validated, immutable records shared by the topic-analysis modules."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _require_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a tuple of nonempty IDs")


def _require_interval(start: float, end: float, name: str) -> None:
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int | float)
        or not isinstance(end, int | float)
        or not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"{name} interval must be finite, nonnegative, and ordered")


def _require_sha256(value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("audio_sha256 must be a lowercase 64-character hexadecimal digest")


def _require_positive_count(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class Cue:
    start_seconds: float
    end_seconds: float
    text: str

    def __post_init__(self) -> None:
        _require_interval(self.start_seconds, self.end_seconds, "cue")
        _require_text(self.text, "cue text")


@dataclass(frozen=True, slots=True)
class CanonicalEpisode:
    episode_id: str
    duplicate_episode_ids: tuple[str, ...]
    title: str
    published_at: str
    audio_sha256: str
    duration_seconds: float
    audio_url: str
    cues: tuple[Cue, ...]
    source_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        _require_text(self.episode_id, "episode_id")
        _require_ids(self.duplicate_episode_ids, "duplicate_episode_ids")
        _require_text(self.title, "title")
        _require_text(self.published_at, "published_at")
        _require_sha256(self.audio_sha256)
        if not isinstance(self.duration_seconds, int | float) or not math.isfinite(
            self.duration_seconds
        ) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive and finite")
        _require_text(self.audio_url, "audio_url")
        if not isinstance(self.cues, tuple) or not self.cues or any(
            not isinstance(cue, Cue) for cue in self.cues
        ):
            raise ValueError("cues must be a tuple of Cue records")
        if not isinstance(self.source_paths, tuple) or not self.source_paths or any(
            not isinstance(path, Path) for path in self.source_paths
        ):
            raise ValueError("source_paths must be a tuple of paths")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    min_seconds: float = 120.0
    target_seconds: float = 180.0
    max_seconds: float = 240.0
    min_words: int = 400
    max_words: int = 800
    overlap_seconds: float = 15.0
    merge_tail_seconds: float = 60.0
    max_merged_seconds: float = 300.0

    def __post_init__(self) -> None:
        durations = (self.min_seconds, self.target_seconds, self.max_seconds)
        if (
            any(
                not isinstance(value, int | float) or not math.isfinite(value)
                for value in durations
            )
            or not 0 < self.min_seconds <= self.target_seconds <= self.max_seconds
        ):
            raise ValueError("chunk duration bounds must be finite, positive, and ordered")
        _require_positive_count(self.min_words, "min_words")
        _require_positive_count(self.max_words, "max_words")
        if self.min_words > self.max_words:
            raise ValueError("chunk word bounds must be ordered")
        if (
            any(
                not isinstance(value, int | float) or not math.isfinite(value) or value < 0
                for value in (self.overlap_seconds, self.merge_tail_seconds)
            )
            or not isinstance(self.max_merged_seconds, int | float)
            or not math.isfinite(self.max_merged_seconds)
            or self.max_merged_seconds < self.max_seconds
        ):
            raise ValueError("chunk overlap and merge bounds must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class Passage:
    passage_id: str
    episode_id: str
    duplicate_episode_ids: tuple[str, ...]
    title: str
    start_seconds: float
    end_seconds: float
    text: str
    audio_sha256: str
    audio_url: str
    word_count: int
    cue_count: int

    def __post_init__(self) -> None:
        _require_text(self.passage_id, "passage_id")
        _require_text(self.episode_id, "episode_id")
        _require_ids(self.duplicate_episode_ids, "duplicate_episode_ids")
        _require_text(self.title, "title")
        _require_interval(self.start_seconds, self.end_seconds, "passage")
        _require_text(self.text, "passage text")
        _require_sha256(self.audio_sha256)
        _require_text(self.audio_url, "audio_url")
        _require_positive_count(self.word_count, "word_count")
        _require_positive_count(self.cue_count, "cue_count")

    @property
    def timestamped_audio_url(self) -> str:
        start = str(self.start_seconds).removesuffix(".0")
        return f"{self.audio_url}#t={start}"


@dataclass(frozen=True, slots=True)
class SemanticSegmentationConfig:
    atomic_target_seconds: float = 30.0
    atomic_max_seconds: float = 45.0
    context_seconds: float = 60.0
    min_segment_seconds: float = 90.0
    max_segment_seconds: float = 600.0
    boundary_quantile: float = 0.85

    def __post_init__(self) -> None:
        values = (
            self.atomic_target_seconds,
            self.atomic_max_seconds,
            self.context_seconds,
            self.min_segment_seconds,
            self.max_segment_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError("semantic segmentation durations must be finite and positive")
        if self.atomic_target_seconds > self.atomic_max_seconds:
            raise ValueError("atomic block durations must be ordered")
        if self.min_segment_seconds > self.max_segment_seconds:
            raise ValueError("semantic segment durations must be ordered")
        if (
            isinstance(self.boundary_quantile, bool)
            or not isinstance(self.boundary_quantile, int | float)
            or not math.isfinite(self.boundary_quantile)
            or not 0 <= self.boundary_quantile <= 1
        ):
            raise ValueError("boundary_quantile must be between zero and one")

    @property
    def atomic_chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            min_seconds=self.atomic_target_seconds,
            target_seconds=self.atomic_target_seconds,
            max_seconds=self.atomic_max_seconds,
            min_words=1,
            max_words=1_000_000,
            overlap_seconds=0,
            merge_tail_seconds=0,
            max_merged_seconds=self.atomic_max_seconds,
        )


@dataclass(frozen=True, slots=True)
class SemanticBoundary:
    episode_id: str
    timestamp_seconds: float
    left_block_id: str
    right_block_id: str
    score: float
    selected: bool = False
    forced: bool = False

    def __post_init__(self) -> None:
        _require_text(self.episode_id, "episode_id")
        _require_text(self.left_block_id, "left_block_id")
        _require_text(self.right_block_id, "right_block_id")
        for value, name in (
            (self.timestamp_seconds, "boundary timestamp"),
            (self.score, "boundary score"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.selected, bool) or not isinstance(self.forced, bool):
            raise ValueError("boundary flags must be boolean")
        if self.forced and not self.selected:
            raise ValueError("forced boundary must also be selected")


@dataclass(frozen=True, slots=True)
class SemanticSegmentationResult:
    passages: tuple[Passage, ...]
    boundaries: tuple[SemanticBoundary, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.passages, tuple)
            or not self.passages
            or any(not isinstance(item, Passage) for item in self.passages)
        ):
            raise ValueError("passages must be a nonempty tuple of Passage records")
        if not isinstance(self.boundaries, tuple) or any(
            not isinstance(item, SemanticBoundary) for item in self.boundaries
        ):
            raise ValueError("boundaries must be a tuple of SemanticBoundary records")
        ids = tuple(item.passage_id for item in self.passages)
        if len(set(ids)) != len(ids):
            raise ValueError("passages must contain unique IDs")
        last_by_episode: dict[str, float] = {}
        for passage in self.passages:
            last_end = last_by_episode.get(passage.episode_id)
            if last_end is not None and passage.start_seconds < last_end:
                raise ValueError("passages must be ordered and nonoverlapping per episode")
            last_by_episode[passage.episode_id] = passage.end_seconds


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    model_key: str
    model_id: str
    model_revision: str | None
    dimension: int
    passage_ids: tuple[str, ...]
    embeddings: np.ndarray
    cache_hits: int

    def __post_init__(self) -> None:
        _require_text(self.model_key, "model_key")
        _require_text(self.model_id, "model_id")
        if self.model_revision is not None:
            _require_text(self.model_revision, "model_revision")
        _require_positive_count(self.dimension, "dimension")
        _require_ids(self.passage_ids, "passage_ids")
        if (
            not isinstance(self.embeddings, np.ndarray)
            or self.embeddings.ndim != 2
            or self.embeddings.shape != (len(self.passage_ids), self.dimension)
            or not np.issubdtype(self.embeddings.dtype, np.number)
            or not np.isfinite(self.embeddings).all()
        ):
            raise ValueError("embedding rows must match passage IDs and dimension and be finite")
        if (
            not isinstance(self.cache_hits, int)
            or isinstance(self.cache_hits, bool)
            or not 0 <= self.cache_hits <= len(self.passage_ids)
        ):
            raise ValueError("cache_hits must be between zero and the passage count")


@dataclass(frozen=True, slots=True)
class TopicRun:
    model_key: str
    topics: np.ndarray
    probabilities: np.ndarray | None
    reduced_embeddings: np.ndarray
    topic_info: pd.DataFrame
    representative_passages: Mapping[int, tuple[str, ...]]
    cluster_persistence: tuple[float, ...]
    passage_ids: tuple[str, ...]
    topic_model_config: Mapping[str, int | float] | None = None
    probability_topic_ids: tuple[int, ...] = ()
    # reduced_embeddings is the separate 2-D display map retained for API compatibility.
    clustering_embeddings: np.ndarray | None = None

    def __post_init__(self) -> None:
        _require_text(self.model_key, "model_key")
        if not isinstance(self.topics, np.ndarray) or self.topics.ndim != 1:
            raise ValueError("topics must be a one-dimensional array")
        count = len(self.topics)
        if (
            self.probabilities is not None
            and (
                not isinstance(self.probabilities, np.ndarray)
                or self.probabilities.ndim not in (1, 2)
                or self.probabilities.shape[0] != count
            )
        ):
            raise ValueError("probability rows must match topics")
        if self.probabilities is not None and (
            not np.isfinite(self.probabilities).all()
            or np.any(self.probabilities < 0) or np.any(self.probabilities > 1)
        ):
            raise ValueError("probabilities must be finite and between zero and one")
        if self.probabilities is not None and self.probabilities.ndim == 2:
            if (
                not isinstance(self.probability_topic_ids, tuple)
                or len(self.probability_topic_ids) != self.probabilities.shape[1]
                or len(set(self.probability_topic_ids)) != len(self.probability_topic_ids)
                or any(not isinstance(t, int) or t < 0 for t in self.probability_topic_ids)
                or not set(self.topics[self.topics != -1]).issubset(self.probability_topic_ids)
            ):
                raise ValueError("probability_topic_ids must map every matrix column to a topic")
        elif self.probability_topic_ids:
            raise ValueError("probability_topic_ids requires a probability matrix")
        if (
            not isinstance(self.reduced_embeddings, np.ndarray)
            or self.reduced_embeddings.ndim != 2
            or self.reduced_embeddings.shape[0] != count
        ):
            raise ValueError("reduced embedding rows must match topics")
        if self.clustering_embeddings is not None and (
            not isinstance(self.clustering_embeddings, np.ndarray)
            or self.clustering_embeddings.ndim != 2
            or self.clustering_embeddings.shape[0] != count
            or self.clustering_embeddings.shape[1] < 1
            or not np.isfinite(self.clustering_embeddings).all()
        ):
            raise ValueError("clustering embedding rows must match topics and be finite")
        if not isinstance(self.topic_info, pd.DataFrame):
            raise ValueError("topic_info must be a DataFrame")
        if not isinstance(self.representative_passages, Mapping) or any(
            not isinstance(ids, tuple) for ids in self.representative_passages.values()
        ):
            raise ValueError("representative_passages must map to tuples")
        object.__setattr__(
            self, "representative_passages", MappingProxyType(dict(self.representative_passages))
        )
        if not isinstance(self.cluster_persistence, tuple) or any(
            not math.isfinite(value) for value in self.cluster_persistence
        ):
            raise ValueError("cluster_persistence must be a tuple of finite values")
        if (
            not isinstance(self.passage_ids, tuple)
            or len(self.passage_ids) != count
            or not self.passage_ids
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.passage_ids
            )
            or len(set(self.passage_ids)) != count
        ):
            raise ValueError("passage_ids must match topics and contain unique nonempty IDs")
        if self.topic_model_config is not None:
            if (
                not isinstance(self.topic_model_config, Mapping)
                or not self.topic_model_config
                or any(
                    not isinstance(key, str)
                    or not key
                    or isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    for key, value in self.topic_model_config.items()
                )
            ):
                raise ValueError("topic_model_config must map names to finite numbers")
            object.__setattr__(
                self, "topic_model_config", MappingProxyType(dict(self.topic_model_config))
            )


@dataclass(frozen=True, slots=True)
class ManualTopicReview:
    model_key: str
    topic_id: int
    verdict: Literal["coherent", "mixed", "duplicate", "unclear"]
    note: str = ""

    def __post_init__(self) -> None:
        _require_text(self.model_key, "model_key")
        if not isinstance(self.topic_id, int) or isinstance(self.topic_id, bool):
            raise ValueError("topic_id must be an integer")
        if self.verdict not in ("coherent", "mixed", "duplicate", "unclear"):
            raise ValueError("verdict must be coherent, mixed, duplicate, or unclear")
        if not isinstance(self.note, str):
            raise ValueError("note must be text")
