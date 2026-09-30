"""Shared contracts for transcript topic analysis."""

from .chunking import chunk_episode, chunk_episodes
from .corpus import load_corpus, select_diverse_episodes
from .embedders import (
    BgeM3Embedder,
    DeterministicHashEmbedder,
    EmbeddingAdapter,
    GeminiEmbedder,
    QwenEmbedder,
    available_embedders,
    embed_passages,
)
from .evaluation import build_review_rows, compare_runs, evaluate_run, export_experiment
from .modelling import TopicModelConfig, fit_topic_model
from .types import (
    CanonicalEpisode,
    ChunkingConfig,
    Cue,
    EmbeddingResult,
    ManualTopicReview,
    Passage,
    TopicRun,
)

__all__ = [
    "CanonicalEpisode",
    "BgeM3Embedder",
    "ChunkingConfig",
    "Cue",
    "DeterministicHashEmbedder",
    "EmbeddingAdapter",
    "EmbeddingResult",
    "GeminiEmbedder",
    "ManualTopicReview",
    "Passage",
    "QwenEmbedder",
    "TopicRun",
    "TopicModelConfig",
    "available_embedders",
    "chunk_episode",
    "chunk_episodes",
    "build_review_rows",
    "compare_runs",
    "embed_passages",
    "evaluate_run",
    "export_experiment",
    "fit_topic_model",
    "load_corpus",
    "select_diverse_episodes",
]
