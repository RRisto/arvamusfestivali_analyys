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
    "available_embedders",
    "chunk_episode",
    "chunk_episodes",
    "embed_passages",
    "load_corpus",
    "select_diverse_episodes",
]
