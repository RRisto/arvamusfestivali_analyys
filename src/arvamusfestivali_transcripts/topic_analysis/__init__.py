"""Shared contracts for transcript topic analysis."""

from .chunking import chunk_episode, chunk_episodes
from .corpus import load_corpus, select_diverse_episodes
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
    "ChunkingConfig",
    "Cue",
    "EmbeddingResult",
    "ManualTopicReview",
    "Passage",
    "TopicRun",
    "chunk_episode",
    "chunk_episodes",
    "load_corpus",
    "select_diverse_episodes",
]
