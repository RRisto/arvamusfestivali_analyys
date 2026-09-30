"""Shared contracts for transcript topic analysis."""

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
    "load_corpus",
    "select_diverse_episodes",
]
