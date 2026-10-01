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
from .evaluation import (
    build_boundary_rows,
    build_disagreement_rows,
    build_review_rows,
    compare_runs,
    evaluate_run,
    export_experiment,
)
from .modelling import TopicModelConfig, fit_topic_model
from .plotting import (
    plot_episode_timeline,
    plot_episode_topic_heatmap,
    plot_semantic_map,
    plot_topic_correspondence,
    plot_topic_sizes,
)
from .semantic_segmentation import (
    build_atomic_blocks,
    build_atomic_blocks_many,
    score_semantic_boundaries,
    segment_episode_semantically,
    segment_episodes_semantically,
    select_semantic_boundaries,
)
from .types import (
    CanonicalEpisode,
    ChunkingConfig,
    Cue,
    EmbeddingResult,
    ManualTopicReview,
    Passage,
    SemanticBoundary,
    SemanticSegmentationConfig,
    SemanticSegmentationResult,
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
    "SemanticBoundary",
    "SemanticSegmentationConfig",
    "SemanticSegmentationResult",
    "TopicRun",
    "TopicModelConfig",
    "available_embedders",
    "build_atomic_blocks",
    "build_atomic_blocks_many",
    "chunk_episode",
    "chunk_episodes",
    "build_review_rows",
    "build_boundary_rows",
    "build_disagreement_rows",
    "compare_runs",
    "embed_passages",
    "evaluate_run",
    "export_experiment",
    "fit_topic_model",
    "load_corpus",
    "plot_episode_timeline",
    "plot_episode_topic_heatmap",
    "plot_semantic_map",
    "plot_topic_correspondence",
    "plot_topic_sizes",
    "select_diverse_episodes",
    "score_semantic_boundaries",
    "segment_episode_semantically",
    "segment_episodes_semantically",
    "select_semantic_boundaries",
]
