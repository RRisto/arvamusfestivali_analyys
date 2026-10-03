"""Fit BERTopic with one controlled configuration per embedding space."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .types import EmbeddingResult, Passage, TopicRun


@dataclass(frozen=True, slots=True)
class TopicModelConfig:
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    min_cluster_size: int = 10
    min_samples: int = 5
    top_n_words: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_neighbors", 2),
            ("n_components", 1),
            ("min_cluster_size", 2),
            ("min_samples", 1),
            ("top_n_words", 1),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if not isinstance(self.min_dist, int | float) or not math.isfinite(self.min_dist) or not (
            0 <= self.min_dist <= 1
        ):
            raise ValueError("min_dist must be finite and between zero and one")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError("random_state must be an integer")


def _model_classes() -> tuple[type, type, type, type]:
    """Load substantial optional modelling packages only when fitting."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    return UMAP, HDBSCAN, CountVectorizer, BERTopic


def _umap_options(config: TopicModelConfig, dimensions: int) -> dict[str, object]:
    return {
        "n_neighbors": config.n_neighbors,
        "n_components": dimensions,
        "min_dist": config.min_dist,
        "metric": "cosine",
        "random_state": config.random_state,
    }


def _build_bertopic(config: TopicModelConfig) -> object:
    umap_class, hdbscan_class, vectorizer_class, bertopic_class = _model_classes()
    return bertopic_class(
        language="multilingual",
        embedding_model=None,
        umap_model=umap_class(**_umap_options(config, config.n_components)),
        hdbscan_model=hdbscan_class(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        # BERTopic vectorizes aggregated topics, including the single all-outlier topic.
        vectorizer_model=vectorizer_class(ngram_range=(1, 3), min_df=1),
        top_n_words=config.top_n_words,
        calculate_probabilities=True,
    )


def _reduce_for_display(rows: np.ndarray, config: TopicModelConfig) -> np.ndarray:
    umap_class = _model_classes()[0]
    return np.asarray(umap_class(**_umap_options(config, 2)).fit_transform(rows), dtype=float)


def _representative_ids(
    passages: Sequence[Passage], topics: np.ndarray, model: object
) -> dict[int, tuple[str, ...]]:
    available: dict[tuple[int, str], list[str]] = defaultdict(list)
    for passage, topic in zip(passages, topics, strict=True):
        available[(int(topic), passage.text)].append(passage.passage_id)
    used: set[tuple[int, str]] = set()
    mapped: dict[int, tuple[str, ...]] = {}
    representative_docs = model.get_representative_docs() or {}
    for topic, documents in representative_docs.items():
        ids = []
        for text in documents or ():
            key = (int(topic), text)
            candidates = available[key]
            if len(candidates) != 1 or key in used:
                raise ValueError(
                    f"ambiguous representative document for topic {topic}: "
                    "text does not identify exactly one passage"
                )
            ids.append(candidates[0])
            used.add(key)
        mapped[int(topic)] = tuple(ids)
    return mapped


def fit_topic_model(
    passages: Sequence[Passage], embeddings: EmbeddingResult, config: TopicModelConfig
) -> TopicRun:
    """Fit with precomputed rows and retain exact passage provenance."""
    passage_ids = tuple(passage.passage_id for passage in passages)
    if not passage_ids or len(set(passage_ids)) != len(passage_ids):
        raise ValueError("passages must have unique nonempty IDs")
    if passage_ids != embeddings.passage_ids:
        raise ValueError("embedding passage IDs and order must match passages")
    model = _build_bertopic(config)
    documents = [passage.text for passage in passages]
    topics, probabilities = model.fit_transform(documents, embeddings=embeddings.embeddings)
    topic_array = np.asarray(topics, dtype=int)
    if topic_array.shape != (len(passages),):
        raise ValueError("BERTopic returned a topic count different from the passage count")
    projection = _reduce_for_display(embeddings.embeddings, config)
    cluster_model = getattr(model, "hdbscan_model", None)
    persistence = getattr(cluster_model, "cluster_persistence_", ())
    probability_array = None if probabilities is None else np.asarray(probabilities)
    return TopicRun(
        model_key=embeddings.model_key,
        topics=topic_array,
        probabilities=probability_array,
        # BERTopic maps membership columns to its final contiguous non-outlier IDs.
        probability_topic_ids=(
            tuple(range(probability_array.shape[1]))
            if probability_array is not None and probability_array.ndim == 2 else ()
        ),
        reduced_embeddings=projection,
        # BERTopic passes nan_to_num(UMAP.fit_transform(...)) to HDBSCAN.
        clustering_embeddings=np.nan_to_num(np.asarray(model.umap_model.embedding_, dtype=float)),
        topic_info=model.get_topic_info().copy(),
        representative_passages=_representative_ids(passages, topic_array, model),
        cluster_persistence=tuple(float(value) for value in persistence),
        passage_ids=passage_ids,
        topic_model_config=asdict(config),
    )
