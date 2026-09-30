"""Controlled topic fitting without loading real model weights."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from arvamusfestivali_transcripts.topic_analysis import modelling
from arvamusfestivali_transcripts.topic_analysis.types import EmbeddingResult, Passage


def _passages(texts: tuple[str, ...]) -> tuple[Passage, ...]:
    return tuple(
        Passage(
            passage_id=f"p{index}",
            episode_id="episode-1",
            duplicate_episode_ids=("episode-1",),
            title="Discussion",
            start_seconds=float(index * 10),
            end_seconds=float(index * 10 + 10),
            text=text,
            audio_sha256="a" * 64,
            audio_url="https://example.test/audio.mp3",
            word_count=len(text.split()),
            cue_count=1,
        )
        for index, text in enumerate(texts)
    )


def _embeddings(passages: tuple[Passage, ...]) -> EmbeddingResult:
    rows = np.eye(len(passages), dtype=np.float32)
    return EmbeddingResult(
        model_key="qwen",
        model_id="qwen-id",
        model_revision=None,
        dimension=len(passages),
        passage_ids=tuple(p.passage_id for p in passages),
        embeddings=rows,
        cache_hits=0,
    )


class FakeBERTopic:
    def __init__(self, topics: list[int], representatives: dict[int, list[str]]) -> None:
        self.topics = topics
        self.representatives = representatives
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.hdbscan_model = type("Cluster", (), {"cluster_persistence_": np.array([0.6])})()

    def fit_transform(
        self, documents: list[str], embeddings: np.ndarray
    ) -> tuple[list[int], np.ndarray]:
        self.documents = documents
        self.embeddings = embeddings
        return self.topics, np.full((len(documents), 2), 0.5)

    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame({"Topic": [-1, 0, 1], "Count": [1, 1, 1]})

    def get_representative_docs(self) -> dict[int, list[str]]:
        return self.representatives


def test_fit_topic_model_preserves_order_and_maps_duplicate_text_by_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passages = _passages(("repeated text", "repeated text", "other text"))
    embedding = _embeddings(passages)
    fake = FakeBERTopic([1, 0, -1], {1: ["repeated text"], 0: ["repeated text"]})
    monkeypatch.setattr(modelling, "_build_bertopic", lambda config: fake)
    monkeypatch.setattr(
        modelling,
        "_reduce_for_display",
        lambda rows, config: np.arange(len(rows) * 2).reshape(len(rows), 2),
    )

    config = modelling.TopicModelConfig(random_state=17)
    run = modelling.fit_topic_model(passages, embedding, config)

    assert fake.documents == ["repeated text", "repeated text", "other text"]
    assert fake.embeddings is embedding.embeddings
    assert run.model_key == "qwen"
    assert run.passage_ids == ("p0", "p1", "p2")
    assert run.topics.tolist() == [1, 0, -1]
    assert run.representative_passages == {1: ("p0",), 0: ("p1",)}
    assert run.reduced_embeddings.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert run.topic_model_config == asdict(config)


def test_fit_topic_model_rejects_embedding_order_mismatch() -> None:
    passages = _passages(("one", "two"))
    embedding = _embeddings(passages)
    reordered = (passages[1], passages[0])

    with pytest.raises(ValueError, match="passage IDs and order"):
        modelling.fit_topic_model(reordered, embedding, modelling.TopicModelConfig())


def test_fit_topic_model_rejects_unmatched_representative_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passages = _passages(("one", "two", "three"))
    fake = FakeBERTopic([0, 0, -1], {0: ["absent"]})
    monkeypatch.setattr(modelling, "_build_bertopic", lambda config: fake)
    monkeypatch.setattr(
        modelling, "_reduce_for_display", lambda rows, config: np.zeros((len(rows), 2))
    )

    with pytest.raises(ValueError, match="representative document"):
        modelling.fit_topic_model(passages, _embeddings(passages), modelling.TopicModelConfig())


def test_fit_topic_model_rejects_ambiguous_duplicate_representatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passages = _passages(("repeated text", "repeated text", "other text"))
    fake = FakeBERTopic([0, 0, -1], {0: ["repeated text"]})
    monkeypatch.setattr(modelling, "_build_bertopic", lambda config: fake)
    monkeypatch.setattr(
        modelling, "_reduce_for_display", lambda rows, config: np.zeros((len(rows), 2))
    )

    with pytest.raises(ValueError, match="ambiguous representative document"):
        modelling.fit_topic_model(passages, _embeddings(passages), modelling.TopicModelConfig())


def test_builder_controls_umap_hdbscan_and_vectorizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[dict[str, object]]] = defaultdict(list)

    def constructor(name: str):
        def build(**kwargs: object) -> object:
            captured[name].append(kwargs)
            return type(
                "Component",
                (),
                {"kwargs": kwargs, "fit_transform": lambda self, rows: np.zeros((len(rows), 2))},
            )()

        return build

    monkeypatch.setattr(
        modelling,
        "_model_classes",
        lambda: tuple(constructor(name) for name in ("umap", "hdbscan", "vectorizer", "bertopic")),
    )
    config = modelling.TopicModelConfig(random_state=17, min_cluster_size=8)

    modelling._build_bertopic(config)
    modelling._reduce_for_display(np.eye(3), config)

    assert captured["umap"][0]["n_components"] == 5
    assert captured["umap"][0]["metric"] == "cosine"
    assert captured["umap"][0]["random_state"] == 17
    assert captured["umap"][1]["n_components"] == 2
    assert captured["umap"][1]["random_state"] == 17
    assert captured["hdbscan"][0]["min_cluster_size"] == 8
    assert captured["hdbscan"][0]["metric"] == "euclidean"
    assert captured["hdbscan"][0]["cluster_selection_method"] == "eom"
    assert captured["hdbscan"][0]["prediction_data"] is True
    assert captured["vectorizer"][0] == {"ngram_range": (1, 3), "min_df": 2}
    assert captured["bertopic"][0]["calculate_probabilities"] is True
    assert captured["bertopic"][0]["embedding_model"] is None
