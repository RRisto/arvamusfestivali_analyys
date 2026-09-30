"""Metrics and reproducible review export for controlled topic runs."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arvamusfestivali_transcripts.topic_analysis import evaluation
from arvamusfestivali_transcripts.topic_analysis.cache import CacheIdentity
from arvamusfestivali_transcripts.topic_analysis.modelling import TopicModelConfig
from arvamusfestivali_transcripts.topic_analysis.types import (
    ChunkingConfig,
    EmbeddingResult,
    ManualTopicReview,
    Passage,
    TopicRun,
)


def _passages(count: int = 4) -> tuple[Passage, ...]:
    return tuple(
        Passage(
            passage_id=f"p{index}",
            episode_id=f"episode-{index // 2}",
            duplicate_episode_ids=(f"episode-{index // 2}", f"copy-{index // 2}"),
            title=f"Title {index // 2}",
            start_seconds=float(index * 10),
            end_seconds=float(index * 10 + 10),
            text=f"Passage text {index}",
            audio_sha256=("a" if index < 2 else "b") * 64,
            audio_url="https://example.test/audio.mp3",
            word_count=3,
            cue_count=1,
        )
        for index in range(count)
    )


def _embedding(model: str = "qwen", count: int = 4) -> EmbeddingResult:
    rows = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )[:count]
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return EmbeddingResult(
        model_key=model,
        model_id=f"{model}-id",
        model_revision="revision-1",
        dimension=2,
        passage_ids=tuple(f"p{index}" for index in range(count)),
        embeddings=rows,
        cache_hits=count,
    )


def _run(
    model: str = "qwen",
    topics: tuple[int, ...] = (0, 0, 1, 1),
    config: TopicModelConfig = TopicModelConfig(),
) -> TopicRun:
    return TopicRun(
        model_key=model,
        topics=np.array(topics),
        probabilities=None,
        reduced_embeddings=np.zeros((len(topics), 2)),
        topic_info=pd.DataFrame(
            {"Topic": [0, 1], "Count": [2, 2], "Representation": [["one", "two"], ["two", "three"]]}
        ),
        representative_passages={0: ("p0",), 1: ("p2",)},
        cluster_persistence=(0.6, 0.8),
        passage_ids=tuple(f"p{index}" for index in range(len(topics))),
        topic_model_config=asdict(config),
    )


def _without_passage_ids(run: TopicRun) -> TopicRun:
    # Simulate a legacy or externally deserialized run that bypassed dataclass validation.
    object.__setattr__(run, "passage_ids", ())
    return run


def test_evaluate_run_handles_all_outliers() -> None:
    run = _run(topics=(-1, -1, -1))
    metrics = evaluation.evaluate_run(run, _embedding(count=3))

    assert metrics["model_key"] == "qwen"
    assert metrics["passage_count"] == 3
    assert metrics["topic_count"] == 0
    assert metrics["outlier_count"] == 3
    assert metrics["outlier_fraction"] == 1.0
    assert metrics["silhouette"] is None
    assert metrics["silhouette_unavailable_reason"] == "fewer than two non-outlier topics"


def test_evaluate_run_reports_diversity_and_valid_silhouette() -> None:
    run = replace(_run(), clustering_embeddings=_embedding().embeddings.copy())
    metrics = evaluation.evaluate_run(run, _embedding())

    assert metrics["topic_count"] == 2
    assert metrics["topic_diversity"] == pytest.approx(0.75)
    assert metrics["mean_cluster_persistence"] == pytest.approx(0.7)
    assert metrics["silhouette"] is not None
    assert metrics["silhouette_unavailable_reason"] is None


def test_topic_run_rejects_missing_passage_ids() -> None:
    with pytest.raises(ValueError, match="passage_ids"):
        replace(_run(), passage_ids=())


def test_evaluate_run_requires_ordered_passage_ids() -> None:
    run = _without_passage_ids(_run())

    with pytest.raises(ValueError, match="passage IDs and order"):
        evaluation.evaluate_run(run, _embedding())


def test_compare_runs_reports_assignment_and_neighbour_agreement() -> None:
    runs = {"qwen": _run(), "bge": _run("bge", (2, 2, 3, 3))}
    embeddings = {"qwen": _embedding(), "bge": _embedding("bge")}

    comparison = evaluation.compare_runs(runs, embeddings, neighbours=2)

    assert len(comparison) == 1
    assert comparison.loc[0, "model_a"] == "qwen"
    assert comparison.loc[0, "model_b"] == "bge"
    assert comparison.loc[0, "adjusted_mutual_information"] == pytest.approx(1.0)
    assert comparison.loc[0, "nearest_neighbour_overlap"] == pytest.approx(1.0)
    assert "winner" not in comparison.columns


def test_compare_runs_rejects_misaligned_passages() -> None:
    wrong = _embedding("bge")
    wrong = EmbeddingResult(
        model_key=wrong.model_key,
        model_id=wrong.model_id,
        model_revision=wrong.model_revision,
        dimension=wrong.dimension,
        passage_ids=tuple(reversed(wrong.passage_ids)),
        embeddings=wrong.embeddings,
        cache_hits=wrong.cache_hits,
    )

    with pytest.raises(ValueError, match="passage IDs and order"):
        evaluation.compare_runs(
            {"qwen": _run(), "bge": _run("bge")},
            {"qwen": _embedding(), "bge": wrong},
        )


def test_compare_runs_rejects_missing_passage_ids() -> None:
    with pytest.raises(ValueError, match="passage IDs and order"):
        evaluation.compare_runs(
            {"qwen": _without_passage_ids(_run()), "bge": _run("bge")},
            {"qwen": _embedding(), "bge": _embedding("bge")},
        )


def test_compare_runs_rejects_different_topic_model_configs() -> None:
    with pytest.raises(ValueError, match="topic model configuration"):
        evaluation.compare_runs(
            {
                "qwen": _run(),
                "bge": _run("bge", config=TopicModelConfig(random_state=17)),
            },
            {"qwen": _embedding(), "bge": _embedding("bge")},
        )


def test_compare_runs_rejects_missing_topic_model_config() -> None:
    run = _run()
    object.__setattr__(run, "topic_model_config", None)
    with pytest.raises(ValueError, match="topic model configuration"):
        evaluation.compare_runs({"qwen": run}, {"qwen": _embedding()})


def test_review_rows_keep_provenance_and_blank_manual_fields() -> None:
    rows = evaluation.build_review_rows({"qwen": _run()}, _passages())

    assert rows.loc[0, "model_key"] == "qwen"
    assert rows.loc[0, "topic_id"] == 0
    assert rows.loc[0, "size"] == 2
    assert json.loads(rows.loc[0, "representative_passage_ids"]) == ["p0"]
    assert json.loads(rows.loc[0, "titles"]) == ["Title 0"]
    assert json.loads(rows.loc[0, "timestamps"]) == [0.0]
    assert json.loads(rows.loc[0, "texts"]) == ["Passage text 0"]
    assert json.loads(rows.loc[0, "audio_links"]) == ["https://example.test/audio.mp3#t=0"]
    assert rows.loc[0, "verdict"] == ""
    assert rows.loc[0, "note"] == ""


def test_review_rows_reject_missing_passage_ids() -> None:
    with pytest.raises(ValueError, match="passage IDs and order"):
        evaluation.build_review_rows({"qwen": _without_passage_ids(_run())}, _passages())


def test_export_experiment_writes_deterministic_tables_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "results"
    kwargs = {
        "chunking": ChunkingConfig(),
        "topic_model": TopicModelConfig(random_state=17),
        "cache_identities": {"qwen": _identity()},
        "created_at": datetime(2026, 9, 30, 12, 0, tzinfo=UTC),
        "git_revision": "abc123",
    }
    args = (
        root, "experiment-1", _passages(),
        {"qwen": _run(config=TopicModelConfig(random_state=17))},
        {"qwen": _embedding()},
    )

    output = evaluation.export_experiment(*args, **kwargs)
    first_contents = {path.name: path.read_bytes() for path in output.iterdir()}
    output = evaluation.export_experiment(*args, **kwargs)

    assert set(first_contents) == {
        "manifest.json", "passages.csv", "topic-assignments.csv", "metrics.csv",
        "cross-model.csv", "manual-review.csv",
    }
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first_contents
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_revision"] == "abc123"
    assert manifest["created_at_utc"] == "2026-09-30T12:00:00Z"
    assert manifest["selected_audio_hashes"] == ["a" * 64, "b" * 64]
    assert manifest["duplicate_mappings"]["episode-0"] == ["episode-0", "copy-0"]
    assert manifest["chunk_configuration"]["target_seconds"] == 180.0
    assert manifest["cache_identities"]["qwen"] == {
        **asdict(_identity()), "digest": _identity().digest,
    }
    assert manifest["model_metadata"]["qwen"]["model_id"] == "qwen-id"
    assert manifest["topic_model_configuration"]["random_state"] == 17
    assert manifest["random_seeds"]["topic_model"] == 17
    assert "bertopic" in manifest["package_versions"]
    assert "sentence-transformers" in manifest["package_versions"]
    assert "python" in manifest["package_versions"]
    assignments = pd.read_csv(output / "topic-assignments.csv")
    assert assignments["passage_id"].tolist() == ["p0", "p1", "p2", "p3"]
    assert assignments["topic_id"].tolist() == [0, 0, 1, 1]
    assert not list(output.glob(".*.tmp"))


def _identity() -> CacheIdentity:
    return CacheIdentity("qwen-id", "revision-1", 2, "cluster Estonian", ChunkingConfig(), 2)


def test_export_preserves_probabilities_and_completed_reviews(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "never-export-this-secret")
    run = replace(
        _run(), probabilities=np.array([[.1, .8], [.2, .7], [.6, .1], [.9, .05]]),
        probability_topic_ids=(1, 0),
    )
    review = ManualTopicReview("qwen", 0, "mixed", "Two subjects overlap")
    output = evaluation.export_experiment(
        tmp_path, "reviewed", _passages(), {"qwen": run}, {"qwen": _embedding()},
        chunking=ChunkingConfig(), topic_model=TopicModelConfig(),
        cache_identities={"qwen": _identity()}, manual_reviews=(review,), git_revision="abc123",
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_identities"]["qwen"]["instruction"] == "cluster Estonian"
    assert manifest["cache_identities"]["qwen"]["adapter_version"] == 2
    assert len(manifest["passage_cache_digests"]) == 4
    assert manifest["probability_topic_ids"]["qwen"] == [1, 0]
    assignments = pd.read_csv(output / "topic-assignments.csv")
    assert assignments["assigned_probability"].tolist() == [.8, .7, .6, .9]
    assert json.loads(assignments.loc[0, "probabilities_by_topic"]) == {"1": .1, "0": .8}
    reviews = pd.read_csv(output / "manual-review.csv").fillna("")
    assert reviews.loc[0, "verdict"] == "mixed"
    assert reviews.loc[0, "note"] == "Two subjects overlap"
    assert reviews.loc[1, "verdict"] == ""
    assert all("never-export-this-secret" not in p.read_text() for p in output.iterdir())


def test_review_annotations_reject_unknown_or_duplicate_topic() -> None:
    unknown = ManualTopicReview("qwen", 99, "unclear")
    duplicate = ManualTopicReview("qwen", 0, "coherent")
    for reviews in ((unknown,), (duplicate, duplicate)):
        with pytest.raises(ValueError, match="review"):
            evaluation.build_review_rows({"qwen": _run()}, _passages(), manual_reviews=reviews)


def test_disputed_passages_detect_partition_disagreement_without_outliers() -> None:
    runs = {"qwen": _run(), "bge": _run("bge", (0, 1, 0, 1))}
    rows = evaluation.build_disagreement_rows(runs, _passages())
    assert rows["passage_id"].tolist() == ["p0", "p1", "p2", "p3"]
    assert rows["disagreement_fraction"].tolist() == pytest.approx([2 / 3] * 4)
    assert json.loads(rows.loc[0, "changed_peer_passage_ids"]) == ["p1", "p2"]
    assert rows.loc[0, "audio_link"] == "https://example.test/audio.mp3#t=0"
    assert rows.loc[0, "audio_sha256"] == "a" * 64
    assert rows.loc[0, "end_seconds"] == 10

    renamed = {"qwen": _run(), "bge": _run("bge", (7, 7, 8, 8))}
    assert evaluation.build_disagreement_rows(renamed, _passages()).empty


def test_outliers_are_not_treated_as_one_shared_cluster_in_disagreements() -> None:
    rows = evaluation.build_disagreement_rows(
        {"qwen": _run(topics=(-1, -1, 1, 1)), "bge": _run("bge", (0, 0, 1, 1))},
        _passages(),
    )
    assert rows["passage_id"].tolist() == ["p0", "p1"]
    assert rows["outlier_disagreement"].all()


def test_boundary_rows_select_lowest_membership_per_topic_with_provenance() -> None:
    run = replace(
        _run(), probabilities=np.array([[.4, .35], [.9, .05], [.1, .8], [.3, .4]]),
        probability_topic_ids=(0, 1),
    )
    rows = evaluation.build_boundary_rows({"qwen": run}, _passages(), per_topic=1)
    assert rows["passage_id"].tolist() == ["p0", "p3"]
    assert rows["assigned_probability"].tolist() == [.4, .4]
    assert rows["membership_margin"].tolist() == pytest.approx([.05, .1])
    assert rows.loc[1, "audio_link"] == "https://example.test/audio.mp3#t=30"
    assert rows.loc[1, "start_seconds"] == 30
    assert rows.loc[1, "text"] == "Passage text 3"
    assert evaluation.build_boundary_rows({"qwen": _run()}, _passages()).empty


def test_silhouette_uses_actual_clustering_coordinates_and_euclidean_distance() -> None:
    run = replace(
        _run(), clustering_embeddings=np.array([[0., 0.], [0., 1.], [10., 0.], [10., 1.]]),
    )
    metrics = evaluation.evaluate_run(run, _embedding())
    assert metrics["silhouette"] == pytest.approx(1 - 2 / (10 + np.sqrt(101)))
    assert metrics["silhouette_space"] == "clustering_reduction"
    assert metrics["silhouette_metric"] == "euclidean"


def test_silhouette_is_unavailable_without_clustering_coordinates() -> None:
    metrics = evaluation.evaluate_run(_run(), _embedding())
    assert metrics["silhouette"] is None
    assert metrics["silhouette_unavailable_reason"] == (
        "clustering reduction coordinates unavailable"
    )


def test_export_rejects_manifest_config_different_from_fitted_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="topic model configuration"):
        evaluation.export_experiment(
            tmp_path, "experiment-1", _passages(),
            {"qwen": _run(config=TopicModelConfig(random_state=17))},
            {"qwen": _embedding()},
            chunking=ChunkingConfig(),
            topic_model=TopicModelConfig(random_state=42),
            cache_identities={"qwen": "cache-digest-1"},
            created_at=datetime(2026, 9, 30, tzinfo=UTC),
            git_revision="abc123",
        )


def test_export_rejects_missing_run_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="passage IDs and order"):
        evaluation.export_experiment(
            tmp_path, "experiment-1", _passages(),
            {"qwen": _without_passage_ids(_run())}, {"qwen": _embedding()},
            chunking=ChunkingConfig(), topic_model=TopicModelConfig(),
            cache_identities={"qwen": "cache-digest-1"}, git_revision="abc123",
        )


def test_export_rejects_missing_run_config(tmp_path: Path) -> None:
    run = _run()
    object.__setattr__(run, "topic_model_config", None)
    with pytest.raises(ValueError, match="topic model configuration"):
        evaluation.export_experiment(
            tmp_path, "experiment-1", _passages(),
            {"qwen": run}, {"qwen": _embedding()},
            chunking=ChunkingConfig(), topic_model=TopicModelConfig(),
            cache_identities={"qwen": "cache-digest-1"}, git_revision="abc123",
        )


@pytest.mark.parametrize("git_revision", (None, "", "   "))
def test_export_rejects_missing_git_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_revision: str | None
) -> None:
    monkeypatch.setattr(evaluation, "_git_revision", lambda: None)
    with pytest.raises(ValueError, match="git revision"):
        evaluation.export_experiment(
            tmp_path, "experiment-1", _passages(),
            {"qwen": _run()}, {"qwen": _embedding()},
            chunking=ChunkingConfig(), topic_model=TopicModelConfig(),
            cache_identities={"qwen": "cache-digest-1"}, git_revision=git_revision,
        )


@pytest.mark.parametrize(
    "cache_identities",
    (None, {}, {"qwen": "cache-digest-1"}, {"qwen": "cache-digest-1", "bge": " "}),
)
def test_export_requires_cache_identity_for_every_model(
    tmp_path: Path, cache_identities: dict[str, str] | None
) -> None:
    with pytest.raises(ValueError, match="cache identities"):
        evaluation.export_experiment(
            tmp_path, "experiment-1", _passages(),
            {"qwen": _run(), "bge": _run("bge")},
            {"qwen": _embedding(), "bge": _embedding("bge")},
            chunking=ChunkingConfig(), topic_model=TopicModelConfig(),
            cache_identities=cache_identities, git_revision="abc123",
        )


@pytest.mark.parametrize("experiment_id", ("../escape", "C:outside", ""))
def test_export_rejects_unsafe_experiment_ids(tmp_path: Path, experiment_id: str) -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        evaluation.export_experiment(
            tmp_path,
            experiment_id,
            _passages(),
            {"qwen": _run()},
            {"qwen": _embedding()},
            chunking=ChunkingConfig(),
            topic_model=TopicModelConfig(),
            created_at=datetime(2026, 9, 30, tzinfo=UTC),
            git_revision="abc123",
        )
