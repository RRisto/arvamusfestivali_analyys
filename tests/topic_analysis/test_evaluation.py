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
from arvamusfestivali_transcripts.topic_analysis.modelling import TopicModelConfig
from arvamusfestivali_transcripts.topic_analysis.types import (
    ChunkingConfig,
    EmbeddingResult,
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
    metrics = evaluation.evaluate_run(_run(), _embedding())

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
        "cache_identities": {"qwen": "cache-digest-1"},
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
    assert manifest["cache_identities"] == {"qwen": "cache-digest-1"}
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
