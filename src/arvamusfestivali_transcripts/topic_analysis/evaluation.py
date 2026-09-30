"""Descriptive topic metrics and reviewable, reproducible experiment exports."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from .cache import CacheIdentity, _passage_digest
from .modelling import TopicModelConfig
from .types import ChunkingConfig, EmbeddingResult, ManualTopicReview, Passage, TopicRun

_METRIC_COLUMNS = (
    "model_key", "passage_count", "topic_count", "outlier_count", "outlier_fraction",
    "silhouette", "silhouette_space", "silhouette_metric", "silhouette_unavailable_reason",
    "mean_cluster_persistence",
    "topic_diversity",
)
_COMPARISON_COLUMNS = (
    "model_a", "model_b", "adjusted_mutual_information", "nearest_neighbour_overlap"
)
_REVIEW_COLUMNS = (
    "model_key", "topic_id", "size", "representative_passage_ids", "titles",
    "timestamps", "texts", "audio_links", "verdict", "note",
)


def _validate_run_and_embedding(run: TopicRun, embedding: EmbeddingResult) -> None:
    if run.model_key != embedding.model_key:
        raise ValueError("run and embedding model keys must match")
    if not run.passage_ids or run.passage_ids != embedding.passage_ids:
        raise ValueError("run and embedding passage IDs and order must match")


def _controlled_config(
    runs: Mapping[str, TopicRun], expected: Mapping[str, int | float] | None = None
) -> dict[str, int | float]:
    if not runs:
        raise ValueError("topic model configuration requires at least one run")
    first: dict[str, int | float] | None = None
    for run in runs.values():
        if run.topic_model_config is None:
            raise ValueError(f"topic model configuration missing for {run.model_key}")
        current = dict(run.topic_model_config)
        if first is None:
            first = current
        elif current != first:
            raise ValueError("topic model configuration differs across runs")
    if expected is not None and first != dict(expected):
        raise ValueError("topic model configuration differs from export manifest")
    assert first is not None
    return first


def _topic_diversity(run: TopicRun) -> float:
    info = run.topic_info
    if "Topic" not in info or "Representation" not in info:
        return 0.0
    terms = [
        str(term)
        for _, row in info.iterrows()
        if int(row["Topic"]) != -1 and isinstance(row["Representation"], list | tuple)
        for term in row["Representation"]
        if str(term).strip()
    ]
    return len(set(terms)) / len(terms) if terms else 0.0


def evaluate_run(run: TopicRun, embedding: EmbeddingResult) -> dict[str, object]:
    """Describe a run without assigning an automatic quality verdict."""
    _validate_run_and_embedding(run, embedding)
    topics = run.topics.astype(int)
    non_outliers = topics != -1
    non_outlier_topics = np.unique(topics[non_outliers])
    silhouette: float | None = None
    reason: str | None = None
    if len(non_outlier_topics) < 2:
        reason = "fewer than two non-outlier topics"
    elif non_outliers.sum() <= len(non_outlier_topics):
        reason = "too few non-outlier passages for silhouette"
    elif run.clustering_embeddings is None:
        reason = "clustering reduction coordinates unavailable"
    else:
        silhouette = float(
            silhouette_score(
                run.clustering_embeddings[non_outliers], topics[non_outliers], metric="euclidean"
            )
        )
    count = len(topics)
    outlier_count = int((~non_outliers).sum())
    return {
        "model_key": run.model_key,
        "passage_count": count,
        "topic_count": len(non_outlier_topics),
        "outlier_count": outlier_count,
        "outlier_fraction": outlier_count / count if count else 0.0,
        "silhouette": silhouette,
        "silhouette_space": "clustering_reduction",
        "silhouette_metric": "euclidean",
        "silhouette_unavailable_reason": reason,
        "mean_cluster_persistence": (
            float(np.mean(run.cluster_persistence)) if run.cluster_persistence else None
        ),
        "topic_diversity": _topic_diversity(run),
    }


def _top_neighbours(embedding: EmbeddingResult, neighbours: int) -> list[set[int]]:
    rows = np.asarray(embedding.embeddings, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms == 0):
        raise ValueError("nearest-neighbour comparison needs nonzero embedding rows")
    rows = rows / norms[:, None]
    similarities = rows @ rows.T
    np.fill_diagonal(similarities, -np.inf)
    count = min(neighbours, len(rows) - 1)
    return [set(np.argsort(-scores, kind="stable")[:count]) for scores in similarities]


def _neighbour_overlap(
    first: EmbeddingResult, second: EmbeddingResult, neighbours: int
) -> float | None:
    if len(first.passage_ids) < 2:
        return None
    first_sets = _top_neighbours(first, neighbours)
    second_sets = _top_neighbours(second, neighbours)
    return float(np.mean([
        len(left & right) / len(left | right)
        for left, right in zip(first_sets, second_sets, strict=True)
    ]))


def compare_runs(
    runs: Mapping[str, TopicRun],
    embeddings: Mapping[str, EmbeddingResult],
    *,
    neighbours: int = 10,
) -> pd.DataFrame:
    """Compare ordered pairs by assignments and same-passage local neighbourhoods."""
    if not isinstance(neighbours, int) or isinstance(neighbours, bool) or neighbours <= 0:
        raise ValueError("neighbours must be positive")
    if set(runs) != set(embeddings):
        raise ValueError("runs and embeddings must have identical model keys")
    for key, run in runs.items():
        if run.model_key != key or embeddings[key].model_key != key:
            raise ValueError("mapping keys must match model keys")
        _validate_run_and_embedding(run, embeddings[key])
    if runs:
        _controlled_config(runs)
    rows = []
    for first_key, second_key in combinations(runs, 2):
        first = embeddings[first_key]
        second = embeddings[second_key]
        if first.passage_ids != second.passage_ids:
            raise ValueError("comparison passage IDs and order must match across models")
        rows.append({
            "model_a": first_key,
            "model_b": second_key,
            "adjusted_mutual_information": float(
                adjusted_mutual_info_score(runs[first_key].topics, runs[second_key].topics)
            ),
            "nearest_neighbour_overlap": _neighbour_overlap(first, second, neighbours),
        })
    return pd.DataFrame(rows, columns=_COMPARISON_COLUMNS)


def _json_array(values: Sequence[object]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def build_review_rows(
    runs: Mapping[str, TopicRun], passages: Sequence[Passage], *,
    manual_reviews: Sequence[ManualTopicReview] = (),
) -> pd.DataFrame:
    """Build one editable review row per non-outlier topic with source links."""
    by_id = {passage.passage_id: passage for passage in passages}
    if len(by_id) != len(passages):
        raise ValueError("passages must have unique IDs")
    passage_ids = tuple(by_id)
    rows = []
    for key, run in runs.items():
        if key != run.model_key:
            raise ValueError("mapping keys must match model keys")
        if not run.passage_ids or run.passage_ids != passage_ids:
            raise ValueError("run and passage IDs and order must match")
        for topic in sorted(int(value) for value in np.unique(run.topics) if int(value) != -1):
            ids = run.representative_passages.get(topic, ())
            try:
                representatives = [by_id[passage_id] for passage_id in ids]
            except KeyError as error:
                raise ValueError(f"unknown representative passage ID: {error.args[0]}") from error
            rows.append({
                "model_key": key,
                "topic_id": topic,
                "size": int(np.sum(run.topics == topic)),
                "representative_passage_ids": _json_array(ids),
                "titles": _json_array([item.title for item in representatives]),
                "timestamps": _json_array([item.start_seconds for item in representatives]),
                "texts": _json_array([item.text for item in representatives]),
                "audio_links": _json_array([
                    item.timestamped_audio_url for item in representatives
                ]),
                "verdict": "",
                "note": "",
            })
    by_topic = {(row["model_key"], row["topic_id"]): row for row in rows}
    seen = set()
    for review in manual_reviews:
        if not isinstance(review, ManualTopicReview):
            raise ValueError("manual reviews must be validated ManualTopicReview records")
        key = (review.model_key, review.topic_id)
        if key not in by_topic or key in seen:
            raise ValueError(f"manual review has unknown or duplicate topic: {key}")
        seen.add(key)
        by_topic[key].update(verdict=review.verdict, note=review.note)
    return pd.DataFrame(rows, columns=_REVIEW_COLUMNS)


def _assignment_probabilities(run: TopicRun, index: int) -> tuple[float | None, dict, float | None]:
    """Assigned-topic membership, explicitly mapped soft memberships, raw 1-D strength.

    Outliers have no assigned-cluster probability. HDBSCAN memberships are not
    calibrated semantic-confidence scores, and their sum need not be one.
    """
    if run.probabilities is None:
        return None, {}, None
    topic = int(run.topics[index])
    if run.probabilities.ndim == 1:
        strength = float(run.probabilities[index])
        return (strength if topic != -1 else None), {}, strength
    memberships = dict(zip(
        run.probability_topic_ids, map(float, run.probabilities[index]), strict=True,
    ))
    return memberships.get(topic), memberships, None


_PROVENANCE_COLUMNS = (
    "passage_id", "episode_id", "duplicate_episode_ids", "title", "start_seconds",
    "end_seconds", "text", "audio_sha256", "audio_link",
)


def _inspection_provenance(passage: Passage) -> dict[str, object]:
    return {
        "passage_id": passage.passage_id, "episode_id": passage.episode_id,
        "duplicate_episode_ids": _json_array(passage.duplicate_episode_ids),
        "title": passage.title, "start_seconds": passage.start_seconds,
        "end_seconds": passage.end_seconds, "text": passage.text,
        "audio_sha256": passage.audio_sha256, "audio_link": passage.timestamped_audio_url,
    }


def _validate_inspection_inputs(runs: Mapping[str, TopicRun], passages: Sequence[Passage]) -> None:
    ids = tuple(p.passage_id for p in passages)
    if len(set(ids)) != len(ids):
        raise ValueError("passages must have unique IDs")
    for key, run in runs.items():
        if key != run.model_key or run.passage_ids != ids:
            raise ValueError("inspection model keys and passage IDs and order must match")


def build_disagreement_rows(
    runs: Mapping[str, TopicRun], passages: Sequence[Passage],
) -> pd.DataFrame:
    """Find label-invariant differences in each passage's cluster mates.

    The fraction is the number of changed co-assignments divided by all other
    passages. Noise points have no cluster mates, even when both are labelled -1.
    Outlier-status changes are also retained for singleton/degenerate partitions.
    """
    _validate_inspection_inputs(runs, passages)
    rows = []
    for left_key, right_key in combinations(runs, 2):
        left, right = runs[left_key].topics, runs[right_key].topics
        for index, passage in enumerate(passages):
            left_mates = (left == left[index]) & (left[index] != -1)
            right_mates = (right == right[index]) & (right[index] != -1)
            changed = left_mates != right_mates
            changed[index] = False
            outlier_change = bool((left[index] == -1) != (right[index] == -1))
            if not changed.any() and not outlier_change:
                continue
            rows.append({
                "model_a": left_key, "model_b": right_key,
                **_inspection_provenance(passage),
                "topic_a": int(left[index]), "topic_b": int(right[index]),
                "disagreement_fraction": float(changed.sum()) / max(1, len(passages) - 1),
                "outlier_disagreement": outlier_change,
                "changed_peer_passage_ids": _json_array([
                    passages[i].passage_id for i in np.flatnonzero(changed)
                ]),
            })
    return pd.DataFrame(rows, columns=(
        "model_a", "model_b", *_PROVENANCE_COLUMNS, "topic_a", "topic_b",
        "disagreement_fraction", "outlier_disagreement", "changed_peer_passage_ids",
    ))


def build_boundary_rows(
    runs: Mapping[str, TopicRun], passages: Sequence[Passage], *, per_topic: int = 3,
) -> pd.DataFrame:
    """Inspect the lowest assigned memberships per non-outlier topic.

    The margin is assigned membership minus the largest other-topic membership,
    where a full matrix exists. Runs without memberships contribute no rows.
    """
    _validate_inspection_inputs(runs, passages)
    if not isinstance(per_topic, int) or isinstance(per_topic, bool) or per_topic < 1:
        raise ValueError("per_topic must be a positive integer")
    rows = []
    for key, run in runs.items():
        for topic in sorted(set(run.topics) - {-1}):
            candidates = []
            for index in np.flatnonzero(run.topics == topic):
                assigned, memberships, _ = _assignment_probabilities(run, index)
                if assigned is None:
                    continue
                competitors = [value for other, value in memberships.items() if other != topic]
                candidates.append({
                    "model_key": key, "topic_id": int(topic),
                    **_inspection_provenance(passages[index]),
                    "assigned_probability": assigned,
                    "membership_margin": assigned - max(competitors) if competitors else None,
                })
            candidates.sort(key=lambda row: (row["assigned_probability"], row["passage_id"]))
            rows.extend(candidates[:per_topic])
    return pd.DataFrame(rows, columns=(
        "model_key", "topic_id", *_PROVENANCE_COLUMNS,
        "assigned_probability", "membership_margin",
    ))


def _git_revision() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in (
        "bertopic", "google-genai", "hdbscan", "ipykernel", "jupyterlab", "nbclient",
        "nbformat", "numpy", "pandas", "plotly", "scikit-learn", "sentence-transformers",
        "torch", "umap-learn",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _atomic_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=f".{path.stem}.",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n", na_rep="")


def export_experiment(
    result_root: Path,
    experiment_id: str,
    passages: Sequence[Passage],
    runs: Mapping[str, TopicRun],
    embeddings: Mapping[str, EmbeddingResult],
    *,
    chunking: ChunkingConfig,
    topic_model: TopicModelConfig,
    cache_identities: Mapping[str, CacheIdentity] | None = None,
    manual_reviews: Sequence[ManualTopicReview] = (),
    created_at: datetime | None = None,
    git_revision: str | None = None,
) -> Path:
    """Write ordered data and an environment manifest using atomic file replacements."""
    if not isinstance(experiment_id, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id
    ) is None:
        raise ValueError("experiment_id must be a single nonempty directory name")
    if not passages:
        raise ValueError("passages must be nonempty")
    passage_ids = tuple(passage.passage_id for passage in passages)
    if len(set(passage_ids)) != len(passage_ids):
        raise ValueError("passages must have unique IDs")
    if set(runs) != set(embeddings):
        raise ValueError("runs and embeddings must have identical model keys")
    for key, embedding in embeddings.items():
        _validate_run_and_embedding(runs[key], embedding)
        if embedding.passage_ids != passage_ids:
            raise ValueError("export passage IDs and order must match embeddings")
    _controlled_config(runs, asdict(topic_model))

    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must have a timezone")
    timestamp_text = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    revision = _git_revision() if git_revision is None else git_revision
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("git revision must be nonempty")
    if cache_identities is None or set(cache_identities) != set(runs):
        raise ValueError("cache identities must be supplied for every run")
    cache_values: dict[str, dict] = {}
    for key, identity in cache_identities.items():
        if not isinstance(identity, CacheIdentity):
            raise ValueError(f"cache identities must contain full CacheIdentity records for {key}")
        result = embeddings[key]
        if (identity.model_id, identity.model_revision, identity.dimension, identity.chunking) != (
            result.model_id, result.model_revision, result.dimension, chunking,
        ):
            raise ValueError(f"cache identities disagree with embedding/configuration for {key}")
        cache_values[key] = {**asdict(identity), "digest": identity.digest}
    review_rows = build_review_rows(runs, passages, manual_reviews=manual_reviews)
    manifest = {
        "git_revision": revision.strip(),
        "created_at_utc": timestamp_text,
        "selected_audio_hashes": list(dict.fromkeys(
            passage.audio_sha256 for passage in passages
        )),
        "duplicate_mappings": {
            passage.episode_id: list(passage.duplicate_episode_ids)
            for passage in passages
        },
        "chunk_configuration": asdict(chunking),
        "cache_identities": cache_values,
        "passage_cache_digests": {
            passage.passage_id: _passage_digest(passage) for passage in passages
        },
        "probability_topic_ids": {
            key: list(run.probability_topic_ids) for key, run in runs.items()
        },
        "probability_semantics": (
            "HDBSCAN soft cluster memberships, not calibrated semantic confidence. "
            "assigned_probability is empty for outliers or unavailable probabilities; "
            "probabilities_by_topic maps final topic IDs to matrix columns; "
            "membership_strength retains raw 1-D output when no matrix is available."
        ),
        "model_metadata": {
            key: {
                "model_id": result.model_id,
                "model_revision": result.model_revision,
                "dimension": result.dimension,
                "cache_hits": result.cache_hits,
            }
            for key, result in embeddings.items()
        },
        "topic_model_configuration": asdict(topic_model),
        "random_seeds": {"topic_model": topic_model.random_state},
        "package_versions": _package_versions(),
    }
    passage_rows = [
        {**asdict(passage), "duplicate_episode_ids": _json_array(passage.duplicate_episode_ids),
         "timestamped_audio_url": passage.timestamped_audio_url}
        for passage in passages
    ]
    assignment_rows = []
    for key, run in runs.items():
        for index, (passage_id, topic) in enumerate(zip(passage_ids, run.topics, strict=True)):
            assigned, memberships, strength = _assignment_probabilities(run, index)
            assignment_rows.append({
                "model_key": key, "passage_id": passage_id, "topic_id": int(topic),
                "assigned_probability": assigned,
                "probabilities_by_topic": json.dumps(memberships, separators=(",", ":")),
                "membership_strength": strength,
            })
    output = Path(result_root) / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest.json": json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "passages.csv": _csv(pd.DataFrame(passage_rows)),
        "topic-assignments.csv": _csv(pd.DataFrame(
            assignment_rows, columns=(
                "model_key", "passage_id", "topic_id", "assigned_probability",
                "probabilities_by_topic", "membership_strength",
            )
        )),
        "metrics.csv": _csv(pd.DataFrame(
            [evaluate_run(runs[key], embeddings[key]) for key in runs], columns=_METRIC_COLUMNS
        )),
        "cross-model.csv": _csv(compare_runs(runs, embeddings)),
        "manual-review.csv": _csv(review_rows),
    }
    for name, content in files.items():
        _atomic_text(output / name, content)
    return output
