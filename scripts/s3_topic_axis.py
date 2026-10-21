"""Summaries for relating existing BERTopic assignments to S³ axis scores."""

from __future__ import annotations

import re

import pandas as pd


def classify_topic_axis_percentiles(
    summary: pd.DataFrame,
    *,
    lower_threshold: float = 0.30,
    upper_threshold: float = 0.70,
) -> pd.DataFrame:
    """Classify topic mean axis scores by percentile among topics per axis.

    Percentile ranks use average ranks for ties. Scores below the lower cutoff
    are ``low``; scores above the upper cutoff are ``high``; values on either
    cutoff or between them are ``typical``.
    """
    required = {"topic_id", "axis_id", "mean_score"}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"Missing topic-axis summary columns: {sorted(missing)}")
    if not 0 <= lower_threshold < upper_threshold <= 1:
        raise ValueError("Percentile thresholds must satisfy 0 <= lower < upper <= 1")
    if summary.duplicated(["topic_id", "axis_id"]).any():
        raise ValueError("Topic-axis summary must contain one row per topic and axis")
    if summary["mean_score"].isna().any():
        raise ValueError("Topic-axis mean scores must not contain missing values")

    result = summary[["topic_id", "axis_id", "mean_score"]].copy()
    result["axis_percentile"] = result.groupby("axis_id")["mean_score"].rank(
        method="average", pct=True
    )
    result["status"] = "typical"
    result.loc[result["axis_percentile"] < lower_threshold, "status"] = "low"
    result.loc[result["axis_percentile"] > upper_threshold, "status"] = "high"
    return result.sort_values(["axis_id", "axis_percentile", "topic_id"]).reset_index(
        drop=True
    )


def summarize_topic_axis_scores(
    passages: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    topic_column: str = "topic_id",
) -> pd.DataFrame:
    """Return score distributions for each existing topic and S³ component.

    Topic IDs are treated as fixed group labels, including BERTopic's ``-1``
    outliers. This does not calculate or alter topic assignments.
    """
    if topic_column not in passages:
        raise KeyError(f"Missing topic column: {topic_column}")
    if len(passages) != len(scores):
        raise ValueError("Passage rows and score rows must have matching lengths")
    if not scores.columns.size:
        raise ValueError("At least one S³ score column is required")
    if scores.isna().any().any():
        raise ValueError("S³ scores must not contain missing values")

    component_ids: list[int] = []
    for column in scores.columns:
        match = re.fullmatch(r"s3_component_(\d+)", str(column))
        if match is None:
            raise ValueError(f"Unexpected S³ score column: {column}")
        component_ids.append(int(match.group(1)))

    long_scores = scores.reset_index(drop=True).copy()
    long_scores["topic_id"] = passages[topic_column].reset_index(drop=True)
    long_scores = long_scores.melt(
        id_vars="topic_id",
        var_name="score_column",
        value_name="score",
    )
    long_scores["axis_id"] = long_scores["score_column"].str.extract(
        r"s3_component_(\d+)"
    )[0].astype(int)
    if set(long_scores["axis_id"].unique()) != set(component_ids):
        raise AssertionError("Unexpected S³ axis IDs in the long score table")

    grouped = long_scores.groupby(["topic_id", "axis_id"], dropna=False)["score"]
    summary = grouped.agg(
        passage_count="count",
        mean_score="mean",
        std_score="std",
        median_score="median",
        q10_score=lambda values: values.quantile(0.10),
        q25_score=lambda values: values.quantile(0.25),
        q75_score=lambda values: values.quantile(0.75),
        q90_score=lambda values: values.quantile(0.90),
    ).reset_index()
    return summary.sort_values(["axis_id", "topic_id"]).reset_index(drop=True)


def topic_axis_mean_extremes(
    summary: pd.DataFrame,
    topic_names: dict[int, str],
) -> pd.DataFrame:
    """Rank each original topic's highest and lowest mean S³ axes by name."""
    required = {"topic_id", "axis_id", "passage_count", "mean_score"}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"Missing topic-axis summary columns: {sorted(missing)}")
    if summary.duplicated(["topic_id", "axis_id"]).any():
        raise ValueError("Topic-axis summary must contain one row per topic and axis")

    records = []
    for topic_id, group in summary.sort_values("axis_id").groupby("topic_id", sort=False):
        if topic_id not in topic_names:
            raise KeyError(f"Missing descriptive name for topic ID {topic_id}")
        high = group.loc[group["mean_score"].idxmax()]
        low = group.loc[group["mean_score"].idxmin()]
        records.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_names[topic_id],
                "passage_count": int(group["passage_count"].max()),
                "highest_axis_id": int(high["axis_id"]),
                "highest_mean_score": float(high["mean_score"]),
                "lowest_axis_id": int(low["axis_id"]),
                "lowest_mean_score": float(low["mean_score"]),
            }
        )
    return pd.DataFrame(records).sort_values("topic_name").reset_index(drop=True)


def build_topic_axis_review_tables(
    summary: pd.DataFrame,
    topic_names: dict[int, str],
    positive_terms: list[list[str]],
    negative_terms: list[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build named topic extremes and the positive/negative S³ axis keywords."""
    axis_ids = sorted(summary["axis_id"].unique().tolist())
    if len(positive_terms) != len(axis_ids) or len(negative_terms) != len(axis_ids):
        raise ValueError("Positive and negative keyword lists must match the axis count")

    topic_extremes = topic_axis_mean_extremes(summary, topic_names)
    means = summary.pivot(index="topic_id", columns="axis_id", values="mean_score")
    means.columns = [f"axis_{int(axis_id)}_mean_z" for axis_id in means.columns]
    topic_extremes = topic_extremes.merge(
        means.reset_index(), on="topic_id", validate="one_to_one"
    )

    keyword_rows = []
    for offset, axis_id in enumerate(axis_ids):
        positive = ", ".join(positive_terms[offset])
        negative = ", ".join(negative_terms[offset])
        keyword_rows.append(
            {
                "axis_id": int(axis_id),
                "axis": f"Axis {axis_id}",
                "positive_end_keywords": positive,
                "negative_end_keywords": negative,
            }
        )
    keywords = pd.DataFrame(keyword_rows)
    keyword_map = keywords.set_index("axis_id")
    for prefix, axis_column in (
        ("highest", "highest_axis_id"),
        ("lowest", "lowest_axis_id"),
    ):
        topic_extremes[f"{prefix}_axis_positive_keywords"] = topic_extremes[
            axis_column
        ].map(keyword_map.positive_end_keywords)
        topic_extremes[f"{prefix}_axis_negative_keywords"] = topic_extremes[
            axis_column
        ].map(keyword_map.negative_end_keywords)
        topic_extremes[f"{prefix}_axis"] = topic_extremes[axis_column].map(
            lambda axis_id: f"Axis {axis_id}"
        )
        topic_extremes[f"{prefix}_mean_z"] = topic_extremes[
            f"{prefix}_mean_score"
        ].round(3)
    return topic_extremes, keywords
