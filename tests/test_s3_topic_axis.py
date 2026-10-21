import pandas as pd

from scripts.s3_topic_axis import (
    summarize_topic_axis_scores,
    topic_axis_mean_extremes,
    build_topic_axis_review_tables,
    classify_topic_axis_percentiles,
)


def test_topic_axis_summary_keeps_outliers_and_reports_score_spread() -> None:
    passages = pd.DataFrame({"topic_id": [-1, 4, 4]})
    scores = pd.DataFrame(
        {
            "s3_component_0": [-2.0, 0.0, 2.0],
            "s3_component_1": [1.0, 3.0, 5.0],
        }
    )

    summary = summarize_topic_axis_scores(passages, scores)
    rows = summary.sort_values(["topic_id", "axis_id"])
    actual = list(
        rows[["topic_id", "axis_id", "passage_count", "median_score"]]
        .itertuples(index=False, name=None)
    )

    assert actual == [(-1, 0, 1, -2.0), (-1, 1, 1, 1.0), (4, 0, 2, 1.0), (4, 1, 2, 4.0)]
    assert rows.loc[(rows.topic_id == 4) & (rows.axis_id == 0), "q10_score"].item() == 0.2
    assert rows.loc[(rows.topic_id == 4) & (rows.axis_id == 0), "q90_score"].item() == 1.8


def test_topic_axis_mean_extremes_uses_descriptive_names_and_signed_means() -> None:
    summary = pd.DataFrame(
        {
            "topic_id": [1, 1, 1, -1, -1, -1],
            "axis_id": [0, 1, 2, 0, 1, 2],
            "passage_count": [3, 3, 3, 2, 2, 2],
            "mean_score": [-2.0, 1.5, 0.1, 0.3, -0.8, 0.2],
        }
    )

    extremes = topic_axis_mean_extremes(
        summary,
        {1: "Climate discussion", -1: "Unassigned"},
    ).set_index("topic_id")

    assert extremes.loc[1, "topic_name"] == "Climate discussion"
    assert extremes.loc[1, "highest_axis_id"] == 1
    assert extremes.loc[1, "highest_mean_score"] == 1.5
    assert extremes.loc[1, "lowest_axis_id"] == 0
    assert extremes.loc[1, "lowest_mean_score"] == -2.0
    assert extremes.loc[-1, "topic_name"] == "Unassigned"


def test_topic_axis_review_tables_include_both_polarity_keywords() -> None:
    summary = pd.DataFrame(
        {
            "topic_id": [7, 7],
            "axis_id": [0, 1],
            "passage_count": [2, 2],
            "mean_score": [-1.0, 1.0],
        }
    )

    extremes, keywords = build_topic_axis_review_tables(
        summary,
        {7: "Named topic"},
        [["care", "health"], ["schools", "teachers"]],
        [["war", "borders"], ["illness", "hospital"]],
    )

    assert extremes.loc[0, "topic_name"] == "Named topic"
    assert extremes.loc[0, "highest_axis_positive_keywords"] == "schools, teachers"
    assert extremes.loc[0, "highest_axis_negative_keywords"] == "illness, hospital"
    assert extremes.loc[0, "lowest_axis_positive_keywords"] == "care, health"
    assert extremes.loc[0, "lowest_axis_negative_keywords"] == "war, borders"
    assert keywords.loc[1, "positive_end_keywords"] == "schools, teachers"


def test_topic_axis_percentiles_classify_low_and_high_within_each_axis() -> None:
    summary = pd.DataFrame(
        {
            "topic_id": list(range(10)) * 2,
            "axis_id": [0] * 10 + [1] * 10,
            "mean_score": list(range(10)) + list(reversed(range(10))),
        }
    )

    classified = classify_topic_axis_percentiles(summary)
    axis_zero = classified.query("axis_id == 0").set_index("topic_id")
    axis_one = classified.query("axis_id == 1").set_index("topic_id")

    assert axis_zero.loc[0, "axis_percentile"] == 0.1
    assert axis_zero.loc[2, "status"] == "typical"
    assert axis_zero.loc[7, "status"] == "high"
    assert axis_zero.loc[1, "status"] == "low"
    assert axis_one.loc[0, "status"] == "high"
    assert axis_one.loc[9, "status"] == "low"
