"""Topic correspondence uses native interval intersections and preserves run scopes."""

import pandas as pd
import pytest

from arvamusfestivali_transcripts.topic_analysis.experiment_dynamics import interval_topic_overlap


def frame(prefix, intervals):
    return pd.DataFrame(
        [
            dict(
                episode_id="1",
                start_seconds=start,
                end_seconds=end,
                cluster_topic_key=f"{prefix}:{topic}",
                cluster_topic_name=f"{prefix} topic {topic}",
            )
            for start, end, topic in intervals
        ]
    )


def test_different_boundaries_are_compared_by_seconds_not_numeric_ids():
    left = frame("left", [(0, 60, 0), (60, 100, 1)])
    right = frame("right", [(0, 20, 0), (20, 80, 1), (80, 100, -1)])
    result = interval_topic_overlap(left, right)
    pairs = {(r.left_topic_key, r.right_topic_key): r.overlap_seconds for r in result.itertuples()}
    assert pairs == {
        ("left:0", "right:0"): 20,
        ("left:0", "right:1"): 40,
        ("left:1", "right:1"): 20,
        ("left:1", "right:-1"): 20,
    }
    assert result.overlap_seconds.sum() == 100


def test_overlapping_native_segments_are_rejected():
    left = frame("left", [(0, 70, 0), (60, 100, 1)])
    right = frame("right", [(0, 100, 0)])
    with pytest.raises(ValueError, match="nonoverlapping"):
        interval_topic_overlap(left, right)
