"""Verify topic centroid geometry, noise exclusion and count-based marker areas."""

import numpy as np
import pandas as pd
import pytest

from arvamusfestivali_transcripts.topic_analysis.topic_map import (
    plot_topic_embedding_map,
    topic_centroids,
)


def test_centroids_average_assigned_embeddings_and_exclude_noise():
    ids, centers = topic_centroids(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]]),
        np.array([0, 0, 1, -1]),
    )
    assert ids.tolist() == [0, 1]
    np.testing.assert_allclose(centers, [[2**-0.5, 2**-0.5], [0, -1]])
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1)


def test_map_sizes_track_segment_counts_and_hover_includes_names():
    frame = pd.DataFrame(
        {
            "topic_id": [0, 1],
            "topic_name": ["Tööohutus", "Õpilased"],
            "segment_count": [10, 40],
            "talk_count": [2, 5],
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
        }
    )
    figure = plot_topic_embedding_map(frame, title="Topics", color="blue")
    assert len(figure.data) == 1
    assert figure.data[0].marker.sizemode == "area"
    np.testing.assert_array_equal(figure.data[0].marker.size, [10, 40])
    assert figure.data[0].customdata[0][1] == "Tööohutus"
    assert "customdata[1]" in figure.data[0].hovertemplate
    other = plot_topic_embedding_map(
        frame.iloc[:1], title="Other", color="blue", max_segment_count=40
    )
    assert other.data[0].marker.sizeref == figure.data[0].marker.sizeref


def test_invalid_or_empty_topic_embeddings_fail():
    with pytest.raises(ValueError, match="align"):
        topic_centroids(np.ones((2, 3)), np.array([0]))
    with pytest.raises(ValueError, match="No assigned"):
        topic_centroids(np.ones((2, 3)), np.array([-1, -1]))
