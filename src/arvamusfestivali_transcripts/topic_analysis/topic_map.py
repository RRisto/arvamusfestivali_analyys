"""Topic centroids and interactive maps with segment-count-proportional marker areas."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def topic_centroids(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the mean original-space embedding of each non-outlier topic."""
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("Embedding rows must align with topic labels")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite")
    ids = np.array(sorted(set(labels.tolist()) - {-1}), dtype=int)
    if not len(ids):
        raise ValueError("No assigned topics to map")
    centers = np.stack([embeddings[labels == topic].mean(axis=0) for topic in ids])
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A topic has a zero-norm centroid")
    return ids, centers / norms


def plot_topic_embedding_map(
    frame: pd.DataFrame, *, title: str, color: str, max_segment_count: int | None = None
) -> go.Figure:
    """One marker per topic; marker area, rather than diameter, tracks segment count."""
    if not frame.segment_count.gt(0).all():
        raise ValueError("Topic segment counts must be positive")
    hover = np.column_stack(
        [
            frame.topic_id,
            frame.topic_name.map(html.escape),
            frame.segment_count,
            frame.talk_count,
        ]
    )
    figure = go.Figure(
        go.Scatter(
            x=frame.x,
            y=frame.y,
            mode="markers",
            customdata=hover,
            marker={
                "size": frame.segment_count,
                "sizemode": "area",
                "sizeref": 2 * (max_segment_count or frame.segment_count.max()) / 42**2,
                "color": color,
                "opacity": 0.8,
                "line": {"width": 1, "color": "white"},
            },
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>Topic ID: %{customdata[0]}"
                "<br>Segments: %{customdata[2]}<br>Talks: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=800,
        dragmode="pan",
        hovermode="closest",
        margin={"l": 30, "r": 30, "b": 90, "t": 70},
        xaxis={"title": "UMAP 1", "showgrid": False, "zeroline": False},
        yaxis={"title": "UMAP 2", "showgrid": False, "zeroline": False},
        showlegend=False,
    )
    return figure
