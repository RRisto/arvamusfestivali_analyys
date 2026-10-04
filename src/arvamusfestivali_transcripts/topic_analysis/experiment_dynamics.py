"""Temporal topic correspondence between independent fits on native segment boundaries."""

from __future__ import annotations

import pandas as pd


def interval_topic_overlap(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Aggregate overlap seconds by run-scoped topic pair; this is not prediction agreement."""
    rows = []
    for episode_id in sorted(set(left.episode_id.astype(str)) & set(right.episode_id.astype(str))):
        a = left[left.episode_id.astype(str) == episode_id].sort_values("start_seconds")
        b = right[right.episode_id.astype(str) == episode_id].sort_values("start_seconds")
        for frame in [a, b]:
            if (frame.end_seconds <= frame.start_seconds).any():
                raise ValueError("Segments require positive intervals")
            if (
                len(frame) > 1
                and (
                    frame.end_seconds.to_numpy()[:-1] > frame.start_seconds.to_numpy()[1:] + 1e-6
                ).any()
            ):
                raise ValueError("Native segments must be nonoverlapping")
        items_a, items_b = list(a.itertuples()), list(b.itertuples())
        i, j = 0, 0
        while i < len(items_a) and j < len(items_b):
            first, second = items_a[i], items_b[j]
            seconds = min(first.end_seconds, second.end_seconds) - max(
                first.start_seconds, second.start_seconds
            )
            if seconds > 0:
                rows.append(
                    {
                        "episode_id": episode_id,
                        "left_topic_key": first.cluster_topic_key,
                        "left_topic_name": first.cluster_topic_name,
                        "right_topic_key": second.cluster_topic_key,
                        "right_topic_name": second.cluster_topic_name,
                        "overlap_seconds": seconds,
                    }
                )
            if first.end_seconds <= second.end_seconds:
                i += 1
            else:
                j += 1
    columns = [
        "episode_id",
        "left_topic_key",
        "left_topic_name",
        "right_topic_key",
        "right_topic_name",
        "overlap_seconds",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).groupby(columns[:-1], as_index=False).overlap_seconds.sum()


def plot_variant_topic_timelines(
    variants: dict[str, pd.DataFrame],
    episode_id: str,
    *,
    title: str = "",
):
    """Compact native timelines sharing one recording axis; topic colors are run-scoped."""
    import plotly.graph_objects as go

    from .segment_review import _color

    figure = go.Figure()
    for variant, data in variants.items():
        if "cluster_topic_id" not in data.columns:
            continue
        selected = data[data.episode_id.astype(str) == str(episode_id)]
        for topic, group in selected.groupby("cluster_topic_id"):
            key = group.cluster_topic_key.iloc[0]
            figure.add_trace(
                go.Bar(
                    x=(group.end_seconds - group.start_seconds) / 60,
                    base=group.start_seconds / 60,
                    y=[variant] * len(group),
                    orientation="h",
                    name=f"{variant}: {topic}",
                    showlegend=False,
                    marker_color=_color(key, int(topic)),
                    customdata=group[
                        [
                            "cluster_topic_name",
                            "start_seconds",
                            "end_seconds",
                            "cluster_membership_strength",
                            "segment_key",
                        ]
                    ].to_numpy(),
                    hovertemplate="%{y}: %{customdata[0]}<br>"
                    "%{customdata[1]:.1f}–%{customdata[2]:.1f}s<br>"
                    "Membership: %{customdata[3]:.3f}<extra></extra>",
                )
            )
    figure.update_layout(
        title=title or f"Episode {episode_id}: refitted topic timelines",
        barmode="overlay",
        height=400,
        xaxis_title="Recording time (minutes)",
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(variants))},
        annotations=[
            dict(
                text="Gray is unassigned. Topic IDs and colors are local to each fit.",
                x=0,
                y=-0.25,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            )
        ],
    )
    return figure
