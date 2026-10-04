"""Export topic-count, native timeline and temporal correspondence charts for refitted pilots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from arvamusfestivali_transcripts.topic_analysis.experiment_dynamics import interval_topic_overlap
from arvamusfestivali_transcripts.topic_analysis.segment_review import (
    _color,
    document_catalog,
    segmentation_experiment_review_html,
)


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis/results"
    output = base / "segmentation-experiments/topic-models"
    original = pd.read_parquet(base / "jev-semantic-sample/review-segments.parquet")
    summaries = pd.read_csv(output / "comparison-summary.csv")
    stability = pd.read_csv(output / "seed-stability.csv")
    variants = ["original", "short3", "cue5", "sentence3", "sentence5"]
    catalog = document_catalog(original)
    links = []
    all_overlap = []
    counts = go.Figure()
    for seed in [42, 7]:
        rows = summaries[summaries.seed == seed]
        counts.add_trace(
            go.Bar(
                x=rows.variant,
                y=rows.topic_count,
                name=f"Seed {seed}",
                customdata=rows[["segments", "outlier_time_pct", "topic_changes"]].to_numpy(),
                hovertemplate="%{x}: %{y} topics<br>%{customdata[0]} segments<br>"
                "%{customdata[1]:.1f}% unassigned time<br>"
                "%{customdata[2]} topic changes<extra></extra>",
            )
        )
    counts.update_layout(
        title="Independent three-talk fits: topic counts",
        barmode="group",
        yaxis_title="Discovered topics (excluding -1)",
    )
    counts.write_html(output / "topic-counts.html", include_plotlyjs=True)
    links.append('<li><a href="topic-counts.html">Topic counts by variant and seed</a></li>')
    for seed in [42, 7]:
        fits = {
            variant: pd.read_parquet(output / variant / f"seed-{seed}" / "review-segments.parquet")
            for variant in variants
        }
        for index in [0, 1, 3]:
            episode_id = catalog.iloc[index].episode_id
            figure = go.Figure()
            for variant, data in fits.items():
                selected = data[data.episode_id.astype(str) == episode_id]
                for topic, group in selected.groupby("cluster_topic_id"):
                    key = group.cluster_topic_key.iloc[0]
                    custom = group[
                        [
                            "cluster_topic_name",
                            "start_seconds",
                            "end_seconds",
                            "cluster_membership_strength",
                            "segment_key",
                        ]
                    ].to_numpy()
                    figure.add_trace(
                        go.Bar(
                            x=(group.end_seconds - group.start_seconds) / 60,
                            base=group.start_seconds / 60,
                            y=[variant] * len(group),
                            orientation="h",
                            marker_color=_color(key, int(topic)),
                            name=f"{variant}: {topic}",
                            showlegend=False,
                            customdata=custom,
                            hovertemplate="%{y}: %{customdata[0]}<br>"
                            "%{customdata[1]:.1f}–%{customdata[2]:.1f}s<br>"
                            "Membership %{customdata[3]:.3f}<br>%{customdata[4]}<extra></extra>",
                        )
                    )
            figure.update_layout(
                title=f"{catalog.iloc[index].talk_name} · seed {seed}",
                barmode="overlay",
                height=450,
                xaxis_title="Recording time (minutes)",
                yaxis={"categoryorder": "array", "categoryarray": list(reversed(variants))},
                annotations=[
                    dict(
                        text="Colors identify topics within a fit; gray is unassigned. "
                        "Topic IDs/colors are not aligned across fits.",
                        x=0,
                        y=-0.22,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        xanchor="left",
                    )
                ],
            )
            filename = f"timeline-document-{index}-seed-{seed}.html"
            figure.write_html(output / filename, include_plotlyjs=True)
            links.append(
                f'<li><a href="{filename}">Document {index}: native topic timelines, '
                f"seed {seed}</a></li>"
            )
            for left, right in [
                ("short3", "sentence3"),
                ("cue5", "sentence5"),
                ("sentence3", "sentence5"),
            ]:
                filename = f"review-document-{index}-{left}-vs-{right}-seed-{seed}.html"
                html = segmentation_experiment_review_html(
                    original, fits, index, left=left, right=right
                )
                (output / filename).write_text('<!doctype html><meta charset="utf-8">' + html)
                links.append(
                    f'<li><a href="{filename}">Document {index}: {left} vs {right}, '
                    f"seed {seed}</a></li>"
                )
        for left, right in [
            ("original", "short3"),
            ("short3", "sentence3"),
            ("cue5", "sentence5"),
            ("sentence3", "sentence5"),
        ]:
            overlap = interval_topic_overlap(fits[left], fits[right])
            overlap["left_variant"] = left
            overlap["right_variant"] = right
            overlap["seed"] = seed
            all_overlap.append(overlap)
            aggregated = overlap.groupby(
                ["left_topic_key", "left_topic_name", "right_topic_key", "right_topic_name"],
                as_index=False,
            ).overlap_seconds.sum()
            left_nodes = aggregated[["left_topic_key", "left_topic_name"]].drop_duplicates()
            right_nodes = aggregated[["right_topic_key", "right_topic_name"]].drop_duplicates()
            keys = left_nodes.left_topic_key.tolist() + right_nodes.right_topic_key.tolist()
            labels = [f"{left}: {name}" for name in left_nodes.left_topic_name] + [
                f"{right}: {name}" for name in right_nodes.right_topic_name
            ]
            mapping = {key: i for i, key in enumerate(keys)}
            chart = go.Figure(
                go.Sankey(
                    node={"label": labels},
                    link={
                        "source": [mapping[key] for key in aggregated.left_topic_key],
                        "target": [mapping[key] for key in aggregated.right_topic_key],
                        "value": aggregated.overlap_seconds / 60,
                        "hovertemplate": "%{value:.2f} overlapping minutes<extra></extra>",
                    },
                )
            )
            chart.update_layout(title=f"{left} → {right} · seed {seed}: temporal topic overlap")
            filename = f"overlap-{left}-vs-{right}-seed-{seed}.html"
            chart.write_html(output / filename, include_plotlyjs=True)
            links.append(
                f'<li><a href="{filename}">{left} vs {right}: topic overlap, seed {seed}</a></li>'
            )
    pd.concat(all_overlap, ignore_index=True).to_csv(
        output / "topic-overlap-seconds.csv", index=False
    )
    display_columns = [
        "variant",
        "seed",
        "segments",
        "topic_count",
        "outlier_time_pct",
        "topic_changes",
        "mean_changes_per_hour",
        "original_cosine_silhouette",
    ]
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Refitted topic models</title>'
        "<style>body{font:15px system-ui;margin:30px}td,th{padding:7px;border:1px solid #ddd}"
        "table{border-collapse:collapse}</style><h1>Refitted three-talk topic models</h1>"
        "<p>Same BGE/BERTopic leaf-local-6-2 configuration; each variant is fitted independently. "
        "Topic IDs are scoped to each fit. Keyword labels are fresh. Jev continues to use the "
        "old full-corpus vocabulary, so numerical ID agreement with these fits is meaningless.</p>"
        "<p>Topic changes include transitions to/from unassigned content. Temporal overlap links "
        "measure shared recording time, not classification agreement or identical subjects.</p>"
        + summaries[display_columns].to_html(index=False, float_format=lambda x: f"{x:.3f}")
        + "<h2>Seed stability</h2>"
        + stability.to_html(index=False, float_format=lambda x: f"{x:.3f}")
        + "<h2>Charts and native segment review</h2><ul>"
        + "".join(links)
        + "</ul>"
    )
    print(
        "Exported interactive topic-count chart, 6 timelines, "
        "8 overlap charts and 18 review pages.",
        flush=True,
    )


if __name__ == "__main__":
    main()
