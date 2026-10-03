"""Build offline charts for both named, saved topic models without refitting."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_charts(data: pd.DataFrame, mode: str) -> dict[str, go.Figure]:
    frame = data[data.segmentation == mode].copy()
    frame["label"] = [
        f"{int(i)} · {name}" for i, name in zip(frame.topic_id, frame.topic_name, strict=True)
    ]
    sizes = (
        frame.groupby(["topic_id", "label"], sort=True)
        .agg(segments=("segment_key", "size"), talks=("episode_id", "nunique"))
        .reset_index()
    )
    sizes = sizes.sort_values("segments", ascending=False)
    counts = go.Figure(
        go.Bar(
            x=sizes.segments,
            y=sizes.label.map(html.escape),
            orientation="h",
            customdata=sizes[["topic_id", "talks"]],
            hovertemplate="%{y}<br>%{x} segments · %{customdata[1]} talks<extra></extra>",
        )
    )
    counts.update_layout(
        title=f"{mode.title()}: segment counts by topic",
        yaxis={"autorange": "reversed"},
        height=max(700, 24 * len(sizes)),
        margin={"l": 380},
        xaxis_title="Segments",
    )

    # Duration shares use interval unions, preventing overlapping fixed windows
    # from being double-counted inside a topic. Topics may themselves overlap.
    def union(group):
        total, last = 0.0, -np.inf
        for start, end in sorted(zip(group.start_seconds, group.end_seconds, strict=True)):
            total += max(0.0, end - max(start, last))
            last = max(last, end)
        return total

    topics = sorted(frame.topic_id.unique())
    talks = frame.drop_duplicates("episode_id").set_index("episode_id").sort_values("talk_name")
    labels = frame.drop_duplicates("topic_id").set_index("topic_id").label
    matrix = pd.DataFrame(0.0, index=talks.index, columns=topics)
    totals = {key: union(group) for key, group in frame.groupby("episode_id")}
    for (episode, topic), group in frame.groupby(["episode_id", "topic_id"]):
        matrix.loc[episode, topic] = union(group) / totals[episode]
    heatmap = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[html.escape(labels[t]) for t in topics],
            y=talks.talk_name.map(html.escape),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "Coverage"},
            hovertemplate="%{y}<br>%{x}<br>%{z:.1%} of analyzed recording<extra></extra>",
        )
    )
    heatmap.update_layout(
        title=f"{mode.title()}: topic coverage by talk",
        height=max(800, 22 * len(talks)),
        margin={"l": 350, "b": 180},
        xaxis_title="Topic ID and name",
        yaxis_title="Talk",
    )
    timeline = go.Figure()
    for index, (episode, row) in enumerate(talks.iterrows()):
        selected = frame[frame.episode_id == episode].sort_values("start_seconds")
        metadata = np.column_stack(
            [
                selected.label.map(html.escape),
                selected.confidence,
                selected.start_seconds,
                selected.end_seconds,
                selected.text.str.slice(0, 200).map(html.escape),
                selected.audio_link.map(html.escape),
                selected.segment_key,
            ]
        )
        timeline.add_trace(
            go.Bar(
                x=(selected.end_seconds - selected.start_seconds) / 60,
                base=selected.start_seconds / 60,
                y=selected.label.map(html.escape),
                orientation="h",
                visible=index == 0,
                name=html.escape(row.talk_name),
                customdata=metadata,
                marker_color=[
                    "#777777" if t == -1 else f"hsl({int(t) * 137.5 % 360},55%,48%)"
                    for t in selected.topic_id
                ],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]:.1f}–"
                "%{customdata[3]:.1f} seconds<br>Membership: %{customdata[1]:.3f}"
                "<br>%{customdata[4]}<br>%{customdata[5]}<extra></extra>",
            )
        )
    buttons = [
        dict(
            label=row.talk_name,
            method="update",
            args=[
                {"visible": [j == i for j in range(len(talks))]},
                {"title": f"{mode.title()}: {html.escape(row.talk_name)}"},
            ],
        )
        for i, (_, row) in enumerate(talks.iterrows())
    ]
    timeline.update_layout(
        title=f"{mode.title()}: {html.escape(talks.iloc[0].talk_name)}",
        updatemenus=[{"buttons": buttons, "x": 0, "y": 1.18}],
        height=850,
        showlegend=False,
        barmode="overlay",
        margin={"l": 380, "t": 150},
        xaxis_title="Recording time (minutes)",
        yaxis={"autorange": "reversed"},
        yaxis_title="Topic ID and name",
    )
    for figure in [counts, heatmap, timeline]:
        figure.update_layout(template="plotly_white")
    return {"topic-sizes": counts, "talk-topic-heatmap": heatmap, "talk-topic-timeline": timeline}


def export(dataset: Path, output: Path):
    data = pd.read_parquet(dataset)
    output.mkdir(parents=True, exist_ok=True)
    links = []
    for mode in ["semantic", "fixed"]:
        for name, figure in build_charts(data, mode).items():
            filename = f"{mode}-{name}.html"
            figure.write_html(
                output / filename,
                include_plotlyjs=True,
                config={"scrollZoom": True, "displaylogo": False},
            )
            links.append(f'<li><a href="{filename}">{mode.title()} — {name}</a></li>')
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        "<title>Named topic charts</title><style>body{font:18px system-ui;max-width:900px;"
        "margin:60px auto}li{margin:18px}</style><h1>Named topic charts</h1>"
        "<p>236 semantic topics and 346 fixed topics. Labels retain IDs and use the new "
        "Estonian names. Timeline: choose a talk from the dropdown. All charts work offline.</p>"
        "<ul>" + "".join(links) + "</ul><p>Coverage uses unioned segment intervals; "
        "overlapping topics may sum above 100%. Topic −1 is unassigned content.</p>"
        '<p><a href="../topic-maps/index.html">Topic embedding maps</a> · '
        '<a href="../topic-names/topic-review.html">Naming evidence</a></p>'
    )
    (output / "README.md").write_text(
        "# Charts with Estonian topic names\n\n"
        "Open index.html. Both selected BGE models have segment-count charts, "
        "talk/topic coverage heatmaps and talk timelines with a recording selector. "
        "All pages are standalone. Hover labels retain topic IDs, names and available "
        "segment timestamps and cluster membership strengths.\n\n"
        "Regenerate without fitting models or API calls:\n\n"
        "```bash\nPYTHONPATH=src python scripts/export_named_topic_charts.py "
        "--dataset data/topic-analysis/results/segment-dataset/all-segments.parquet "
        "--output data/topic-analysis/results/named-topic-charts\n```\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.dataset, args.output)
