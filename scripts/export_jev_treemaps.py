"""Export Jev category/topic treemaps with explicit distinct-talk count semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def build_treemap(data: pd.DataFrame, metric: str) -> tuple[go.Figure, pd.DataFrame]:
    if metric not in {"segments", "talks"}:
        raise ValueError("Metric must be segments or talks")
    rows = []
    for parent, group in data.groupby("jev_high_level_id", sort=True):
        category_id = f"category:{parent}"
        children = []
        for key, topic in group.groupby("jev_topic_key", sort=True):
            children.append(
                {
                    "id": f"topic:{key}",
                    "parent": category_id,
                    "label": topic.jev_topic_name.iloc[0],
                    "segments": len(topic),
                    "talks": topic.episode_id.nunique(),
                    "kind": "Detailed topic",
                }
            )
        rows.extend(children)
        rows.append(
            {
                "id": category_id,
                "parent": "root",
                "label": group.jev_high_level_name.iloc[0],
                "segments": len(group),
                "talks": group.episode_id.nunique(),
                "kind": "High-level category",
            }
        )
    rows.append(
        {
            "id": "root",
            "parent": "",
            "label": "Kõik arutelud",
            "segments": len(data),
            "talks": data.episode_id.nunique(),
            "kind": "All categories",
        }
    )
    table = pd.DataFrame(rows)
    weights = dict(zip(table.id, table[metric]))
    # Distinct talk counts are nonadditive: parents aggregate child topic-talk memberships.
    for category_id in table.loc[table.kind == "High-level category", "id"]:
        weights[category_id] = table.loc[table.parent == category_id, metric].sum()
    weights["root"] = sum(weights[k] for k in table.loc[table.parent == "root", "id"])
    table["area_weight"] = table.id.map(weights)
    style_path = (
        Path(__file__).resolve().parents[1]
        / "data/topic-analysis/results/high-level-topics/category-style.json"
    )
    styles = json.loads(style_path.read_text())
    colors = {f"category:{key}": value["color"] for key, value in styles.items()}
    table["color"] = [colors.get(r.id, colors.get(r.parent, "#eef1f5")) for r in table.itertuples()]
    label = "segments" if metric == "segments" else "distinct talks"
    custom = table[["segments", "talks", "area_weight", "kind"]].to_numpy()
    figure = go.Figure(
        go.Treemap(
            ids=table.id,
            labels=table.label,
            parents=table.parent,
            values=table.area_weight,
            branchvalues="total",
            customdata=custom,
            marker={"colors": table.color},
            sort=True,
            texttemplate="%{label}<br>%{customdata["
            + ("0" if metric == "segments" else "1")
            + "]} "
            + label,
            hovertemplate="<b>%{label}</b><br>%{customdata[3]}<br>Segments: %{customdata[0]}<br>"
            "Distinct talks: %{customdata[1]}<br>Area weight: %{customdata[2]}<extra></extra>",
            pathbar={"visible": True},
            tiling={"pad": 3},
        )
    )
    note = (
        "Area = segment count; category counts sum their topics."
        if metric == "segments"
        else "Topic area = distinct talks. Category counts deduplicate talks; "
        "category area sums topic–talk memberships."
    )
    figure.update_layout(
        title={"text": f"Jev topics · {label}<br><sup>{note}</sup>"},
        height=900,
        margin={"t": 100, "l": 10, "r": 10, "b": 10},
        font={"family": "Arial, sans-serif", "size": 14},
    )
    assert table.loc[table.parent == "root", "segments"].sum() == len(data)
    assert table.loc[table.kind == "Detailed topic", "segments"].sum() == len(data)
    for row in table[table.kind != "Detailed topic"].itertuples():
        assert weights[row.id] == table.loc[table.parent == row.id, "area_weight"].sum()
    return figure, table


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis/results/jev-semantic-all"
    source = base / "review-segments.parquet"
    data = pd.read_parquet(source)
    data = data[data.segmentation == "semantic"].copy()
    assert data.segment_key.is_unique and data.jev_topic_key.notna().all()
    assert data.jev_high_level_id.notna().all()
    output = base / "treemaps"
    output.mkdir(parents=True, exist_ok=True)
    for metric in ("segments", "talks"):
        figure, table = build_treemap(data, metric)
        figure.write_html(output / f"{metric}.html", include_plotlyjs=True)
        (output / f"{metric}.plotly.json").write_text(figure.to_json())
        table.drop(columns="color").to_csv(output / f"{metric}-counts.csv", index=False)
    (output / "index.html").write_text("""<!doctype html><meta charset="utf-8">
<title>Jev topic treemaps</title><style>
body{font:16px system-ui;margin:20px}
button{padding:10px;margin-right:10px}iframe{width:100%;height:960px;border:0}</style>
<h1>Jev topic treemaps</h1><p>Click a category to explore its detailed topics;
use the path bar to return. Gray is unassigned.</p>
<button onclick="document.getElementById('chart').src='segments.html'">Segment counts</button>
<button onclick="document.getElementById('chart').src='talks.html'">Distinct talk counts</button>
<p>In the talks view, each detailed topic counts a talk once.
Category labels count the union of talks, while category areas sum their topic counts:
one talk may occur in several topics. Hover shows both counts and area weight.</p>
<iframe id="chart" src="segments.html" title="Jev topic treemap"></iframe>""")
    manifest = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "segments": len(data),
        "talks": data.episode_id.nunique(),
        "detailed_topics_including_unassigned": data.jev_topic_key.nunique(),
        "categories_including_unassigned": data.jev_high_level_id.nunique(),
        "talk_count_rule": "Distinct episode IDs per node. Category and root areas sum "
        "child topic-talk memberships; displayed category/root counts use unions.",
        "assignment_source": "Detailed Jev primary topic choice; high-level dictionary rollup",
        "unassigned": "Included as a separate gray branch",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
