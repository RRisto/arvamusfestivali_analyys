"""Plot each native semantic segment using cached BGE vectors and saved Jev parents."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/topic-analysis/results"
SOURCE = BASE / "jev-semantic-all"
OUT = SOURCE / "segment-maps"
CONFIG = dict(
    n_neighbors=15, n_components=2, min_dist=0.15, metric="cosine", random_state=42, init="random"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    vector_root = SOURCE / "bubble-maps"
    embedding_manifest = json.loads((vector_root / "embedding-manifest.json").read_text())
    source = SOURCE / "review-segments.parquet"
    assert sha(source) == embedding_manifest["source_sha256"]
    data = pd.read_parquet(source).query("segmentation=='semantic'")
    assert data.segment_key.is_unique
    keys = embedding_manifest["segment_keys"]
    assert set(keys) == set(data.segment_key)
    data = data.set_index("segment_key").loc[keys].reset_index()
    vectors = np.load(vector_root / "segment-embeddings.npy", allow_pickle=False)
    assert vectors.shape == (len(data), 1024) and np.isfinite(vectors).all()
    expected = dict(
        source_sha256=sha(source),
        vectors_sha256=sha(vector_root / "segment-embeddings.npy"),
        segment_keys=keys,
        projection=CONFIG,
    )
    coord_file = OUT / "coordinates.npy"
    projection_manifest = OUT / "projection-manifest.json"
    if coord_file.exists() and projection_manifest.exists():
        assert json.loads(projection_manifest.read_text()) == expected, "Projection inputs changed"
        coordinates = np.load(coord_file, allow_pickle=False)
    else:
        coordinates = UMAP(**CONFIG).fit_transform(vectors)
        np.save(coord_file, coordinates, allow_pickle=False)
        projection_manifest.write_text(json.dumps(expected, indent=2) + "\n")
    assert coordinates.shape == (len(data), 2) and np.isfinite(coordinates).all()
    data["x"], data["y"] = coordinates[:, 0], coordinates[:, 1]
    metadata = pd.read_parquet(BASE / "talk-metadata/talks.parquet")
    data = data.merge(
        metadata[["episode_id", "programme_year", "organizer"]],
        on="episode_id",
        validate="many_to_one",
        sort=False,
    )
    assert data.programme_year.notna().all()
    data.drop(
        columns=["text", "jev_probabilities", "jev_high_level_probabilities"], errors="ignore"
    ).to_csv(OUT / "segment-coordinates.csv", index=False)
    data.to_parquet(OUT / "segments-with-coordinates.parquet", index=False)
    styles = json.loads((BASE / "high-level-topics/category-style.json").read_text())
    ranges = [
        (float(coordinates[:, i].min() - 0.5), float(coordinates[:, i].max() + 0.5)) for i in [0, 1]
    ]
    counts = []
    for year in [None, 2025, 2026]:
        sample = data if year is None else data.loc[data.programme_year.eq(year)]
        name = "all" if year is None else str(year)
        label = "Combined 2025 + 2026" if year is None else str(year)
        figure = go.Figure()
        static, ax = plt.subplots(figsize=(15, 10))
        for parent, group in sample.groupby("jev_high_level_id", sort=True):
            color = styles[parent]["color"]
            category = group.jev_high_level_name.iloc[0]
            hover = np.column_stack(
                [
                    group.title.map(html.escape),
                    group.programme_year,
                    group.start_seconds,
                    group.end_seconds,
                    group.jev_topic_name.map(html.escape),
                    group.jev_high_level_name.map(html.escape),
                    group.text.str.slice(0, 250).map(html.escape),
                    group.segment_key,
                ]
            )
            figure.add_trace(
                go.Scattergl(
                    x=group.x,
                    y=group.y,
                    mode="markers",
                    name=category,
                    customdata=hover,
                    marker=dict(size=6, color=color, opacity=0.85),
                    hovertemplate="<b>%{customdata[0]}</b><br>Year: %{customdata[1]}"
                    "<br>Seconds: %{customdata[2]:.1f}–%{customdata[3]:.1f}"
                    "<br>Detailed topic: %{customdata[4]}<br>Main topic: %{customdata[5]}"
                    "<br>%{customdata[6]}…<br>%{customdata[7]}<extra></extra>",
                )
            )
            ax.scatter(
                group.x, group.y, s=12, c=color, alpha=0.85, label=category, edgecolors="none"
            )
        title = f"{label}: one point per semantic segment ({len(sample):,})"
        figure.update_layout(
            title=title,
            template="plotly_white",
            height=900,
            legend=dict(title="Jev high-level topic", font=dict(size=11)),
            xaxis=dict(title="UMAP 1", range=ranges[0]),
            yaxis=dict(title="UMAP 2", range=ranges[1]),
            margin=dict(b=100),
        )
        figure.add_annotation(
            text="Shared projection from native segment BGE embeddings. "
            "Color = dictionary parent of Jev choice; gray = unassigned. "
            "Distances are approximate; points have equal size.",
            x=0,
            y=-0.1,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
        )
        figure.write_html(
            OUT / f"{name}.html",
            include_plotlyjs=True,
            config=dict(scrollZoom=True, displaylogo=False),
        )
        (OUT / f"{name}.plotly.json").write_text(figure.to_json())
        ax.set(xlim=ranges[0], ylim=ranges[1], xlabel="UMAP 1", ylabel="UMAP 2", title=title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
        static.tight_layout()
        static.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
        static.savefig(OUT / f"{name}.svg", bbox_inches="tight")
        plt.close(static)
        counts.append(
            dict(
                year=year,
                segments=len(sample),
                talks=sample.episode_id.nunique(),
                unassigned_segments=int(sample.jev_topic_id.eq(-1).sum()),
            )
        )
    (OUT / "manifest.json").write_text(
        json.dumps(
            dict(
                sources=expected,
                metadata_sha256=sha(BASE / "talk-metadata/talks.parquet"),
                cohorts=counts,
                colors="Saved high-level dictionary palette; gray unassigned",
                coordinate_scope="One combined UMAP fit of individual segment BGE vectors; "
                "year views subset coordinates without refitting",
                labels="Detailed Jev primary choice, rolled up by dictionary",
                api_requests=0,
            ),
            indent=2,
        )
        + "\n"
    )
    (OUT / "index.html").write_text("""<!doctype html><meta charset="utf-8">
<title>Jev semantic segment maps</title><style>body{font:16px system-ui;margin:20px}
iframe{width:100%;height:960px;border:0}button{padding:10px;margin:5px}</style>
<h1>One point per semantic segment</h1><p>Color = high-level Jev topic. Equal-sized
points, gray unassigned. All three views share positions. Hover for talk, time,
detailed topic and text; click legend entries to filter. 2D distances are approximate.</p>
<button onclick="document.getElementById('map').src='all.html'">Combined</button>
<button onclick="document.getElementById('map').src='2025.html'">2025</button>
<button onclick="document.getElementById('map').src='2026.html'">2026</button>
<iframe id="map" src="all.html" title="Segment scatter"></iframe>""")
    (OUT / "README.md").write_text(
        "# Native semantic segment scatter maps\n\n"
        "One equal-sized point per segment; color is its Jev high-level dictionary parent, "
        "including gray unassigned. All, 2025 and 2026 use a single UMAP projection of "
        "saved 1024-dimensional BGE segment embeddings, not topic centroids or labels "
        "as inputs. No topic refit or classifier API call.\n\n"
        "Offline HTML/Plotly JSON, PNG/SVG, coordinates.npy, full coordinate Parquet "
        "and compact CSV are saved here. Projection cache verifies source/vector "
        "hashes, ordered segment keys and settings. Hover includes title, year, "
        "timestamps, detailed/main topics and a text excerpt.\n"
    )
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
