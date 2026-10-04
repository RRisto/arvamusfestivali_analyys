"""Map Jev-assigned topic centroids with category colors and count-sized bubbles."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from umap import UMAP

from arvamusfestivali_transcripts.topic_analysis.cache import CacheIdentity, EmbeddingCache
from arvamusfestivali_transcripts.topic_analysis.embedders import ADAPTER_VERSION
from arvamusfestivali_transcripts.topic_analysis.topic_map import (
    plot_topic_embedding_map,
    topic_centroids,
)
from arvamusfestivali_transcripts.topic_analysis.types import ChunkingConfig, Passage


def main():
    import torch

    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis"
    source = base / "results/jev-semantic-all/review-segments.parquet"
    data = pd.read_parquet(source)
    data = data[data.segmentation == "semantic"].copy().reset_index(drop=True)
    assert data.segment_key.is_unique and data.jev_topic_id.notna().all()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = source.parent / "bubble-maps"
    output.mkdir(parents=True, exist_ok=True)
    model = json.loads((base / "results/segment-dataset/manifest.json").read_text())["models"][
        "semantic"
    ]
    identity = CacheIdentity(
        model_id=model["embedding_model_id"],
        model_revision=model["embedding_revision"],
        dimension=model["embedding_dimension"],
        instruction="",
        chunking=ChunkingConfig(
            min_seconds=90,
            target_seconds=300,
            max_seconds=600,
            min_words=1,
            max_words=1_000_000,
            overlap_seconds=0,
            merge_tail_seconds=0,
            max_merged_seconds=600,
        ),
        adapter_version=ADAPTER_VERSION,
    )
    cache = EmbeddingCache(base / "cache")
    vector_path = output / "segment-embeddings.npy"
    vector_manifest = output / "embedding-manifest.json"
    expected = {
        "source_sha256": source_hash,
        "cache_digest": identity.digest,
        "segment_keys": data.segment_key.tolist(),
    }
    if vector_path.exists() and vector_manifest.exists():
        assert json.loads(vector_manifest.read_text()) == expected
        vectors = np.load(vector_path, allow_pickle=False)
    else:

        def cached(row):
            passage = Passage(
                passage_id=row.passage_id,
                episode_id=row.episode_id,
                duplicate_episode_ids=tuple(json.loads(row.duplicate_episode_ids)),
                title=row.title,
                start_seconds=row.start_seconds,
                end_seconds=row.end_seconds,
                text=row.text,
                audio_sha256=row.audio_sha256,
                audio_url=row.audio_link.split("#t=", 1)[0],
                word_count=len(row.text.split()),
                cue_count=1,
            )
            vector = cache.get(identity, passage)
            if vector is None:
                raise ValueError(f"Missing matching BGE cache for {row.segment_key}")
            return vector

        with ThreadPoolExecutor(max_workers=16) as pool:
            vectors = np.stack(list(pool.map(cached, data.itertuples())))
        np.save(vector_path, vectors, allow_pickle=False)
        vector_manifest.write_text(json.dumps(expected, indent=2) + "\n")
    assert vectors.shape == (len(data), model["embedding_dimension"])
    assert np.isfinite(vectors).all()
    ids, centers = topic_centroids(vectors, data.jev_topic_id.astype(int).to_numpy())
    assigned = data[data.jev_topic_id != -1]
    table = (
        assigned.groupby("jev_topic_id")
        .agg(
            topic_name=("jev_topic_name", "first"),
            topic_key=("jev_topic_key", "first"),
            high_level_id=("jev_high_level_id", "first"),
            high_level_name=("jev_high_level_name", "first"),
            segment_count=("segment_key", "size"),
            talk_count=("episode_id", "nunique"),
        )
        .loc[ids]
        .reset_index()
        .rename(columns={"jev_topic_id": "topic_id"})
    )
    reducer = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.15,
        metric="cosine",
        random_state=42,
        init="random",
    )
    coordinates = reducer.fit_transform(centers)
    assert np.isfinite(coordinates).all()
    table["x"], table["y"] = coordinates[:, 0], coordinates[:, 1]
    table.to_csv(output / "topic-coordinates.csv", index=False)
    np.savez_compressed(
        output / "topic-centroids.npz",
        embeddings=centers,
        topic_ids=ids,
        topic_keys=table.topic_key.to_numpy(dtype=str),
    )
    # Match the category palette used by the Jev treemaps, including its sort order.
    styles = json.loads((base / "results/high-level-topics/category-style.json").read_text())
    colors = {key: value["color"] for key, value in styles.items()}
    ranges = [
        (float(coordinates[:, i].min() - 0.5), float(coordinates[:, i].max() + 0.5)) for i in (0, 1)
    ]
    for metric, column in [("segments", "segment_count"), ("talks", "talk_count")]:
        figure = go.Figure()
        for parent, group in table.groupby("high_level_id", sort=True):
            trace = plot_topic_embedding_map(group, title="", color=colors[parent]).data[0]
            trace.marker.symbol = styles[parent]["symbol"]
            trace.marker.opacity = 1
            trace.marker.size = group[column].to_numpy()
            trace.marker.sizeref = 2 * int(table[column].max()) / 52**2
            trace.name = group.high_level_name.iloc[0]
            trace.showlegend = True
            figure.add_trace(trace)
        figure.update_layout(
            title=f"Jev detailed topics · bubble area = {metric}",
            template="plotly_white",
            height=850,
            dragmode="pan",
            hovermode="closest",
            legend={"title": "High-level category", "font": {"size": 11}},
            margin={"t": 70, "b": 80, "l": 30, "r": 30},
            xaxis={"title": "UMAP 1", "range": ranges[0], "showgrid": False, "zeroline": False},
            yaxis={"title": "UMAP 2", "range": ranges[1], "showgrid": False, "zeroline": False},
        )
        figure.add_annotation(
            text="Same positions in both views · Color = high-level category · "
            "Click legend to filter · 2D distances are approximate",
            x=0,
            y=-0.1,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
        )
        figure.write_html(
            output / f"{metric}.html",
            include_plotlyjs=True,
            config={"scrollZoom": True, "displaylogo": False},
        )
        (output / f"{metric}.plotly.json").write_text(figure.to_json())
    manifest = {
        "source_sha256": source_hash,
        "segments": len(data),
        "assigned_segments": len(assigned),
        "topics": len(ids),
        "categories": table.high_level_id.nunique(),
        "centroid_method": "Normalized mean of original 1024-dimensional BGE vectors "
        "for Jev primary-topic members, equal segment weights",
        "embedding_model": model["embedding_model_id"],
        "embedding_revision": model["embedding_revision"],
        "cache_digest": identity.digest,
        "projection": {
            "method": "UMAP",
            "n_neighbors": 15,
            "n_components": 2,
            "min_dist": 0.15,
            "metric": "cosine",
            "random_state": 42,
            "init": "random",
        },
        "size": "Bubble area, not diameter, proportional to segment count "
        "or distinct episode count",
        "colors": "Dictionary high-level categories, same palette as treemaps",
        "unassigned": "389 unassigned segments excluded; no coherent centroid implied",
        "coordinate_scope": "Jev member centroids, not original model-cluster centroids. "
        "Both views share the same coordinates.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (
        output / "index.html"
    ).write_text("""<!doctype html><meta charset="utf-8"><title>Jev bubble maps</title>
<style>body{font:16px system-ui;margin:20px}button{padding:10px;margin-right:10px}
iframe{width:100%;height:900px;border:0}</style><h1>Jev topic bubble maps</h1>
<p>One bubble per detailed topic. Color shows its high-level category. Hover for topic
names and both counts; click legend entries to filter categories. Zoom and pan to explore.</p>
<button onclick="document.getElementById('chart').src='segments.html'">Size by segments</button>
<button onclick="document.getElementById('chart').src='talks.html'">Size by distinct talks</button>
<p>Positions are a shared UMAP projection of BGE centroids from Jev-assigned segments.
2D distances are approximate. Unassigned content is excluded.</p>
<iframe id="chart" src="segments.html" title="Jev bubble map"></iframe>""")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
