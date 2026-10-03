"""Map both recommended models' original BGE topic centroids in a shared 2D projection."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

from arvamusfestivali_transcripts.topic_analysis.cache import CacheIdentity, EmbeddingCache
from arvamusfestivali_transcripts.topic_analysis.embedders import ADAPTER_VERSION
from arvamusfestivali_transcripts.topic_analysis.topic_map import (
    plot_topic_embedding_map,
    topic_centroids,
)
from arvamusfestivali_transcripts.topic_analysis.types import ChunkingConfig, Passage


def export_maps(dataset_root: Path, cache_root: Path, output_root: Path) -> None:
    segments = pd.read_parquet(dataset_root / "all-segments.parquet")
    metadata = json.loads((dataset_root / "manifest.json").read_text())
    output_root.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(cache_root)
    frames, vectors = [], []
    for mode in ("semantic", "fixed"):
        subset = segments[segments.segmentation == mode].copy()
        model = metadata["models"][mode]
        chunking = (
            ChunkingConfig(**model["segmentation_parameters"])
            if mode == "fixed"
            else ChunkingConfig(
                min_seconds=90,
                target_seconds=300,
                max_seconds=600,
                min_words=1,
                max_words=1_000_000,
                overlap_seconds=0,
                merge_tail_seconds=0,
                max_merged_seconds=600,
            )
        )
        identity = CacheIdentity(
            model_id=model["embedding_model_id"],
            model_revision=model["embedding_revision"],
            dimension=model["embedding_dimension"],
            instruction="",
            chunking=chunking,
            adapter_version=ADAPTER_VERSION,
        )

        def cached_row(row):
            # Counts are not part of embedding cache keys. This object is for cache lookup only.
            passage = Passage(
                passage_id=row.passage_id,
                episode_id=row.episode_id,
                duplicate_episode_ids=tuple(json.loads(row.duplicate_episode_ids)),
                title=row.title,
                start_seconds=float(row.start_seconds),
                end_seconds=float(row.end_seconds),
                text=row.text,
                audio_sha256=row.audio_sha256,
                audio_url=row.audio_link.split("#t=", 1)[0],
                word_count=len(row.text.split()),
                cue_count=1,
            )
            vector = cache.get(identity, passage)
            if vector is None:
                raise ValueError(f"Missing cached embedding: {mode} {row.passage_id}")
            return vector

        with ThreadPoolExecutor(max_workers=16) as pool:
            rows = list(pool.map(cached_row, subset.itertuples()))
        ids, centers = topic_centroids(np.stack(rows), subset.topic_id.to_numpy())
        assigned = subset[subset.topic_id != -1]
        statistics = (
            assigned.groupby("topic_id")
            .agg(
                topic_name=("topic_name", "first"),
                segment_count=("passage_id", "size"),
                talk_count=("episode_id", "nunique"),
            )
            .loc[ids]
            .reset_index()
        )
        statistics["segmentation"] = mode
        statistics["model_key"] = subset.model_key.iloc[0]
        frames.append(statistics)
        vectors.append(centers)
        print(f"{mode}: {len(rows)} cached embeddings; {len(ids)} topics", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    centroids = np.concatenate(vectors)
    # Joint projection gives the two maps comparable coordinates and axis ranges.
    reducer = UMAP(
        n_neighbors=15,
        n_components=2,
        metric="cosine",
        min_dist=0.15,
        random_state=42,
        init="random",
    )
    coordinates = reducer.fit_transform(centroids)
    combined["x"], combined["y"] = coordinates[:, 0], coordinates[:, 1]
    np.savez_compressed(
        output_root / "topic-centroids.npz",
        embeddings=centroids,
        topic_ids=combined.topic_id.to_numpy(),
        model_keys=combined.model_key.to_numpy(dtype=str),
    )
    combined.to_csv(output_root / "topic-coordinates.csv", index=False)
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    pad = (high - low) * 0.07
    for mode, color in (("semantic", "#2563eb"), ("fixed", "#d97706")):
        frame = combined[combined.segmentation == mode]
        figure = plot_topic_embedding_map(
            frame,
            title=f"{mode.title()} BGE · {len(frame)} topics · leaf-local-6-2",
            color=color,
            max_segment_count=int(combined.segment_count.max()),
        )
        figure.update_xaxes(range=[float(low[0] - pad[0]), float(high[0] + pad[0])])
        figure.update_yaxes(range=[float(low[1] - pad[1]), float(high[1] + pad[1])])
        figure.add_annotation(
            x=0,
            y=-0.09,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            text="Dot area ∝ assigned segments · Hover for topic name · Pan and zoom to explore",
        )
        figure.write_html(
            output_root / f"{mode}-bge-topic-map.html",
            include_plotlyjs=True,
            full_html=True,
            config={"scrollZoom": True, "displaylogo": False},
        )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "topic_counts": {frame.segmentation.iloc[0]: len(frame) for frame in frames},
                "centroid_method": (
                    "Normalized mean original BGE segment embedding; equal segment weights."
                ),
                "projection": {
                    "method": "joint UMAP",
                    "n_neighbors": 15,
                    "n_components": 2,
                    "metric": "cosine",
                    "min_dist": 0.15,
                    "random_state": 42,
                },
                "marker_size": "Area proportional to assigned segment count.",
                "outliers": "Topic -1 excluded: it is unassigned content, not a coherent topic.",
                "interpretation": (
                    "Nearby topics have similar centroids; 2D distances are approximate."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (output_root / "README.md").write_text(
        "# Topic embedding maps\n\n"
        "Open `semantic-bge-topic-map.html` (236 topics) or "
        "`fixed-bge-topic-map.html` (346 topics) in a browser. "
        "Both files are standalone and work offline. Hover shows topic names and counts; "
        "dot area is proportional to the number of assigned segments.\n\n"
        "The maps share a UMAP projection of normalized mean BGE embeddings for all 582 topics. "
        "The two panels use the same coordinates and axis ranges. Topic -1 is excluded. "
        "Distances in 2D are approximate. Names come from the dataset topic_name field, "
        "including available LLM names.\n\n"
        "Original 1,024-dimensional topic centroids are in `topic-centroids.npz`; "
        "labels, counts and plotted coordinates are in `topic-coordinates.csv`.\n"
    )
    print(f"Saved maps: {output_root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    export_maps(args.dataset_root, args.cache_root, args.output_root)
