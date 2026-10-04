"""Build a separately namespaced short-segment pilot from cached atomic BGE vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from arvamusfestivali_transcripts.topic_analysis.cache import CacheIdentity, EmbeddingCache
from arvamusfestivali_transcripts.topic_analysis.corpus import _load_archive
from arvamusfestivali_transcripts.topic_analysis.embedders import ADAPTER_VERSION, BgeM3Embedder
from arvamusfestivali_transcripts.topic_analysis.semantic_segmentation import (
    build_atomic_blocks,
    segment_episode_semantically,
)
from arvamusfestivali_transcripts.topic_analysis.sentence_boundaries import (
    is_sentence_ending,
    snap_sentence_boundaries,
)
from arvamusfestivali_transcripts.topic_analysis.types import SemanticSegmentationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["short3", "cue5", "sentence3", "sentence5"], default="short3"
    )
    parser.add_argument("--episode-ids", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--original-dataset", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis"
    output = (
        base / "results/jev-short-semantic-sample"
        if args.variant == "short3"
        else base / "results/segmentation-experiments" / args.variant
    )
    output = args.output or output
    output.mkdir(parents=True, exist_ok=True)
    config = SemanticSegmentationConfig(
        min_segment_seconds=60,
        max_segment_seconds=300 if args.variant in {"cue5", "sentence5"} else 180,
        boundary_quantile=0.70,
    )
    adapter = BgeM3Embedder()
    identity = CacheIdentity(
        adapter.model_id,
        adapter.model_revision,
        adapter.dimension,
        adapter.instruction,
        config.atomic_chunking,
        ADAPTER_VERSION,
    )
    cache = EmbeddingCache(base / "cache")
    original = pd.read_parquet(
        args.original_dataset or base / "results/jev-semantic-sample/review-segments.parquet"
    )
    original = original[original.segmentation == "semantic"]
    ids = args.episode_ids or ["2397177249", "2400164916", "2271676346"]
    rows, boundaries, sources = [], [], []
    namespace = (
        "semantic-short:bge-boundaries:60-180:q70:v1"
        if args.variant == "short3"
        else f"semantic-experiment:{args.variant}:bge:q70:v1"
    )
    for episode_id in ids:
        source = root / "data/transcripts/2026" / f"{episode_id}.json"
        sources.append(
            {
                "path": str(source.relative_to(root)),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
        episode = _load_archive(source, 2026)
        blocks = build_atomic_blocks(episode, config)
        vectors = cache.assemble(identity, blocks)
        if any(vector is None for vector in vectors):
            raise ValueError(f"Missing matching atomic BGE cache for {episode_id}")
        result = segment_episode_semantically(episode, blocks, np.stack(vectors), config)
        passages = result.passages
        if args.variant.startswith("sentence"):
            passages, cut_diagnostics = snap_sentence_boundaries(
                episode,
                passages,
                maximum_seconds=config.max_segment_seconds,
            )
            boundaries.extend(cut_diagnostics)
        else:
            boundaries.extend(asdict(boundary) for boundary in result.boundaries)
        original_rows = original[original.episode_id.astype(str) == episode_id]
        # Check every original cue is represented exactly once, with unchanged text.
        assert sum(p.cue_count for p in passages) == len(episode.cues)
        assert " ".join(p.text for p in passages) == " ".join(c.text for c in episode.cues)
        assert all(
            p.end_seconds - p.start_seconds
            <= config.max_segment_seconds + (20 if args.variant.startswith("sentence") else 0)
            for p in passages
        )
        for passage in passages:
            overlaps = []
            for old in original_rows.itertuples():
                seconds = max(
                    0,
                    min(old.end_seconds, passage.end_seconds)
                    - max(old.start_seconds, passage.start_seconds),
                )
                if seconds:
                    overlaps.append({"segment_key": old.segment_key, "overlap_seconds": seconds})
            rows.append(
                {
                    "model_key": namespace,
                    "segment_key": f"{namespace}:{passage.passage_id}",
                    "passage_id": passage.passage_id,
                    "episode_id": episode_id,
                    "talk_name": episode.title,
                    "title": episode.title,
                    "duplicate_episode_ids": original_rows.iloc[0].duplicate_episode_ids,
                    "audio_sha256": episode.audio_sha256,
                    "start_seconds": passage.start_seconds,
                    "end_seconds": passage.end_seconds,
                    "duration_seconds": passage.end_seconds - passage.start_seconds,
                    "text": passage.text,
                    "audio_link": passage.timestamped_audio_url,
                    "segmentation": "semantic",
                    "cue_count": passage.cue_count,
                    "word_count": passage.word_count,
                    "ends_at_sentence": is_sentence_ending(passage.text),
                    "variant": args.variant,
                    "original_segment_overlaps": json.dumps(overlaps),
                }
            )
        print(
            f"{episode_id}: {len(original_rows)} original → {len(passages)} {args.variant}",
            flush=True,
        )
    short = pd.DataFrame(rows)
    assert short.segment_key.is_unique
    short.to_parquet(output / "segments.parquet", index=False)
    short.to_csv(output / "segments.csv", index=False)
    pd.DataFrame(boundaries).to_csv(output / "boundaries.csv", index=False)
    manifest = {
        "namespace": namespace,
        "configuration": asdict(config),
        "sentence_snap_window_seconds": 20 if args.variant.startswith("sentence") else 0,
        "sentence_snap_minimum_seconds": 45 if args.variant.startswith("sentence") else None,
        "sentence_method": "ASR punctuation with abbreviation exclusions; "
        "not a full thought detector",
        "original_dataset_sha256": hashlib.sha256(
            (base / "results/segment-dataset/all-segments.parquet").read_bytes()
        ).hexdigest(),
        "sources": sources,
        "atomic_cache_identity": asdict(identity),
        "atomic_cache_digest": identity.digest,
        "episode_ids": ids,
        "short_segment_count": len(short),
        "topic_assignment": "No clustering fit; Jev separately uses the original fitted vocabulary",
        "duration_seconds": short.duration_seconds.describe().to_dict(),
    }
    (output / "segmentation-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["duration_seconds"], indent=2), flush=True)


if __name__ == "__main__":
    main()
