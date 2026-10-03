"""Export the two recommended native segment sets with named topics and overlap links."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from arvamusfestivali_transcripts.topic_analysis.segment_assignments import (
    link_overlapping_segments,
    name_segment_topics,
)


def export_dataset(input_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    datasets, models = {}, {}
    for mode in ("semantic", "fixed"):
        source = input_root / f"{mode}-bge" / "leaf-local-6-2" / "seed-42"
        assignments = pd.read_csv(
            source / "passage-assignments.csv",
            dtype={"passage_id": str, "episode_id": str, "audio_sha256": str},
        )
        topics = pd.read_csv(source / "topics.csv")
        key = f"{mode}:bge:leaf-local-6-2:42"
        named = name_segment_topics(assignments, topics, model_key=key)
        named["segmentation"] = mode
        named["candidate"] = "leaf-local-6-2"
        named["seed"] = 42
        named["segment_key"] = key + ":" + named.passage_id
        datasets[mode] = named
        models[mode] = json.loads((source / "manifest.json").read_text())
    semantic = link_overlapping_segments(datasets["semantic"], datasets["fixed"])
    fixed = link_overlapping_segments(datasets["fixed"], datasets["semantic"])
    combined = pd.concat([semantic, fixed], ignore_index=True)
    assert combined.segment_key.is_unique
    assert len(combined) == len(semantic) + len(fixed)

    files = {}
    for name, frame in {
        "semantic-segments": semantic,
        "fixed-segments": fixed,
        "all-segments": combined,
    }.items():
        csv_path = output_root / f"{name}.csv.gz"
        frame.to_csv(csv_path, index=False, compression={"method": "gzip", "mtime": 0})
        records = json.loads(frame.to_json(orient="records", force_ascii=False))
        jsonl_path = output_root / f"{name}.jsonl.gz"
        with gzip.open(jsonl_path, "wt", encoding="utf-8") as stream:
            for record in records:
                record["other_model_overlaps"] = json.loads(record["other_model_overlaps"])
                record["duplicate_episode_ids"] = json.loads(record["duplicate_episode_ids"])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        parquet = output_root / f"{name}.parquet"
        if importlib.util.find_spec("pyarrow") is not None:
            frame.to_parquet(parquet, index=False, engine="pyarrow", compression="zstd")
        for path in (csv_path, jsonl_path, parquet):
            if path.exists():
                files[path.name] = {
                    "rows": len(frame),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    # Small, portable tables can be committed without repeating full transcript text.
    compact = combined.drop(columns=["text", "other_model_overlaps"])
    compact.to_csv(output_root / "segment-topic-index.csv", index=False)
    pd.concat([semantic.head(3), fixed.head(2)], ignore_index=True).to_csv(
        output_root / "preview.csv", index=False
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "semantic_segments": len(semantic),
        "fixed_segments": len(fixed),
        "combined_segments": len(combined),
        "talk_count": int(combined.episode_id.nunique()),
        "confidence_semantics": (
            "HDBSCAN assigned-cluster membership strength, not calibrated semantic certainty. "
            "Null for topic -1."
        ),
        "topic_name_semantics": (
            "Automatic BERTopic keyword label, with separators reformatted; "
            "not a manually reviewed title."
        ),
        "topic_id_scope": "IDs belong to a model_key. Use topic_key across models.",
        "segment_id_scope": (
            "Native passage IDs can recur across models. Use segment_key across models."
        ),
        "timestamps": "Seconds from recording start; intervals are [start_seconds, end_seconds).",
        "overlap_semantics": (
            "Other-model labels belong to overlapping native segments, not new predictions "
            "on this segment. Fractions can sum above 1 for overlapping fixed windows."
        ),
        "columns": combined.columns.tolist(),
        "models": models,
        "files": files,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (output_root / "README.md").write_text(
        "# Reusable segment dataset\n\n"
        "Start with `all-segments.parquet` or `all-segments.csv.gz`. "
        "Separate `semantic-segments` and `fixed-segments` datasets are also included.\n\n"
        "Every row preserves original text, recording title, timestamps, audio link, "
        "episode ID, duplicate recording IDs and audio hash. Added fields include "
        "`topic_id`, `topic_name`, `confidence`, `model_key`, `topic_key`, and `segment_key`.\n\n"
        "`other_model_overlaps` preserves the other segmentation's native topic assignments "
        "and their exact time overlaps. JSONL stores these as arrays; CSV and Parquet "
        "store JSON strings. Read the manifest for confidence and coverage semantics.\n\n"
        "```python\nimport pandas as pd\nsegments = pd.read_parquet('all-segments.parquet')\n"
        "# Or: pd.read_csv('all-segments.csv.gz', dtype={'episode_id': str})\n"
        "talk = segments[segments.episode_id == '2247188684']\n```\n\n"
        "`segment-topic-index.csv` is a compact version without text or overlap arrays. "
        "Full files live in the analysis workspace under "
        "`data/topic-analysis/results/segment-dataset/`. "
        "Install `pyarrow` to regenerate Parquet output.\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = export_dataset(args.input_root, args.output_root)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "semantic_segments",
                    "fixed_segments",
                    "combined_segments",
                    "talk_count",
                ]
            }
        )
    )
