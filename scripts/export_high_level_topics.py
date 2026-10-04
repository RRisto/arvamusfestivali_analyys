"""Export an editorial parent taxonomy for the existing 236 semantic topics."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pandas as pd

MODEL = "semantic:bge:leaf-local-6-2:42"


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis/results"
    output = base / "high-level-topics"
    definitions_path = output / "category-definitions.json"
    categories = json.loads(definitions_path.read_text())
    criteria = {
        c["id"]: {"name": c["name_et"], "description": c["description_et"]} for c in categories
    }
    criteria["unassigned"] = {
        "name": "Määramata",
        "description": "Ükski sisuline kategooria ei sobi: "
        "korralduslik, ebaselge või teemaväline tekst.",
    }
    (output / "jev-high-level-criteria.json").write_text(
        json.dumps(criteria, ensure_ascii=False, indent=2) + "\n"
    )
    names_path = base / "topic-names/topic-names.csv"
    names = pd.read_csv(names_path)
    names = names[names.model_key == MODEL].copy()
    source = base / "segment-dataset/all-segments.parquet"
    data = pd.read_parquet(source)
    semantic = data[(data.segmentation == "semantic") & (data.model_key == MODEL)].copy()
    category_lookup = {c["id"]: c for c in categories}
    assert 15 <= len(categories) <= 20 and len(category_lookup) == len(categories)
    pairs = [(str(t), c["id"]) for c in categories for t in c["topic_ids"]]
    mapping = dict(pairs)
    assert len(pairs) == len(mapping), "A fine topic belongs to more than one primary category"
    assert set(mapping) == set(names.topic_id.astype(str))
    assert set(mapping) == set(semantic.loc[semantic.topic_id >= 0, "topic_id"].astype(str))
    mapping["-1"] = "unassigned"
    (output / "topic-to-high-level.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    )
    scoped = {f"{MODEL}:{topic}": parent for topic, parent in mapping.items()}
    (output / "topic-key-to-high-level.json").write_text(
        json.dumps(scoped, ensure_ascii=False, indent=2) + "\n"
    )
    rows = names.sort_values("topic_id", key=lambda x: x.astype(int)).copy()
    rows["high_level_id"] = rows.topic_id.astype(str).map(mapping)
    rows["high_level_name_et"] = rows.high_level_id.map({c["id"]: c["name_et"] for c in categories})
    rows[
        [
            "topic_key",
            "topic_id",
            "topic_name_et",
            "summary_et",
            "segment_count",
            "talk_count",
            "high_level_id",
            "high_level_name_et",
        ]
    ].to_csv(output / "topic-mapping.csv", index=False)
    semantic["high_level_topic_id"] = semantic.topic_id.astype(str).map(mapping)
    semantic["high_level_topic_name"] = semantic.high_level_topic_id.map(
        {**{c["id"]: c["name_et"] for c in categories}, "unassigned": "Määramata"}
    )
    assert semantic.high_level_topic_id.notna().all()
    semantic.to_parquet(output / "semantic-segments.parquet", index=False)
    pd.testing.assert_frame_equal(semantic[data.columns], data.loc[semantic.index])
    summary = []
    parts = []
    for category in categories:
        members = rows[rows.high_level_id == category["id"]]
        segments = semantic[semantic.high_level_topic_id == category["id"]]
        summary.append(
            {
                "high_level_id": category["id"],
                "name_et": category["name_et"],
                "description_et": category["description_et"],
                "fine_topics": len(members),
                "segments": len(segments),
                "talks": segments.episode_id.nunique(),
                "recording_hours": segments.duration_seconds.sum() / 3600,
            }
        )
        table = members[["topic_id", "topic_name_et", "summary_et", "segment_count"]].to_html(
            index=False, escape=True
        )
        parts.append(
            f"<details><summary>{escape(category['name_et'])} · {len(members)} fine topics · "
            f"{len(segments)} segments</summary><p>{escape(category['description_et'])}</p>"
            f"{table}</details>"
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "categories.csv", index=False)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>High-level topic '
        "taxonomy</title><style>body{font:15px system-ui;margin:30px;max-"
        "width:1500px}td,th{padding:7px;border:1px solid #ddd}table{border-"
        "collapse:collapse}details{margin:18px 0}summary{cursor:pointer;font-"
        "weight:bold}</style><h1>20 high-level categories for 236 semantic "
        "topics</h1><p>Editorial grouping based on the saved topic names and "
        "summaries. Fine memberships and transcript boundaries are unchanged. "
        "Categories have one primary parent per fine topic. Cross-domain topics are "
        "assigned by their main subject. This is a reviewable taxonomy, not a new "
        "clustering fit.</p><p>827 unassigned model segments stay unassigned; no "
        "catch-all substantive category is invented.</p>"
        + summary.to_html(index=False, escape=True)
        + "".join(parts)
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "fitted_model_key": MODEL,
        "fine_topic_count": len(rows),
        "high_level_category_count": len(categories),
        "native_semantic_segments": len(semantic),
        "unassigned_segments": int((semantic.topic_id == -1).sum()),
        "source_dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_names_sha256": hashlib.sha256(names_path.read_bytes()).hexdigest(),
        "definitions_sha256": hashlib.sha256(definitions_path.read_bytes()).hexdigest(),
        "method": "Codex editorial primary-parent grouping based on saved evidence-grounded "
        "names and summaries; not a statistical model merge",
        "mapping_scope": "Only the specified existing fitted semantic vocabulary; "
        "do not apply numeric IDs to new fits",
        "jev_rollup": "Map each Jev primary fine-topic choice to its parent. Summing fine-topic "
        "probabilities gives parent probability mass; parent argmax may differ from "
        "the parent of the primary fine topic.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(summary[["high_level_id", "fine_topics", "segments"]].to_string(index=False))
    print(
        f"Validated {len(rows)} fine topics, {len(categories)} categories "
        f"and {len(semantic)} native segments."
    )


if __name__ == "__main__":
    main()
