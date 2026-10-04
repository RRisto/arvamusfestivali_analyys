"""Validate full-corpus detailed Jev labels, map parents and export review summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis/results"
    output = base / "jev-semantic-all"
    source = base / "segment-dataset/all-segments.parquet"
    data = pd.read_parquet(source)
    semantic = data[data.segmentation == "semantic"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    assigned = pd.read_parquet(output / "assignments.parquet")
    assert assigned.jev_model.eq("jev-1.13.0").all()
    assert assigned.jev_confidence.between(0, 1).all()
    assert assigned.segment_key.is_unique
    assert set(assigned.segment_key) == set(semantic.segment_key)
    mapping_root = base / "high-level-topics"
    mapping = json.loads((mapping_root / "topic-key-to-high-level.json").read_text())
    fine_mapping = json.loads((mapping_root / "topic-to-high-level.json").read_text())
    categories = pd.read_csv(mapping_root / "categories.csv")
    names = dict(zip(categories.high_level_id, categories.name_et))
    names["unassigned"] = "Määramata"
    assigned["jev_high_level_id"] = assigned.jev_topic_key.map(mapping)
    assigned["jev_high_level_name"] = assigned.jev_high_level_id.map(names)
    assert assigned.jev_high_level_id.notna().all()
    parent_distributions = []
    for row in assigned.itertuples():
        probabilities = json.loads(row.jev_probabilities)
        assert set(probabilities) == set(fine_mapping)
        assert all(0 <= float(p) <= 1 for p in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) < 0.02
        assert np.isclose(row.jev_probability, probabilities[str(row.jev_topic_id)])
        assert np.isclose(row.jev_probability, max(probabilities.values()))
        parents = {}
        for fine, probability in probabilities.items():
            parent = fine_mapping[fine]
            parents[parent] = parents.get(parent, 0) + probability
        parent_distributions.append(parents)
    assigned["jev_high_level_probabilities"] = [json.dumps(p) for p in parent_distributions]
    assigned["jev_high_level_probability_mass"] = [
        p[c] for p, c in zip(parent_distributions, assigned.jev_high_level_id)
    ]
    assigned.to_parquet(output / "assignments.parquet", index=False)
    assigned.to_csv(output / "assignments.csv", index=False)
    review = data.merge(assigned, on="segment_key", how="left", validate="one_to_one")
    review["model_high_level_id"] = review.topic_key.map(mapping)
    pd.testing.assert_frame_equal(review[data.columns], data)
    review.to_parquet(output / "review-segments.parquet", index=False)
    selected = review[review.segmentation == "semantic"].copy()
    selected["fine_topic_agrees"] = selected.topic_id == selected.jev_topic_id
    selected["high_level_agrees"] = selected.model_high_level_id == selected.jev_high_level_id
    selected.groupby(["jev_topic_id", "jev_topic_name"]).agg(
        segments=("segment_key", "size"),
        talks=("episode_id", "nunique"),
        recording_hours=("duration_seconds", lambda x: x.sum() / 3600),
        mean_choice_probability=("jev_probability", "mean"),
    ).reset_index().to_csv(output / "fine-topic-summary.csv", index=False)
    parent_summary = (
        selected.groupby(["jev_high_level_id", "jev_high_level_name"])
        .agg(
            segments=("segment_key", "size"),
            talks=("episode_id", "nunique"),
            recording_hours=("duration_seconds", lambda x: x.sum() / 3600),
        )
        .reset_index()
    )
    parent_summary.to_csv(output / "high-level-summary.csv", index=False)
    px.bar(
        parent_summary,
        x="jev_high_level_name",
        y="recording_hours",
        hover_data=["segments", "talks"],
    ).write_html(output / "high-level-coverage.html", include_plotlyjs=True)
    disagreements = selected[~selected.high_level_agrees]
    disagreements[
        [
            "segment_key",
            "episode_id",
            "talk_name",
            "start_seconds",
            "end_seconds",
            "topic_id",
            "topic_name",
            "model_high_level_id",
            "jev_topic_id",
            "jev_topic_name",
            "jev_high_level_id",
            "jev_probability",
            "text",
            "audio_link",
        ]
    ].to_parquet(output / "high-level-disagreements.parquet", index=False)
    stats = {
        "semantic_segments": len(selected),
        "talks": selected.episode_id.nunique(),
        "jev_unassigned_segments": int((selected.jev_topic_id == -1).sum()),
        "jev_unassigned_time_pct": float(
            selected.loc[selected.jev_topic_id == -1, "duration_seconds"].sum()
            / selected.duration_seconds.sum()
            * 100
        ),
        "fine_label_agreement_pct": float(selected.fine_topic_agrees.mean() * 100),
        "parent_label_agreement_pct": float(selected.high_level_agrees.mean() * 100),
        "interpretation": "Agreement with the original clustering is descriptive, not accuracy. "
        "Jev uses detailed 236-topic choice; parents are dictionary rollups, "
        "not direct high-level predictions.",
    }
    (output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    manifest["high_level_mapping_sha256"] = hashlib.sha256(
        (mapping_root / "topic-key-to-high-level.json").read_bytes()
    ).hexdigest()
    manifest["high_level_assignment_type"] = "dictionary_parent_of_detailed_jev_choice"
    manifest["validation"] = (
        "All native semantic keys covered once; input columns unchanged; "
        "probability vocabularies and values checked."
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Jev labels for all current '
        "semantic segments</h1><p>Detailed topics are direct Jev choices. High-level"
        " categories come from the saved dictionary. Original model topics and "
        'boundaries are preserved. Agreement is not accuracy.</p><p><a href="high-'
        'level-coverage.html">High-level coverage chart</a></p>'
        + pd.DataFrame([stats]).to_html(index=False)
        + parent_summary.to_html(index=False)
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
