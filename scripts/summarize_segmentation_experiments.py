"""Export offline comparison summaries and review pages for the three-talk experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from arvamusfestivali_transcripts.topic_analysis.segment_review import (
    segmentation_experiment_review_html,
)
from arvamusfestivali_transcripts.topic_analysis.sentence_boundaries import is_sentence_ending


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis/results"
    output = base / "segmentation-experiments"
    original = pd.read_parquet(base / "jev-semantic-sample/review-segments.parquet")
    short = pd.read_parquet(base / "jev-short-semantic-sample/review-segments.parquet")
    ids = set(short.episode_id.astype(str))
    variants = {
        "original": original[
            (original.segmentation == "semantic") & original.episode_id.astype(str).isin(ids)
        ],
        "short3": short,
    }
    for key in ["cue5", "sentence3", "sentence5"]:
        variants[key] = pd.read_parquet(output / key / "review-segments.parquet")
    summaries, annotations = [], []
    for key, data in variants.items():
        duration = data.end_seconds - data.start_seconds
        internal = data.drop(index=data.groupby("episode_id").tail(1).index)
        sentence_rate = internal.text.map(is_sentence_ending).mean()
        summaries.append(
            {
                "variant": key,
                "segments": len(data),
                "median_seconds": round(duration.median(), 1),
                "max_seconds": round(duration.max(), 1),
                "sentence_ending_cut_pct": round(sentence_rate * 100, 1),
                "jev_unassigned_segments": int((data.jev_topic_id == -1).sum()),
                "jev_unassigned_time_pct": round(
                    duration[data.jev_topic_id == -1].sum() / duration.sum() * 100, 1
                ),
                "median_jev_choice_probability": round(data.jev_probability.median(), 2),
            }
        )
        for row in data.itertuples():
            annotations.append(
                {
                    "variant": key,
                    "episode_id": str(row.episode_id),
                    "segment_key": row.segment_key,
                    "start_seconds": row.start_seconds,
                    "end_seconds": row.end_seconds,
                    "jev_topic_name": row.jev_topic_name,
                    "boundary_quality": "",
                    "topic_fit": "",
                    "lost_context": "",
                    "notes": "",
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "comparison-summary.csv", index=False)
    review_path = output / "manual-review.csv"
    if not review_path.exists():
        pd.DataFrame(annotations).to_csv(review_path, index=False)
    pages = []
    for index in [0, 1, 3]:
        for left, right in [
            ("short3", "sentence3"),
            ("cue5", "sentence5"),
            ("sentence3", "sentence5"),
        ]:
            filename = f"review-document-{index}-{left}-vs-{right}.html"
            html = segmentation_experiment_review_html(
                original,
                variants,
                index,
                left=left,
                right=right,
            )
            (output / filename).write_text('<!doctype html><meta charset="utf-8">' + html)
            pages.append(f'<li><a href="{filename}">Document {index}: {left} vs {right}</a></li>')
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Segmentation experiments</title>'
        "<style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse}"
        "td,th{padding:8px;border:1px solid #ddd}</style>"
        "<h1>Three-talk segmentation experiments</h1>"
        "<p>Document 0: teacher support; 1: AI entrepreneurship; 3: military service.</p>"
        "<p>Sentence-ending percentages measure punctuation only. Jev probabilities "
        "and abstention are descriptive, not accuracy.</p>"
        + summary.to_html(index=False)
        + "<h2>Side-by-side full-text comparisons</h2><ul>"
        + "".join(pages)
        + "</ul>"
    )
    manifest = {
        "variants": list(variants),
        "sample_episode_ids": sorted(ids),
        "original_vocabulary": "semantic:bge:leaf-local-6-2:42",
        "metrics_are_accuracy": False,
        "sentence_method": "Punctuation plus abbreviation exclusions; ±20s cue-aligned snap",
    }
    (output / "comparison-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
