"""Export comparable time windows, timelines and metrics for the broader pilot."""

import html
from pathlib import Path

import pandas as pd
import plotly.express as px

from arvamusfestivali_transcripts.topic_analysis.experiment_dynamics import (
    interval_topic_overlap,
    plot_variant_topic_timelines,
)
from arvamusfestivali_transcripts.topic_analysis.segment_review import (
    document_catalog,
    segmentation_experiment_review_html,
)
from arvamusfestivali_transcripts.topic_analysis.sentence_boundaries import is_sentence_ending


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/topic-analysis/results/segmentation-experiments/broader-pilot"
    models = output / "topic-models"
    original = pd.read_parquet(output / "original.parquet")
    original["episode_id"] = original.episode_id.astype(str)
    catalog = document_catalog(original)
    summary = pd.read_csv(models / "comparison-summary.csv")
    stability = pd.read_csv(models / "seed-stability.csv")
    px.bar(
        summary,
        x="variant",
        y="topic_count",
        color=summary.seed.astype(str),
        barmode="group",
        hover_data=["outlier_time_pct", "mean_changes_per_hour", "original_cosine_silhouette"],
    ).write_html(output / "topic-counts.html", include_plotlyjs="directory")
    links, reviews, overlaps = [], [], []
    for seed in (42, 7, 19):
        fits = {
            v: pd.read_parquet(models / v / f"seed-{seed}/review-segments.parquet")
            for v in ("original", "sentence5")
        }
        overlap = interval_topic_overlap(fits["original"], fits["sentence5"])
        overlap["seed"] = seed
        overlaps.append(overlap)
        for index, talk in catalog.iterrows():
            episode_id = str(talk.episode_id)
            filename = f"timeline-{episode_id}-seed-{seed}.html"
            plot_variant_topic_timelines(
                fits, episode_id, title=f"{talk.talk_name} · seed {seed}"
            ).write_html(output / filename, include_plotlyjs="directory")
            links.append(
                f'<li><a href="{filename}">{html.escape(talk.talk_name)} · seed {seed}</a></li>'
            )
            # Three shared, distributed eight-minute windows for each talk.
            duration = original[original.episode_id == episode_id].end_seconds.max()
            for position, fraction in [("early", 0.1), ("middle", 0.45), ("late", 0.8)]:
                start = float(max(0, min(duration - 480, duration * fraction)))
                filename = f"review-{episode_id}-{position}-seed-{seed}.html"
                rendered = segmentation_experiment_review_html(
                    original,
                    fits,
                    index,
                    left="original",
                    right="sentence5",
                    start_seconds=start,
                    end_seconds=start + 480,
                    assignment_source="topic_model",
                )
                (output / filename).write_text('<!doctype html><meta charset="utf-8">' + rendered)
                links.append(
                    f'<li><a href="{filename}">Compare {html.escape(talk.talk_name)} · '
                    f"{position} · seed {seed}</a></li>"
                )
                if seed == 42:
                    reviews.append(
                        {
                            "episode_id": episode_id,
                            "talk_name": talk.talk_name,
                            "position": position,
                            "start_seconds": start,
                            "end_seconds": start + 480,
                            "baseline_coherence_1_to_5": None,
                            "sentence5_coherence_1_to_5": None,
                            "baseline_boundary_quality_1_to_5": None,
                            "sentence5_boundary_quality_1_to_5": None,
                            "meaningful_topic_changes": None,
                            "preferred_variant": None,
                            "notes": None,
                            "review_html": filename,
                        }
                    )
    pd.concat(overlaps).to_csv(output / "topic-overlap-seconds.csv", index=False)
    review_path = output / "manual-review.csv"
    if not review_path.exists():
        pd.DataFrame(reviews).to_csv(review_path, index=False)
    diagnostics = []
    for variant in ("original", "sentence5"):
        for seed in (42, 7, 19):
            fitted = pd.read_parquet(models / variant / f"seed-{seed}/review-segments.parquet")
            for episode_id, group in fitted.groupby("episode_id"):
                group = group.sort_values("start_seconds")
                labels = group.cluster_topic_id.to_numpy()
                changes = labels[1:] != labels[:-1]
                assigned_pairs = (labels[1:] != -1) & (labels[:-1] != -1)
                diagnostics.append(
                    {
                        "episode_id": episode_id,
                        "variant": variant,
                        "seed": seed,
                        "boundaries": len(changes),
                        "assigned_pair_boundaries": int(assigned_pairs.sum()),
                        "assigned_topic_changes": int((changes & assigned_pairs).sum()),
                        "noise_transitions": int((changes & ~assigned_pairs).sum()),
                        "changes_per_100_boundaries": float(changes.mean() * 100)
                        if len(changes)
                        else 0,
                        "assigned_changes_per_100_assigned_boundaries": float(
                            changes[assigned_pairs].mean() * 100
                        )
                        if assigned_pairs.any()
                        else None,
                    }
                )
    pd.DataFrame(diagnostics).to_csv(output / "transition-diagnostics.csv", index=False)
    dynamics = []
    for v in ("original", "sentence5"):
        for seed in (42, 7, 19):
            frame = pd.read_csv(models / v / f"seed-{seed}/talk-dynamics.csv")
            frame["variant"], frame["seed"] = v, seed
            dynamics.append(frame)
    pd.concat(dynamics).to_csv(output / "talk-dynamics.csv", index=False)
    segments = pd.read_parquet(output / "sentence5/segments.parquet")
    segmentation = pd.DataFrame(
        [
            {
                "variant": v,
                "segments": len(data),
                "median_seconds": data.duration_seconds.median(),
                "max_seconds": data.duration_seconds.max(),
                "sentence_ending_internal_cut_pct": float(
                    pd.concat(
                        [
                            group.sort_values("start_seconds")
                            .iloc[:-1]
                            .text.map(is_sentence_ending)
                            for _, group in data.groupby("episode_id")
                        ]
                    ).mean()
                    * 100
                ),
            }
            for v, data in [("original", original), ("sentence5", segments)]
        ]
    )
    segmentation.to_csv(output / "segmentation-summary.csv", index=False)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:15px system-'
        "ui;margin:25px}td,th{padding:7px}table{border-"
        "collapse:collapse}</style><h1>18-talk segmentation pilot</h1><p>Independent "
        "fits: IDs and colors are local to each model. Keyword labels are derived from "
        "each new fit. Switching includes noise transitions; overlap is temporal "
        "correspondence, not semantic agreement. Manual scores are blank until "
        'reviewed.</p><p><a href="topic-counts.html">Topic count chart</a> · <a '
        'href="manual-review.csv">54 matched windows to score</a> · <a '
        'href="qualitative-examples.csv">Six Codex qualitative readings</a></p>'
        + summary.to_html(index=False)
        + stability.to_html(index=False)
        + "<ul>"
        + "".join(links)
        + "</ul>"
    )
    (output / "README.md").write_text(
        "# Broader segmentation pilot\n\nPurposeful 18-talk sample; original full corpus "
        "remains unchanged. Compare original native segments with sentence-aware semantic"
        " cuts (q70, minimum60s, nominal maximum300s, sentence snap ±20s, minimum "
        "snapped45s). Three seeds:42,7,19. Same BGE-M3 encoder and leaf-local-6-2 "
        "BERTopic settings. Topic names are fresh c-TF-IDF keywords, not transferred "
        "original names.\n\nSee index.html for metrics, native timelines and 54 shared "
        "eight-minute windows per seed. Charts load the bundled local plotly.min.js; keep "
        "it beside the HTML files for offline review. manual-review.csv holds blank "
        "human ratings; no "
        "quality scores have been fabricated. Topic switches include -1 and increase "
        "mechanically with more segments. Seed ARI uses jointly assigned segments; "
        "inspect shared-inlier fraction too. Sample topic counts must not be compared "
        "directly to the full-corpus236.\n\n"
        "This compares a package of changes: shorter maximum, lower semantic boundary "
        "threshold and sentence snapping. It does not isolate snapping alone. The "
        "English-language candidate2314163822 was excluded for sparse cue timing "
        "including an indivisible377.96-second cue; see inputs.json.\n\n"
        + ("```csv\n" + summary.to_csv(index=False) + "```\n")
        + "\n\n"
        + ("```csv\n" + stability.to_csv(index=False) + "```\n")
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
