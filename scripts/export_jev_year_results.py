"""Split saved Jev results by verified festival year, without new classification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from export_jev_treemaps import build_treemap

from arvamusfestivali_transcripts.topic_analysis.topic_map import plot_topic_embedding_map

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/topic-analysis/results"
SOURCE = BASE / "jev-semantic-all"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assigned = pd.read_parquet(SOURCE / "assignments.parquet")
    original = pd.read_parquet(BASE / "segment-dataset/all-segments.parquet")
    review = pd.read_parquet(SOURCE / "review-segments.parquet")
    metadata = pd.read_parquet(BASE / "talk-metadata/talks.parquet")
    assert metadata.episode_id.is_unique and metadata.programme_year.notna().all()
    full_manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert sha(BASE / "segment-dataset/all-segments.parquet") == full_manifest["source_sha256"]
    coordinates = pd.read_csv(SOURCE / "bubble-maps/topic-coordinates.csv")
    styles = json.loads((BASE / "high-level-topics/category-style.json").read_text())
    responses = [json.loads(line) for line in (SOURCE / "responses.jsonl").read_text().splitlines()]
    assert len(responses) == len(assigned)
    differences = pd.read_parquet(SOURCE / "high-level-disagreements.parquet")
    seen = set()
    for year in sorted(metadata.programme_year.unique()):
        year = int(year)
        out = SOURCE / str(year)
        out.mkdir(exist_ok=True)
        talks = metadata.loc[metadata.programme_year.eq(year)].copy()
        episodes = set(talks.episode_id)
        native = original.loc[original.episode_id.isin(episodes)].copy()
        keys = set(native.loc[native.segmentation.eq("semantic"), "segment_key"])
        assert not seen & keys
        seen |= keys
        labels = assigned.loc[assigned.segment_key.isin(keys)].copy()
        assert set(labels.segment_key) == keys
        labels.to_parquet(out / "assignments.parquet", index=False)
        labels.to_csv(out / "assignments.csv", index=False)
        native.to_parquet(out / "segment-dataset.parquet", index=False)
        talks.to_parquet(out / "talk-metadata.parquet", index=False)
        talks.to_csv(out / "talk-metadata.csv", index=False)
        enriched = review.loc[review.episode_id.isin(episodes)].copy()
        protected = enriched.copy()
        fields = [c for c in talks if c.startswith("organizer") or c.startswith("programme")]
        enriched = enriched.merge(
            talks[["episode_id", *fields]], on="episode_id", validate="many_to_one", sort=False
        )
        pd.testing.assert_frame_equal(
            enriched[protected.columns].reset_index(drop=True), protected.reset_index(drop=True)
        )
        enriched.to_parquet(out / "review-segments.parquet", index=False)
        differences.loc[differences.episode_id.isin(episodes)].to_parquet(
            out / "high-level-disagreements.parquet", index=False
        )
        checkpoint = [r for r in responses if r["segment_key"] in keys]
        assert len(checkpoint) == len(labels)
        (out / "responses.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in checkpoint)
        )
        semantic = enriched.loc[enriched.segmentation.eq("semantic")].copy()
        for name, columns in [
            ("fine-topic", ["jev_topic_id", "jev_topic_name"]),
            ("high-level", ["jev_high_level_id", "jev_high_level_name"]),
        ]:
            semantic.groupby(columns).agg(
                segments=("segment_key", "size"),
                talks=("episode_id", "nunique"),
                recording_hours=("duration_seconds", lambda x: x.sum() / 3600),
                mean_choice_probability=("jev_probability", "mean"),
            ).reset_index().to_csv(out / f"{name}-summary.csv", index=False)
        parent = pd.read_csv(out / "high-level-summary.csv")
        px.bar(
            parent,
            x="jev_high_level_name",
            y="recording_hours",
            hover_data=["segments", "talks"],
            title=f"{year}: Jev high-level coverage",
        ).write_html(out / "high-level-coverage.html", include_plotlyjs=True)
        noise = semantic.jev_topic_id.eq(-1)
        stats = dict(
            programme_year=year,
            semantic_segments=len(semantic),
            talks=len(talks),
            jev_unassigned_segments=int(noise.sum()),
            jev_unassigned_time_pct=float(
                semantic.loc[noise].duration_seconds.sum() / semantic.duration_seconds.sum() * 100
            ),
            fine_label_agreement_pct=float(
                semantic.topic_id.eq(semantic.jev_topic_id).mean() * 100
            ),
            parent_label_agreement_pct=float(
                semantic.model_high_level_id.eq(semantic.jev_high_level_id).mean() * 100
            ),
            organizer_known_talks=int(talks.organizer.ne("").sum()),
            interpretation="Agreement is descriptive, not accuracy; original shared "
            "236-topic vocabulary, no year-specific refit.",
        )
        (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
        provenance = {
            k: full_manifest[k]
            for k in [
                "model",
                "fitted_model_key",
                "segment_model_key",
                "config_sha256",
                "names_sha256",
                "assignment_type",
                "high_level_mapping_sha256",
                "high_level_assignment_type",
            ]
        }
        provenance.update(
            programme_year=year,
            segment_count=len(labels),
            episode_ids=sorted(episodes),
            source_sha256=sha(out / "segment-dataset.parquet"),
            full_source_sha256=full_manifest["source_sha256"],
            parent_assignments_sha256=sha(SOURCE / "assignments.parquet"),
            assignments_sha256=sha(out / "assignments.parquet"),
            talk_metadata_sha256=sha(out / "talk-metadata.parquet"),
            export_type="derived_year_subset",
            api_requests_this_export=0,
            classification="Existing saved responses, unchanged native boundaries and labels",
            bubble_coordinates=("Combined-corpus projection; year-specific counts"),
        )
        (out / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for chart in ["treemaps", "bubble-maps"]:
            chart_out = out / chart
            chart_out.mkdir(exist_ok=True)
            if chart == "bubble-maps":
                table = (
                    semantic.loc[~noise]
                    .groupby("jev_topic_id")
                    .agg(
                        segment_count=("segment_key", "size"), talk_count=("episode_id", "nunique")
                    )
                    .reset_index()
                    .rename(columns={"jev_topic_id": "topic_id"})
                )
                table = coordinates.drop(columns=["segment_count", "talk_count"]).merge(
                    table, on="topic_id", validate="one_to_one"
                )
                table.to_csv(chart_out / "topic-coordinates.csv", index=False)
            for metric, column in [("segments", "segment_count"), ("talks", "talk_count")]:
                if chart == "treemaps":
                    figure, counts = build_treemap(semantic, metric)
                    counts.to_csv(chart_out / f"{metric}-counts.csv", index=False)
                    figure.update_layout(title=f"{year}: {figure.layout.title.text}")
                else:
                    figure = go.Figure()
                    for category, group in table.groupby("high_level_id"):
                        trace = plot_topic_embedding_map(
                            group, title="", color=styles[category]["color"]
                        ).data[0]
                        trace.marker.size = group[column].to_numpy()
                        # Same size scale as the combined view allows year-to-year comparisons.
                        trace.marker.sizeref = 2 * int(coordinates[column].max()) / 52**2
                        trace.marker.symbol = "circle"
                        trace.marker.opacity = 1
                        trace.name = group.high_level_name.iloc[0]
                        trace.showlegend = True
                        figure.add_trace(trace)
                    figure.update_layout(
                        title=f"{year}: Jev topics · bubble area = {metric}",
                        template="plotly_white",
                        height=850,
                        xaxis=dict(range=[coordinates.x.min() - 0.5, coordinates.x.max() + 0.5]),
                        yaxis=dict(range=[coordinates.y.min() - 0.5, coordinates.y.max() + 0.5]),
                    )
                    figure.add_annotation(
                        text="Combined-corpus positions and size scale; "
                        "counts restricted to this year. Unassigned excluded.",
                        x=0,
                        y=-0.1,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                    )
                figure.write_html(chart_out / f"{metric}.html", include_plotlyjs=True)
                (chart_out / f"{metric}.plotly.json").write_text(figure.to_json())
            (chart_out / "index.html").write_text(
                f'<!doctype html><meta charset="utf-8"><h1>{year}: {chart}</h1>'
                '<p><a href="segments.html">Segment counts</a> · '
                '<a href="talks.html">Distinct talk counts</a></p>'
            )
            (chart_out / "manifest.json").write_text(
                json.dumps(
                    dict(
                        programme_year=year,
                        source_sha256=sha(out / "review-segments.parquet"),
                        counts_scope="Festival year only",
                        segments=len(semantic),
                        talks=len(talks),
                        coordinate_scope="Combined-corpus projection reused"
                        if chart == "bubble-maps"
                        else None,
                        talk_area_rule=("Area sums topic memberships; parent labels count unions"),
                    ),
                    indent=2,
                )
                + "\n"
            )
        (out / "index.html").write_text(
            f'<!doctype html><meta charset="utf-8"><h1>{year}: detailed Jev classification</h1>'
            '<p><a href="../index.html">Combined results</a> · '
            '<a href="high-level-coverage.html">Category coverage</a> · '
            '<a href="treemaps/index.html">Treemaps</a> · '
            '<a href="bubble-maps/index.html">Bubble maps</a></p>'
            + pd.DataFrame([stats]).to_html(index=False)
            + parent.to_html(index=False)
        )
        (out / "README.md").write_text(
            f"# {year}: saved Jev classification\n\n"
            f"{len(talks)} talks, {len(labels)} native semantic segments. "
            "Detailed labels and probability distributions are copied unchanged from the combined "
            "classification; high-level categories remain dictionary rollups. No API calls or "
            "topic refitting. Year comes from verified programme metadata.\n\n"
            "assignments.parquet / .csv and responses.jsonl contain only this year's predictions. "
            "segment-dataset.parquet retains both original segmentations for this year's talks. "
            "review-segments.parquet adds saved Jev results and organizer/programme metadata; "
            "fixed rows have no Jev prediction. "
            "talk-metadata.parquet / .csv contains talk metadata. "
            "Fine/high-level summaries, disagreements and charts are year-specific.\n\n"
            "Bubble positions and size scales reuse the combined projection for comparison. "
            "Counts are year-specific; absent topics are omitted. Treemap talk areas sum "
            "topic–talk memberships, while parent labels report distinct-talk unions.\n"
        )
        print(year, len(talks), "talks", len(labels), "semantic labels", flush=True)
    assert seen == set(assigned.segment_key)
    index = SOURCE / "index.html"
    page = index.read_text()
    if 'id="year-results"' not in page:
        page += '<p id="year-results">Festival year: <a href="2025/index.html">2025</a> · '
        page += '<a href="2026/index.html">2026</a></p>'
        index.write_text(page)


if __name__ == "__main__":
    main()
