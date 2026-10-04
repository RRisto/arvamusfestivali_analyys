"""Compare saved Jev topic shares between festival years, without new predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/topic-analysis/results"
OUT = BASE / "jev-topic-analysis/year-comparison"
MIN_MINUTES = 10
MIN_TALKS = 2


def compare(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    counts = data.groupby(["programme_year", *keys]).agg(
        minutes=("duration_seconds", lambda v: v.sum() / 60),
        segments=("segment_key", "size"),
        talks=("episode_id", "nunique"),
    )
    result = counts.unstack("programme_year").fillna(0)
    result.columns = [f"{metric}_{int(year)}" for metric, year in result.columns]
    result = result.reset_index()
    for year in [2025, 2026]:
        sample = data.loc[data.programme_year.eq(year)]
        result[f"time_share_{year}"] = (
            result[f"minutes_{year}"] / sample.duration_seconds.sum() * 6000
        )
        result[f"segment_share_{year}"] = result[f"segments_{year}"] / len(sample) * 100
        result[f"talk_coverage_{year}"] = (
            result[f"talks_{year}"] / sample.episode_id.nunique() * 100
        )
        assigned_total = sample.loc[sample.jev_topic_id.ne(-1)].duration_seconds.sum() / 60
        result[f"assigned_time_share_{year}"] = result[f"minutes_{year}"] / assigned_total * 100
    for metric in ["time_share", "segment_share", "talk_coverage", "assigned_time_share"]:
        result[f"{metric}_delta_pp"] = result[f"{metric}_2026"] - result[f"{metric}_2025"]
    for year in [2025, 2026]:
        is_unassigned = (
            result["jev_topic_id"].eq(-1)
            if "jev_topic_id" in result
            else result["jev_high_level_id"].eq("unassigned")
        )
        result.loc[is_unassigned, f"assigned_time_share_{year}"] = np.nan
    result["assigned_time_share_delta_pp"] = (
        result.assigned_time_share_2026 - result.assigned_time_share_2025
    )
    result["status"] = np.select(
        [result.segments_2025.eq(0), result.segments_2026.eq(0)],
        ["only_observed_2026", "only_observed_2025"],
        default="both_years",
    )
    result["supported_2026"] = result.minutes_2026.ge(MIN_MINUTES) & result.talks_2026.ge(MIN_TALKS)
    result["supported_2025"] = result.minutes_2025.ge(MIN_MINUTES) & result.talks_2025.ge(MIN_TALKS)
    return result


def change_plot(
    table: pd.DataFrame, name: str, metric: str, title: str, path: Path, top: int | None = None
) -> None:
    t = table.copy()
    if top:
        t = t.loc[t[metric].abs().nlargest(top).index]
    t = t.sort_values(metric)
    fig, ax = plt.subplots(figsize=(15, max(5, len(t) * 0.42)))
    ax.barh(t[name], t[metric], color="#ADD8E6")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Change in percentage points: 2026 minus 2025")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    px.bar(
        t,
        x=metric,
        y=name,
        orientation="h",
        title=title,
        hover_data=["minutes_2025", "minutes_2026", "talks_2025", "talks_2026"],
        color_discrete_sequence=["#ADD8E6"],
    ).write_html(path.with_suffix(".html"), include_plotlyjs=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = BASE / "jev-semantic-all/review-segments.parquet"
    metadata = pd.read_parquet(BASE / "talk-metadata/talks.parquet")
    data = (
        pd.read_parquet(source)
        .query("segmentation=='semantic'")
        .merge(metadata[["episode_id", "programme_year"]], on="episode_id", validate="many_to_one")
    )
    talk_coverage = (
        data.groupby("episode_id")
        .agg(
            title=("title", "first"),
            programme_year=("programme_year", "first"),
            segments=("segment_key", "size"),
            assigned_segments=("jev_topic_id", lambda v: v.ne(-1).sum()),
            unassigned_segments=("jev_topic_id", lambda v: v.eq(-1).sum()),
        )
        .reset_index()
    )
    talk_coverage["only_unassigned"] = talk_coverage.assigned_segments.eq(0)
    coverage_summary = (
        talk_coverage.groupby("programme_year")
        .agg(
            total_talks=("episode_id", "size"),
            only_unassigned_talks=("only_unassigned", "sum"),
        )
        .reset_index()
    )
    coverage_summary["percent_only_unassigned"] = (
        coverage_summary.only_unassigned_talks / coverage_summary.total_talks * 100
    )
    coverage_summary = pd.concat(
        [
            coverage_summary,
            pd.DataFrame(
                [
                    dict(
                        programme_year="Combined",
                        total_talks=len(talk_coverage),
                        only_unassigned_talks=int(talk_coverage.only_unassigned.sum()),
                        percent_only_unassigned=talk_coverage.only_unassigned.mean() * 100,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    coverage_summary.to_csv(OUT / "talks-only-unassigned-summary.csv", index=False)
    talk_coverage.loc[talk_coverage.only_unassigned].to_csv(
        OUT / "talks-only-unassigned.csv", index=False
    )
    talk_coverage.to_csv(OUT / "talk-assignment-coverage.csv", index=False)
    parent = compare(data, ["jev_high_level_id", "jev_high_level_name"])
    fine = compare(
        data, ["jev_topic_id", "jev_topic_name", "jev_high_level_id", "jev_high_level_name"]
    )
    for year in [2025, 2026]:
        totals = parent.set_index("jev_high_level_id")[f"minutes_{year}"]
        fine[f"within_parent_share_{year}"] = (
            fine[f"minutes_{year}"].div(fine.jev_high_level_id.map(totals).replace(0, np.nan)) * 100
        )
    fine["within_parent_delta_pp"] = fine.within_parent_share_2026 - fine.within_parent_share_2025
    # Exact decomposition: parent expansion at the old subtopic mix, then mix shift
    # evaluated at the new parent's share. Terms are order-dependent, not causal.
    p = parent.set_index("jev_high_level_id")
    fine["parent_scale_component_pp"] = (
        fine.jev_high_level_id.map(p.time_share_delta_pp) * fine.within_parent_share_2025 / 100
    )
    fine["within_parent_mix_component_pp"] = (
        fine.jev_high_level_id.map(p.time_share_2026) * fine.within_parent_delta_pp / 100
    )
    np.testing.assert_allclose(
        fine.parent_scale_component_pp + fine.within_parent_mix_component_pp,
        fine.time_share_delta_pp,
        atol=1e-10,
    )
    parent.to_csv(OUT / "main-topic-changes.csv", index=False)
    fine.to_csv(OUT / "detailed-topic-changes.csv", index=False)
    emerged = fine.loc[fine.status.eq("only_observed_2026") & fine.jev_topic_id.ne(-1)]
    absent = fine.loc[fine.status.eq("only_observed_2025") & fine.jev_topic_id.ne(-1)]
    emerged.sort_values("time_share_2026", ascending=False).to_csv(
        OUT / "only-observed-2026.csv", index=False
    )
    absent.sort_values("time_share_2025", ascending=False).to_csv(
        OUT / "only-observed-2025.csv", index=False
    )
    change_plot(
        parent,
        "jev_high_level_name",
        "time_share_delta_pp",
        "Main topic shares of all segmented time: 2026 vs 2025",
        OUT / "main-time-share-changes",
    )
    change_plot(
        parent,
        "jev_high_level_name",
        "talk_coverage_delta_pp",
        "Main topic talk coverage: 2026 vs 2025",
        OUT / "main-talk-coverage-changes",
    )
    change_plot(
        fine.loc[fine.jev_topic_id.ne(-1)],
        "jev_topic_name",
        "time_share_delta_pp",
        "Largest detailed-topic time-share changes",
        OUT / "detailed-time-share-changes",
        30,
    )
    for key, group in fine.loc[fine.jev_topic_id.ne(-1)].groupby("jev_high_level_id"):
        group.to_csv(OUT / f"subtopics-{key}.csv", index=False)
        change_plot(
            group,
            "jev_topic_name",
            "within_parent_delta_pp",
            f"Within {group.jev_high_level_name.iloc[0]}: subtopic share changes",
            OUT / f"subtopics-{key}",
            15,
        )
    cohort = data.groupby("programme_year").agg(
        talks=("episode_id", "nunique"),
        segments=("segment_key", "size"),
        hours=("duration_seconds", lambda v: v.sum() / 3600),
    )
    cohort.to_csv(OUT / "cohorts.csv")
    notes = """# Topic changes: 2025 vs 2026

This compares **available transcribed recordings**, not the complete festival programmes.
The corpus has 114 talks from 2025 and 105 from 2026. Podcast publication ends in
May 2026 for the 2025 subset and September 2026 for the 2026 subset; programme areas
may have been uploaded in batches. Apparent shifts can therefore reflect recording
availability and selected programme areas, not a change in society or festival-wide
priorities. No claim of significance or causal evolution is made.

The shared 236-topic Jev vocabulary and 20-category dictionary keep labels comparable.
Primary percentages use each year's **total segmented time, including unassigned**.
Talk coverage is the proportion of that year's talks containing a topic; it is
nonadditive. Segment-share and assigned-time-only alternatives are included in CSVs
(unassigned rows should not be interpreted as assigned-time shares).

Within-parent percentages divide each subtopic's time by its category's time in the
same year. A subtopic can gain global share just because its whole category grew;
within-parent changes show shifts in the category's composition. CSV decomposition
splits global change into parent expansion at the old mix, plus mix change at the
new parent size; it is an accounting identity whose components depend on ordering.

“Only observed in 2026/2025” means zero saved primary-label segments in the other
year, not that the subject was newly invented or disappeared. All such topics are
exported. A separate support flag requires at least 10 minutes across 2 talks in the
observed year; sparse examples remain visible. Categories/subtopics have no refit.
"""
    report = notes + "\n## What changed in these recordings\n\n"
    report += (
        "AI and digital technology expanded from 0.73% to 8.21% of discussion time "
        "(+7.47 percentage points); healthcare grew from 6.85% to 10.76% (+3.91 pp). "
        "Place/housing rose by 3.58 pp, migration by 3.04 pp and family by 2.87 pp. "
        "The largest decreases were governance (−7.58 pp), nature/forestry/resources "
        "(−7.56 pp), defence (−7.18 pp) and international affairs (−5.31 pp).\n\n"
        "Inside AI, agent responsibility/control rose from 3.03% to 25.13% of AI time; "
        "prompt construction rose from 2.27% to 12.32%. AI impact on work fell "
        "from 55.19% to 20.90% of the category, while its share of ALL discussion "
        "time still increased by 1.31 pp. This is relative redistribution inside "
        "a much larger category, not disappearance of work-related AI.\n\n"
        "Energy's overall share was almost stable (5.91% → 6.05%), yet its mix "
        "changed: electricity consumption/production targets rose from 4.51% "
        "to 21.45% of energy time, and exchange pricing from 0.82% to 13.47%. "
        "Biogas/hydrogen fell from 15.94% to 5.18%, and turbine infrasound "
        "from 11.47% to 0.40%.\n\n"
        "Education slightly declined overall (13.73% → 12.45%), but early-childhood "
        "education responsibility/quality accounted for 17.71% of education in 2026 "
        "and had no primary-label segments in the 2025 sample. Within healthcare, "
        "health-data use grew from 1.29% to 12.43% of the category. These patterns "
        "can reflect the programme areas available in each recording subset.\n"
    )
    report += "\n## Main topic changes\n\n"
    cols = [
        "jev_high_level_name",
        "time_share_2025",
        "time_share_2026",
        "time_share_delta_pp",
        "talk_coverage_2025",
        "talk_coverage_2026",
    ]
    report += (
        parent.sort_values("time_share_delta_pp", ascending=False)[cols]
        .round(2)
        .to_markdown(index=False)
    )
    report += "\n\n## Observed emergence and absence\n\n"
    report += (
        f"{len(emerged)} detailed topics only observed in 2026 "
        f"({int(emerged.supported_2026.sum())} meet the support flag); "
    )
    report += f"{len(absent)} only observed in 2025 ({int(absent.supported_2025.sum())} meet it).\n"
    for title, table, year in [
        ("Only observed in 2026", emerged, 2026),
        ("Only observed in 2025", absent, 2025),
    ]:
        report += f"\n### {title}: largest by time share\n\n"
        report += (
            table.sort_values(f"time_share_{year}", ascending=False)
            .head(15)[
                [
                    "jev_topic_name",
                    f"time_share_{year}",
                    f"minutes_{year}",
                    f"talks_{year}",
                    f"supported_{year}",
                ]
            ]
            .round(2)
            .to_markdown(index=False)
            + "\n"
        )
    report += "\n## Inside each main topic\n"
    for key, group in fine.loc[fine.jev_topic_id.ne(-1)].groupby("jev_high_level_id"):
        report += f"\n### {group.jev_high_level_name.iloc[0]}\n\n"
        selected = group.loc[group.within_parent_delta_pp.abs().nlargest(5).index]
        report += (
            selected[
                [
                    "jev_topic_name",
                    "within_parent_share_2025",
                    "within_parent_share_2026",
                    "within_parent_delta_pp",
                    "time_share_delta_pp",
                    "talks_2025",
                    "talks_2026",
                ]
            ]
            .round(2)
            .to_markdown(index=False)
            + "\n"
        )
    (OUT / "COMPARISON.md").write_text(report + "\n")
    (OUT / "METHODS.md").write_text(notes)
    # Keep the offline report dependency-free: escaped prose plus real HTML tables.
    import html

    page = (
        '<!doctype html><meta charset="utf-8"><style>'
        "body{font:15px system-ui;max-width:1300px;margin:30px auto}"
        "img{width:100%}table{border-collapse:collapse}"
        "td,th{padding:6px;border:1px solid #ddd}</style>"
    )
    page += (
        '<h1>2025–2026 topic comparison</h1><pre style="white-space:pre-wrap">'
        + html.escape(notes)
        + "</pre>"
    )
    for image in [
        "main-time-share-changes",
        "main-talk-coverage-changes",
        "detailed-time-share-changes",
    ]:
        page += f'<img src="{image}.png">'
    page += parent[cols].round(2).to_html(index=False)
    for key, group in fine.loc[fine.jev_topic_id.ne(-1)].groupby("jev_high_level_id"):
        page += "<h2>" + html.escape(group.jev_high_level_name.iloc[0]) + "</h2>"
        page += f'<img src="subtopics-{key}.png">' + group.round(2).to_html(index=False)
    (OUT / "index.html").write_text(page)
    (OUT / "manifest.json").write_text(
        json.dumps(
            dict(
                source="../year-specific exports from saved Jev assignments",
                years=[2025, 2026],
                vocabulary="semantic:bge:leaf-local-6-2:42",
                support_minutes=MIN_MINUTES,
                support_talks=MIN_TALKS,
                only_observed_2026=len(emerged),
                only_observed_2025=len(absent),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(
                    (BASE / "talk-metadata/talks.parquet").read_bytes()
                ).hexdigest(),
                api_requests=0,
                interpretation="Descriptive available-recording comparison",
            ),
            indent=2,
        )
        + "\n"
    )
    print(cohort.to_string())
    print("Only observed 2026:", len(emerged), "2025:", len(absent))


if __name__ == "__main__":
    main()
