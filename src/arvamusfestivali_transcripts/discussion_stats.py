"""Build static per-discussion participation and turn-taking statistics pages."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .diarization_analysis import (
    align_topic_chunks_to_speakers,
    participation_metrics,
    speaker_transition_rows,
    summarize_conversation,
)
from .text_quality import has_meaningful_text

RESULTS = Path("data/topic-analysis/results")
WINDOW_SECONDS = 600
STEP_SECONDS = 300


def _load_inputs(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    results = root / RESULTS
    talks = pd.read_parquet(
        results / "talk-metadata/talks.parquet",
        columns=["episode_id", "duplicate_episode_ids", "title", "organizer", "programme_stage"],
    )
    aliases: dict[str, dict[str, Any]] = {}
    for row in talks.to_dict("records"):
        canonical_id = str(row["episode_id"])
        meta = {
            "canonical_episode_id": canonical_id,
            "title": str(row.get("title") or canonical_id),
            "organizer": str(row.get("organizer") or ""),
            "programme_stage": str(row.get("programme_stage") or ""),
        }
        aliases[canonical_id] = meta
        duplicate_ids_value = row.get("duplicate_episode_ids")
        duplicate_ids = (
            json.loads(duplicate_ids_value)
            if isinstance(duplicate_ids_value, str) and duplicate_ids_value
            else duplicate_ids_value
            if isinstance(duplicate_ids_value, list)
            else []
        )
        for alias in duplicate_ids:
            aliases[str(alias)] = meta

    assignments = pd.read_parquet(
        results / "jev-semantic-all/assignments.parquet",
        columns=["segment_key", "jev_high_level_id", "jev_high_level_name"],
    )
    segments = pd.read_parquet(
        results / "segment-dataset/all-segments.parquet",
        columns=["segment_key", "episode_id", "start_seconds", "end_seconds", "text"],
    )
    segments = segments.loc[segments.text.map(has_meaningful_text)].copy()
    chunks = assignments.merge(segments, on="segment_key", how="inner", validate="one_to_one")
    chunks["episode_id"] = chunks.episode_id.astype(str)
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for episode_id, group in chunks.groupby("episode_id", sort=False):
        by_episode[str(episode_id)] = [
            {
                "start_seconds": float(row.start_seconds),
                "end_seconds": float(row.end_seconds),
                "high_level_id": str(row.jev_high_level_id),
                "high_level_name": str(row.jev_high_level_name),
            }
            for row in group.itertuples()
        ]
    return aliases, by_episode


def _participation_over_time(turns: list[dict[str, Any]]) -> dict[str, Any]:
    valid: list[tuple[float, float, str]] = []
    for turn in turns:
        try:
            start = float(turn["start_seconds"])
            end = float(turn["end_seconds"])
            speaker = str(turn["speaker"])
        except (KeyError, TypeError, ValueError):
            continue
        if speaker and math.isfinite(end - start) and end > start:
            valid.append((start, end, speaker))
    if not valid:
        return {"minutes": [], "speaker_shares": {}, "metrics": {}}

    recording_end = max(end for _, end, _ in valid)
    all_speakers = sorted({speaker for _, _, speaker in valid})
    windows: list[dict[str, Any]] = []
    speaker_shares: dict[str, list[float]] = {speaker: [] for speaker in all_speakers}
    stop = max(1, math.ceil(recording_end / STEP_SECONDS)) * STEP_SECONDS
    for window_start in range(0, stop, STEP_SECONDS):
        window_end = min(window_start + WINDOW_SECONDS, recording_end)
        if window_end <= window_start:
            continue
        durations: dict[str, float] = defaultdict(float)
        for start, end, speaker in valid:
            overlap = min(window_end, end) - max(window_start, start)
            if overlap > 0:
                durations[speaker] += overlap
        total = sum(durations.values())
        metrics = participation_metrics(durations)
        minutes = (window_start + window_end) / 120
        windows.append(
            {
                "minute": minutes,
                "dominant_speaker_share": metrics["dominant_speaker_share"],
                "gini": metrics["gini"],
                "effective_speaker_share": (
                    metrics["effective_speaker_count"] / metrics["unique_speakers"]
                    if metrics["effective_speaker_count"] and metrics["unique_speakers"]
                    else None
                ),
            }
        )
        for speaker in all_speakers:
            speaker_shares[speaker].append(durations.get(speaker, 0.0) / total if total else 0.0)
    ordered_speakers = sorted(
        speaker_shares,
        key=lambda speaker: (-sum(speaker_shares[speaker]) / max(1, len(windows)), speaker),
    )
    visible_speakers = ordered_speakers[:8]
    omitted_speakers = ordered_speakers[8:]
    visible_shares = {speaker: speaker_shares[speaker] for speaker in visible_speakers}
    if omitted_speakers:
        visible_shares["Other speakers"] = [
            sum(speaker_shares[speaker][index] for speaker in omitted_speakers)
            for index in range(len(windows))
        ]
    return {
        "minutes": [row["minute"] for row in windows],
        "speaker_shares": visible_shares,
        "metrics": {
            key: [row[key] for row in windows]
            for key in ("dominant_speaker_share", "gini", "effective_speaker_share")
        },
    }


def _discussion_payload(
    episode_id: str,
    canonical_id: str,
    title: str,
    organizer: str,
    programme_stage: str,
    audio_url: str,
    diarization: dict[str, Any],
    topic_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    turns = diarization.get("turns", [])
    topic_overlap = align_topic_chunks_to_speakers(topic_chunks, turns)
    topic_totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in topic_overlap:
        key = (str(row["speaker"]), str(row["high_level_id"]), str(row["high_level_name"]))
        topic_totals[key] += float(row["overlap_seconds"])
    speaker_totals: dict[str, float] = defaultdict(float)
    topic_seconds: dict[tuple[str, str], float] = defaultdict(float)
    for (speaker, topic_id, topic_name), seconds in topic_totals.items():
        speaker_totals[speaker] += seconds
        topic_seconds[(topic_id, topic_name)] += seconds
    speakers = sorted(speaker_totals, key=lambda speaker: (-speaker_totals[speaker], speaker))
    topics = sorted(topic_seconds, key=lambda topic: (-topic_seconds[topic], topic[1]))
    topic_matrix = [
        [
            topic_totals.get((speaker, topic_id, topic_name), 0.0)
            / topic_seconds[(topic_id, topic_name)]
            if topic_seconds[(topic_id, topic_name)]
            else 0.0
            for topic_id, topic_name in topics
        ]
        for speaker in speakers
    ]

    transition_rows = speaker_transition_rows(turns)
    transition_speakers = sorted(
        {str(row["from_speaker"]) for row in transition_rows}
        | {str(row["to_speaker"]) for row in transition_rows}
    )
    transition_counts: dict[tuple[str, str], int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    for row in transition_rows:
        source = str(row["from_speaker"])
        target = str(row["to_speaker"])
        count = int(row["transition_count"])
        transition_counts[(source, target)] += count
        outgoing[source] += count
    transition_matrix = [
        [
            transition_counts[(source, target)] / outgoing[source] if outgoing[source] else 0.0
            for target in transition_speakers
        ]
        for source in transition_speakers
    ]
    participation = _participation_over_time(turns)
    summary = summarize_conversation(diarization)
    return {
        "episode_id": episode_id,
        "canonical_episode_id": canonical_id,
        "title": title,
        "organizer": organizer,
        "programme_stage": programme_stage,
        "audio_url": audio_url,
        "speaker_count": summary["unique_speakers"],
        "speech_minutes": summary["total_speech_seconds"] / 60,
        "speaker_switches": sum(int(row["transition_count"]) for row in transition_rows),
        "participation": participation,
        "speaker_topic": {
            "speakers": speakers,
            "topics": [topic_name for _, topic_name in topics],
            "shares": topic_matrix,
        },
        "transitions": {"speakers": transition_speakers, "shares": transition_matrix},
    }


def _render_stats_page(episode_id: str, title: str) -> str:
    safe_id = html.escape(episode_id, quote=True)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title} — discussion statistics</title>
<script src="https://cdn.plot.ly/plotly-3.1.0.min.js" defer></script>
<style>
:root{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f5f7fb}}
*{{box-sizing:border-box}}
body{{margin:0}}
header,main{{width:min(1120px,calc(100% - 32px));margin-inline:auto}}
header{{padding:30px 0 16px}}
h1{{font-size:clamp(1.6rem,4vw,2.5rem);margin:0 0 8px}}
a{{color:#2458c6;font-weight:700}}.meta{{color:#59657a;margin:8px 0}}
.player{{position:sticky;top:0;z-index:3;background:#f5f7f7ed;
  backdrop-filter:blur(12px);padding:12px max(16px,calc((100vw - 1120px)/2));
  border-block:1px solid #dce2ec}}
audio{{width:100%}}
main{{padding:18px 0 54px}}
.summary{{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 18px}}
.pill{{background:white;border:1px solid #dce2ec;border-radius:999px;
  padding:7px 12px;font-size:.9rem}}
.chart-card{{background:white;border:1px solid #dce2ec;border-radius:14px;
  padding:14px 16px;margin:14px 0;box-shadow:0 2px 8px #1720330b}}
h2{{margin:4px 0 0;font-size:1.22rem}}
.chart{{min-height:300px}}
.note{{color:#59657a}}
@media(max-width:600px){{header,main{{width:calc(100% - 20px)}}
  .chart{{min-height:250px}}}}
</style></head><body>
<header><p><a href="{safe_id}.html">← Listen and read transcript</a></p><h1>{safe_title}</h1>
<p class="meta">Conversation statistics · <span id="organizer"></span></p>
<div id="summary" class="summary"></div></header>
<div class="player"><audio id="audio" controls preload="metadata"></audio></div>
<main>
<section class="chart-card">
<h2>Participation over time</h2>
<p class="note">Ten-minute windows advance every five minutes. Turns crossing a window
boundary are divided by overlap. Click the chart to seek the recording.</p>
<div id="participation" class="chart"></div><div id="equality" class="chart"></div>
</section>
<section class="chart-card">
<h2>Speaker–topic participation</h2>
<p class="note">Each cell shows the speaker’s share of aligned speech within that
high-level topic.</p>
<div id="speaker-topic" class="chart"></div>
<p id="no-topics" class="note" hidden>No matching topic assignments are available
for this discussion.</p>
</section>
<section class="chart-card">
<h2>Turn-transition structure</h2>
<p class="note">Adjacent runs by the same speaker are collapsed. Rows show the share
of each speaker’s outgoing turns passed to another speaker.</p>
<div id="transitions" class="chart"></div>
</section>
</main><script src="../assets/discussion-stats.js" defer></script>
<script>window.DISCUSSION_STATS_ID="{safe_id}";</script></body></html>
"""


def build_discussion_stats_site(project_root: str | Path) -> tuple[int, int]:
    """Write one listening-linked static stats page for each diarization recording."""
    root = Path(project_root).resolve()
    aliases, chunks_by_episode = _load_inputs(root)
    transcript_root = root / "data/transcripts"
    diarization_root = root / "data/diarization"
    stats_root = root / "site/diarization"
    json_root = root / "site/data/discussion-stats"
    page_count = 0
    for diarization_path in sorted(diarization_root.glob("*/*.json")):
        episode_id = diarization_path.stem
        if not episode_id.isdigit():
            continue
        transcript_path = next(transcript_root.glob(f"*/{episode_id}.json"), None)
        if transcript_path is None:
            continue
        diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        episode = transcript.get("episode", {})
        metadata = aliases.get(episode_id, {})
        canonical_id = str(metadata.get("canonical_episode_id", episode_id))
        title = str(episode.get("title") or metadata.get("title") or episode_id)
        payload = _discussion_payload(
            episode_id=episode_id,
            canonical_id=canonical_id,
            title=title,
            organizer=str(metadata.get("organizer") or ""),
            programme_stage=str(metadata.get("programme_stage") or ""),
            audio_url=str(episode.get("audio_url") or ""),
            diarization=diarization,
            topic_chunks=chunks_by_episode.get(canonical_id, []),
        )
        data_path = json_root / f"{episode_id}.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        page_path = stats_root / f"{episode_id}-stats.html"
        page_path.write_text(_render_stats_page(episode_id, title), encoding="utf-8")
        page_count += 1
    from .diarization_batch import build_review_index

    build_review_index(root)
    return page_count, len(list(json_root.glob("*.json"))) if json_root.exists() else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    pages, _ = build_discussion_stats_site(args.root)
    print(f"wrote {pages} discussion statistics pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
