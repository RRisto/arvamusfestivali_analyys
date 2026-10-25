"""Build a standalone, clickable review page from one diarized transcript."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .text_quality import has_meaningful_text

LOW_CONFIDENCE_THRESHOLD = 0.65
SPEAKER_COLORS = (
    "#2563eb",
    "#db2777",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#dc2626",
    "#4f46e5",
)


def merge_speaker_cues(
    cues: Sequence[Mapping[str, Any]],
    *,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Merge consecutive cues carrying the same diarized speaker label."""
    grouped: list[dict[str, Any]] = []
    for cue in sorted(cues, key=lambda item: float(item["start_seconds"])):
        speaker = str(cue.get("speaker") or "UNASSIGNED")
        start = float(cue["start_seconds"])
        end = float(cue["end_seconds"])
        text = str(cue.get("text", "")).strip()
        if not has_meaningful_text(text):
            continue
        share = float(cue.get("speaker_share", 0.0))
        if grouped and grouped[-1]["speaker"] == speaker:
            current = grouped[-1]
            current["end"] = end
            current["text"] = " ".join(part for part in (current["text"], text) if part)
            current["cue_count"] += 1
            current["minimum_speaker_share"] = min(
                float(current["minimum_speaker_share"]), share
            )
            current["needs_review"] = (
                float(current["minimum_speaker_share"]) < low_confidence_threshold
            )
            continue
        grouped.append(
            {
                "id": f"segment-{len(grouped) + 1:04d}",
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "cue_count": 1,
                "minimum_speaker_share": share,
                "needs_review": share < low_confidence_threshold,
            }
        )
    return grouped


def _load_object(path: Path, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _format_time(seconds: float) -> str:
    rounded = max(0, round(seconds))
    return f"{rounded // 60}:{rounded % 60:02d}"


def _speaker_color(speaker: str, speakers: Sequence[str]) -> str:
    try:
        index = speakers.index(speaker)
    except ValueError:
        index = len(speakers)
    return SPEAKER_COLORS[index % len(SPEAKER_COLORS)]


def _segment_markup(segment: Mapping[str, Any], speakers: Sequence[str]) -> str:
    speaker = str(segment["speaker"])
    color = _speaker_color(speaker, speakers)
    review_class = " needs-review" if segment["needs_review"] else ""
    review_badge = (
        '<span class="review-badge">Kontrolli kõnelejat</span>'
        if segment["needs_review"]
        else ""
    )
    text = html.escape(str(segment["text"]))
    start = float(segment["start"])
    end = float(segment["end"])
    return (
        f'<button type="button" class="segment-card{review_class}" '
        f'data-segment-id="{html.escape(str(segment["id"]), quote=True)}" '
        f'data-start="{start:g}" style="--speaker-color:{color}">'
        '<span class="segment-head">'
        f'<strong>{html.escape(speaker)}</strong>'
        f'<span class="timestamp">▶ {_format_time(start)}–{_format_time(end)}</span>'
        f"{review_badge}</span>"
        f'<span class="segment-text">{text}</span>'
        "</button>"
    )


def _render_page(
    episode: Mapping[str, Any],
    diarization: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    *,
    stats_available: bool = False,
) -> str:
    title = str(episode["title"])
    audio_url = str(episode["audio_url"])
    page_url = str(episode.get("page_url", ""))
    speakers = [str(item) for item in diarization.get("speakers", [])]
    review_count = sum(bool(segment["needs_review"]) for segment in segments)
    legend = "".join(
        f'<span class="speaker-chip" style="--speaker-color:{_speaker_color(speaker, speakers)}">'
        f"{html.escape(speaker)}</span>"
        for speaker in speakers
    )
    cards = "\n".join(_segment_markup(segment, speakers) for segment in segments)
    source_link = (
        f'<a href="{html.escape(page_url, quote=True)}" target="_blank" rel="noreferrer">'
        "Ava salvestuse leht ↗</a>"
        if page_url
        else ""
    )
    stats_link = (
        f'<a href="{html.escape(str(episode["id"]), quote=True)}-stats.html">'
        "View discussion statistics ↗</a>"
        if stats_available
        else ""
    )
    return f"""<!doctype html>
<html lang="et">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} — kõnelejate ülevaade</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f5f7fb; color: #172033; }}
    header, main {{ width: min(920px, calc(100% - 32px)); margin-inline: auto; }}
    header {{ padding: 38px 0 20px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(1.65rem, 4vw, 2.65rem); line-height: 1.08; }}
    .summary {{ color: #59657a; margin: 0 0 18px; }}
    .source-link a {{ color: #2458c6; font-weight: 700; }}
    .player-dock {{
      position: sticky; top: 0; z-index: 5; background: rgba(245,247,251,.94);
      backdrop-filter: blur(12px); border-block: 1px solid #dce2ec; padding: 14px 0;
    }}
    .player-inner {{ width: min(920px, calc(100% - 32px)); margin-inline: auto; }}
    audio {{ display: block; width: 100%; }}
    #now-playing {{ min-height: 1.4em; margin: 8px 0 0; color: #445069; font-size: .92rem; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 22px 0 14px; }}
    .speaker-chip {{
      border-left: 5px solid var(--speaker-color); background: white; border-radius: 8px;
      padding: 7px 10px; box-shadow: 0 1px 3px #17203314;
      font-weight: 750; font-size: .82rem;
    }}
    .segments {{ display: grid; gap: 12px; padding: 4px 0 54px; }}
    .segment-card {{
      width: 100%; border: 1px solid #dce2ec;
      border-left: 7px solid var(--speaker-color); border-radius: 12px;
      background: white; color: inherit; padding: 16px 18px; text-align: left;
      cursor: pointer; box-shadow: 0 2px 8px #1720330c;
      transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
    }}
    .segment-card:hover, .segment-card:focus-visible {{
      transform: translateY(-1px); box-shadow: 0 7px 20px #1720331a;
      outline: 3px solid #93b4ff; outline-offset: 2px;
    }}
    .segment-card.active {{
      border-color: var(--speaker-color);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--speaker-color) 22%, transparent);
    }}
    .segment-card.needs-review {{ background: #fffaf0; }}
    .segment-head {{
      display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 9px;
    }}
    .timestamp {{ color: #526077; font-variant-numeric: tabular-nums; }}
    .review-badge {{
      background: #f8d98b; color: #6c4400; border-radius: 999px;
      padding: 3px 8px; font-size: .74rem; font-weight: 800;
    }}
    .segment-text {{ display: block; font-size: 1.02rem; line-height: 1.62; }}
    @media (max-width: 560px) {{
      header, main, .player-inner {{ width: min(100% - 20px, 920px); }}
      .segment-card {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="summary">
      {len(speakers)} kõnelejat · {len(segments)} segmenti ·
      {review_count} vajab kõneleja kontrolli
    </p>
    <p class="source-link">{source_link} {stats_link}</p>
  </header>
  <div class="player-dock">
    <div class="player-inner">
      <audio id="episode-audio" controls preload="metadata"
             src="{html.escape(audio_url, quote=True)}"></audio>
      <p id="now-playing">Vali allpool segment, et alustada kuulamist.</p>
    </div>
  </div>
  <main>
    <div class="legend" aria-label="Kõnelejad">{legend}</div>
    <div class="segments">{cards}</div>
  </main>
  <script>
    const player = document.getElementById("episode-audio");
    const nowPlaying = document.getElementById("now-playing");
    const cards = [...document.querySelectorAll(".segment-card")];
    cards.forEach((card) => {{
      card.addEventListener("click", () => {{
        const start = Number(card.dataset.start);
        player.currentTime = start;
        player.play().catch(() => {{}});
        cards.forEach((item) => item.classList.toggle("active", item === card));
        nowPlaying.textContent = card.querySelector("strong").textContent
          + " · "
          + card.querySelector(".timestamp").textContent.replace("▶ ", "");
      }});
    }});
  </script>
</body>
</html>
"""


def build_review_page(transcript_path: Path, diarization_path: Path, output: Path) -> Path:
    """Create one self-contained HTML page for reviewing speaker-labelled text."""
    transcript = _load_object(transcript_path, "transcript archive")
    diarization = _load_object(diarization_path, "diarization output")
    episode = transcript.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("transcript archive is missing episode metadata")
    episode_id = str(episode.get("id", ""))
    if episode_id != str(diarization.get("episode_id")):
        raise ValueError("transcript and diarization episode IDs do not match")
    cues = diarization.get("cues")
    if not isinstance(cues, list):
        raise ValueError("diarization output is missing cues")
    segments = merge_speaker_cues(cues)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_page = output.with_name(f"{episode_id}-stats.html")
    output.write_text(
        _render_page(episode, diarization, segments, stats_available=stats_page.is_file()),
        encoding="utf-8",
    )
    return output


def _project_root(transcript: Path) -> Path:
    if transcript.parent.parent.name == "transcripts":
        return transcript.parent.parent.parent.parent
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a clickable diarization review page")
    parser.add_argument("transcript", type=Path, help="canonical transcript JSON")
    parser.add_argument("--diarization", type=Path, help="diarization JSON; inferred by default")
    parser.add_argument("--output", type=Path, help="HTML output; inferred by default")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        transcript = args.transcript.resolve()
        archive = _load_object(transcript, "transcript archive")
        episode = archive.get("episode", {})
        episode_id = str(episode["id"])
        year = str(episode.get("published_at", transcript.parent.name))[:4]
        root = _project_root(transcript)
        diarization = args.diarization or (
            root / "data" / "diarization" / year / f"{episode_id}.json"
        )
        output = args.output or root / "site" / "diarization" / f"{episode_id}.html"
        result = build_review_page(transcript, diarization.resolve(), output.resolve())
        print(f"wrote {result}")
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
