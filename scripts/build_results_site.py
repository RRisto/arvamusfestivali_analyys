"""Build compact JSON assets for the static Arvamusfestival results site."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RESULTS_RELATIVE = Path("data/topic-analysis/results")


def normalize_search_text(value: str) -> str:
    """Return a case- and diacritic-insensitive search representation."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(character for character in decomposed if not unicodedata.combining(character))
    words_only = "".join(character if character.isalnum() else " " for character in plain)
    return re.sub(r"\s+", " ", words_only).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _load_cues(
    root: Path,
    episode_id: str,
    cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if episode_id in cache:
        return cache[episode_id]
    path = root / "data/transcripts/2026" / f"{episode_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Transcript archive missing for episode {episode_id}: {path}")
    archive = json.loads(path.read_text(encoding="utf-8"))
    cues = archive.get("transcription", {}).get("cues", [])
    cache[episode_id] = sorted(cues, key=lambda cue: float(cue["start_seconds"]))
    return cache[episode_id]


def _passage_text(cues: list[dict[str, Any]], start: float, end: float) -> str:
    texts = [
        str(cue.get("text", "")).strip()
        for cue in cues
        if float(cue["end_seconds"]) > start and float(cue["start_seconds"]) < end
    ]
    return " ".join(text for text in texts if text)


def _category_colors(results: Path) -> dict[str, str]:
    path = results / "high-level-topics/category-style.json"
    styles = json.loads(path.read_text(encoding="utf-8"))
    return {key: str(value["color"]) for key, value in styles.items()}


def _overview_payload(
    segments: list[dict[str, str]],
    categories: list[dict[str, str]],
    colors: dict[str, str],
) -> dict[str, object]:
    talks = {row["episode_id"] for row in segments}
    years = sorted({_int(row["programme_year"]) for row in segments})
    category_rows = [
        {
            "id": row["jev_high_level_id"],
            "name": row["jev_high_level_name"],
            "segments": _int(row["segments"]),
            "talks": _int(row["talks"]),
            "hours": round(_float(row["recording_hours"]), 3),
            "color": colors.get(row["jev_high_level_id"], "#6b7280"),
        }
        for row in categories
    ]

    category_organizer_talks: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    organizer_talks: defaultdict[str, set[str]] = defaultdict(set)
    for row in segments:
        organizer = row.get("organizer", "").strip() or "Korraldaja teadmata"
        category = row["jev_high_level_id"]
        episode = row["episode_id"]
        category_organizer_talks[(organizer, category)].add(episode)
        organizer_talks[organizer].add(episode)
    top_organizers = {
        organizer
        for organizer, _ in sorted(
            organizer_talks.items(), key=lambda item: (-len(item[1]), item[0].casefold())
        )[:16]
    }
    organizer_topics = [
        {"organizer": organizer, "category": category, "talks": len(episode_ids)}
        for (organizer, category), episode_ids in category_organizer_talks.items()
        if organizer in top_organizers
    ]
    organizer_topics.sort(key=lambda row: (row["organizer"].casefold(), row["category"]))

    scatter = [
        {
            "id": row["segment_key"],
            "episode": row["episode_id"],
            "title": row["title"],
            "start": round(_float(row["start_seconds"]), 3),
            "end": round(_float(row["end_seconds"]), 3),
            "audio": row["audio_link"],
            "topic": row.get("jev_topic_name", "") or "Määramata",
            "category_id": row["jev_high_level_id"],
            "category": row["jev_high_level_name"],
            "x": round(_float(row["x"]), 6),
            "y": round(_float(row["y"]), 6),
            "year": _int(row["programme_year"]),
            "organizer": row.get("organizer", "").strip() or "Korraldaja teadmata",
        }
        for row in segments
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "metrics": {
            "categories": len(category_rows),
            "segments": len(segments),
            "talks": len(talks),
            "years": years,
        },
        "categories": category_rows,
        "scatter": scatter,
        "organizer_topics": organizer_topics,
    }


def _comparison_payload(
    rows: list[dict[str, str]], segments: list[dict[str, str]], colors: dict[str, str]
) -> dict[str, object]:
    numeric_fields = [
        "minutes_2025",
        "minutes_2026",
        "talks_2025",
        "talks_2026",
        "time_share_2025",
        "time_share_2026",
        "talk_coverage_2025",
        "talk_coverage_2026",
        "time_share_delta_pp",
        "talk_coverage_delta_pp",
    ]
    categories: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {
            "id": row["jev_high_level_id"],
            "name": row["jev_high_level_name"],
            "color": colors.get(row["jev_high_level_id"], "#6b7280"),
        }
        item.update({field: round(_float(row.get(field)), 6) for field in numeric_fields})
        categories.append(item)
    categories.sort(key=lambda row: str(row["name"]).casefold())
    cohorts: dict[str, dict[str, int]] = {}
    for year in sorted({_int(row["programme_year"]) for row in segments}):
        sample = [row for row in segments if _int(row["programme_year"]) == year]
        cohorts[str(year)] = {
            "segments": len(sample),
            "talks": len({row["episode_id"] for row in sample}),
        }
    return {"schema_version": SCHEMA_VERSION, "cohorts": cohorts, "categories": categories}


def _transcript_payload(
    root: Path,
    segments: list[dict[str, str]],
    talk_rows: list[dict[str, str]],
) -> dict[str, object]:
    talks_by_id = {row["episode_id"]: row for row in talk_rows}
    category_sets: defaultdict[str, set[str]] = defaultdict(set)
    cue_cache: dict[str, list[dict[str, Any]]] = {}
    segment_payload: list[dict[str, object]] = []
    for row in segments:
        episode_id = row["episode_id"]
        start = _float(row["start_seconds"])
        end = _float(row["end_seconds"])
        text = _passage_text(_load_cues(root, episode_id, cue_cache), start, end)
        organizer = row.get("organizer", "").strip() or "Korraldaja teadmata"
        topic = row.get("jev_topic_name", "") or "Määramata"
        category = row["jev_high_level_name"]
        category_sets[episode_id].add(row["jev_high_level_id"])
        segment_payload.append(
            {
                "id": row["segment_key"],
                "episode": episode_id,
                "title": row["title"],
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "audio": row["audio_link"],
                "topic": topic,
                "category_id": row["jev_high_level_id"],
                "category": category,
                "year": _int(row["programme_year"]),
                "organizer": organizer,
            }
        )
    segment_payload.sort(key=lambda row: (str(row["title"]).casefold(), float(row["start"])))
    talk_payload = []
    for episode_id in sorted({row["episode_id"] for row in segments}):
        metadata = talks_by_id.get(episode_id, {})
        sample = next(row for row in segments if row["episode_id"] == episode_id)
        talk_payload.append(
            {
                "episode": episode_id,
                "title": (
                    metadata.get("programme_title")
                    or metadata.get("title")
                    or sample["title"]
                ),
                "year": _int(metadata.get("programme_year") or sample["programme_year"]),
                "organizer": metadata.get("organizer", "").strip()
                or sample.get("organizer", "").strip()
                or "Korraldaja teadmata",
                "moderator": metadata.get("programme_moderator", ""),
                "panelists": metadata.get("programme_panelists", ""),
                "stage": metadata.get("programme_stage", ""),
                "page": metadata.get("page_url", ""),
                "audio": metadata.get("audio_url", ""),
                "duration": round(_float(metadata.get("duration_seconds")), 3),
                "categories": sorted(category_sets[episode_id]),
            }
        )
    talk_payload.sort(key=lambda row: str(row["title"]).casefold())
    return {"schema_version": SCHEMA_VERSION, "talks": talk_payload, "segments": segment_payload}


def build_site_data(root: Path, output_dir: Path) -> dict[str, Path]:
    """Build all page-specific JSON payloads and return their paths."""
    root = root.resolve()
    output_dir = output_dir.resolve()
    results = root / RESULTS_RELATIVE
    segments = _read_csv(results / "jev-semantic-all/segment-maps/segment-coordinates.csv")
    categories = _read_csv(results / "jev-semantic-all/high-level-summary.csv")
    comparison = _read_csv(
        results / "jev-topic-analysis/year-comparison/main-topic-changes.csv"
    )
    talks = _read_csv(results / "talk-metadata/talks.csv")
    colors = _category_colors(results)
    payloads = {
        "overview": _overview_payload(segments, categories, colors),
        "comparison": _comparison_payload(comparison, segments, colors),
        "transcripts": _transcript_payload(root, segments, talks),
    }
    return {
        name: _write_json(output_dir / f"{name}.json", payload)
        for name, payload in payloads.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("site/data"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    paths = build_site_data(args.root, output)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
