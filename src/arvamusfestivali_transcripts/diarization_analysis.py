"""Summary statistics for speaker-diarization result files."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .text_quality import has_meaningful_text


def _valid_turns(payload: dict[str, Any]) -> list[tuple[str, float]]:
    valid: list[tuple[str, float]] = []
    for turn in payload.get("turns", []):
        try:
            start = float(turn["start_seconds"])
            end = float(turn["end_seconds"])
            speaker = str(turn["speaker"])
        except (KeyError, TypeError, ValueError):
            continue
        duration = end - start
        if not speaker or not math.isfinite(duration) or duration <= 0:
            continue
        valid.append((speaker, duration))
    return valid


def _gini(values: list[float]) -> float | None:
    if not values or sum(values) <= 0:
        return None
    ordered = sorted(values)
    count = len(ordered)
    weighted_sum = sum((2 * index - count - 1) * value for index, value in enumerate(ordered, 1))
    return weighted_sum / (count * sum(ordered))


def _normalized_hhi(shares: list[float]) -> float | None:
    if not shares:
        return None
    if len(shares) == 1:
        return 1.0
    hhi = sum(share**2 for share in shares)
    equal_share_hhi = 1 / len(shares)
    return (hhi - equal_share_hhi) / (1 - equal_share_hhi)


def participation_metrics(speaker_durations: dict[str, float]) -> dict[str, Any]:
    """Calculate duration-based participation equality for a group of speakers."""
    durations = sorted(
        (
            float(duration)
            for duration in speaker_durations.values()
            if math.isfinite(float(duration)) and float(duration) > 0
        ),
        reverse=True,
    )
    total_duration = sum(durations)
    shares = [duration / total_duration for duration in durations]
    hhi = sum(share**2 for share in shares) if shares else None
    return {
        "unique_speakers": len(durations),
        "total_speech_seconds": total_duration,
        "dominant_speaker_share": shares[0] if shares else None,
        "top_two_speaker_share": sum(shares[:2]) if shares else None,
        "normalized_hhi": _normalized_hhi(shares),
        "gini": _gini(durations),
        "effective_speaker_count": 1 / hhi if hhi else None,
    }


def align_topic_chunks_to_speakers(
    topic_chunks: list[dict[str, Any]], turns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Allocate topic-chunk time to every diarization turn with positive overlap."""
    aligned: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(topic_chunks):
        try:
            chunk_start = float(chunk["start_seconds"])
            chunk_end = float(chunk["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(chunk_end - chunk_start) or chunk_end <= chunk_start:
            continue
        chunk_metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {"start_seconds", "end_seconds"}
        }
        for turn in turns:
            try:
                turn_start = float(turn["start_seconds"])
                turn_end = float(turn["end_seconds"])
                speaker = str(turn["speaker"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap = min(chunk_end, turn_end) - max(chunk_start, turn_start)
            if not speaker or not math.isfinite(overlap) or overlap <= 0:
                continue
            aligned.append(
                {
                    "topic_chunk_index": chunk_index,
                    "chunk_start_seconds": chunk_start,
                    "chunk_end_seconds": chunk_end,
                    **chunk_metadata,
                    "speaker": speaker,
                    "overlap_seconds": overlap,
                }
            )
    return aligned


def speaker_transition_rows(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count transitions after collapsing adjacent turns with the same speaker label."""
    ordered: list[tuple[float, float, str]] = []
    for turn in turns:
        try:
            start = float(turn["start_seconds"])
            end = float(turn["end_seconds"])
            speaker = str(turn["speaker"])
        except (KeyError, TypeError, ValueError):
            continue
        if speaker and math.isfinite(end - start) and end > start:
            ordered.append((start, end, speaker))
    ordered.sort()

    collapsed: list[str] = []
    for _, _, speaker in ordered:
        if not collapsed or collapsed[-1] != speaker:
            collapsed.append(speaker)
    counts = Counter(zip(collapsed, collapsed[1:], strict=False))
    return [
        {
            "from_speaker": from_speaker,
            "to_speaker": to_speaker,
            "transition_count": count,
        }
        for (from_speaker, to_speaker), count in sorted(counts.items())
    ]


def primary_topic_summary(
    rows: list[dict[str, Any]],
    *,
    topic_id_key: str,
    topic_name_key: str,
    duration_key: str = "duration_seconds",
) -> dict[str, Any]:
    """Select the primary topic by cumulative valid segment duration."""
    durations: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        try:
            topic_id = str(row[topic_id_key])
            topic_name = str(row[topic_name_key])
            duration = float(row[duration_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not topic_id or not topic_name or not math.isfinite(duration) or duration <= 0:
            continue
        durations[(topic_id, topic_name)] += duration

    total_duration = sum(durations.values())
    if not durations:
        return {
            "primary_topic_id": None,
            "primary_topic_name": None,
            "primary_topic_share": None,
            "topic_count": 0,
            "topic_duration_seconds": 0.0,
        }

    primary = min(durations, key=lambda topic: (-durations[topic], topic))
    return {
        "primary_topic_id": primary[0],
        "primary_topic_name": primary[1],
        "primary_topic_share": durations[primary] / total_duration,
        "topic_count": len(durations),
        "topic_duration_seconds": total_duration,
    }


def speaker_duration_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one row per observed speaker, ranked by total speaking duration."""
    durations: dict[str, float] = defaultdict(float)
    segment_counts: dict[str, int] = defaultdict(int)
    for speaker, duration in _valid_turns(payload):
        durations[speaker] += duration
        segment_counts[speaker] += 1

    total_duration = sum(durations.values())
    ranked = sorted(durations, key=lambda speaker: (-durations[speaker], speaker))
    return [
        {
            "episode_id": str(payload.get("episode_id", "")),
            "speaker": speaker,
            "speech_seconds": durations[speaker],
            "segment_count": segment_counts[speaker],
            "speech_share": durations[speaker] / total_duration,
            "speaker_rank": rank,
        }
        for rank, speaker in enumerate(ranked, 1)
    ]


def summarize_conversation(
    payload: dict[str, Any], *, short_segment_threshold: float = 0.25
) -> dict[str, Any]:
    """Calculate segment, speaker, duration, and concentration metrics for one episode."""
    turns = payload.get("turns", [])
    valid_turns = _valid_turns(payload)
    durations = [duration for _, duration in valid_turns]
    speaker_rows = speaker_duration_rows(payload)
    shares = [row["speech_share"] for row in speaker_rows]
    hhi = sum(share**2 for share in shares) if shares else None
    cues = payload.get("cues", [])
    text_cue_count = sum(has_meaningful_text(cue.get("text")) for cue in cues)

    return {
        "episode_id": str(payload.get("episode_id", "")),
        "segment_count": len(turns),
        "valid_segment_count": len(valid_turns),
        "invalid_segment_count": len(turns) - len(valid_turns),
        "short_segment_count": sum(duration < short_segment_threshold for duration in durations),
        "text_cue_count": text_cue_count,
        "discarded_text_cue_count": len(cues) - text_cue_count,
        "unique_speakers": len(speaker_rows),
        "total_speech_seconds": sum(durations),
        "median_segment_seconds": median(durations) if durations else None,
        "dominant_speaker_share": shares[0] if shares else None,
        "top_two_speaker_share": sum(shares[:2]) if shares else None,
        "normalized_hhi": _normalized_hhi(shares),
        "gini": _gini([row["speech_seconds"] for row in speaker_rows]),
        "effective_speaker_count": 1 / hhi if hhi else None,
    }


def load_conversation_summaries(
    directory: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load numeric episode JSON files and return conversation and speaker-level rows."""
    summaries: list[dict[str, Any]] = []
    speakers: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.json")):
        if not path.stem.isdigit():
            continue
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        summaries.append(summarize_conversation(payload))
        speakers.extend(speaker_duration_rows(payload))
    return summaries, speakers
