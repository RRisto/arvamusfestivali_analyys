# ruff: noqa: E402
"""Rank possible cross-recording speaker matches from short clean audio samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arvamusfestivali_transcripts.download import audio_path, download_episode
from arvamusfestivali_transcripts.schema import Episode
from arvamusfestivali_transcripts.speaker_voice_matching import (
    rank_named_candidates,
    select_speaker_cues,
)

MODEL_ID = "microsoft/wavlm-base-plus-sv"
SAMPLE_RATE = 16_000


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _transcript_path(root: Path, episode_id: str) -> Path:
    return root / "data" / "transcripts" / "2026" / f"{episode_id}.json"


def _diarization_path(root: Path, episode_id: str) -> Path:
    return root / "data" / "diarization" / "2026" / f"{episode_id}.json"


def _audio_episode(root: Path, episode_id: str) -> Episode:
    payload = _load_json(_transcript_path(root, episode_id))["episode"]
    published_at = datetime.fromisoformat(payload["published_at"].replace("Z", "+00:00"))
    return Episode(
        id=str(payload["id"]),
        rss_guid=str(payload["rss_guid"]),
        title=str(payload["title"]),
        published_at=published_at,
        published_raw="",
        page_url=str(payload["page_url"]),
        audio_url=str(payload["audio_url"]),
        audio_bytes=payload.get("audio_bytes"),
        duration_seconds=float(payload.get("duration_seconds", 0)),
    )


def _speaker_names(
    root: Path, *, include_voice_matches: bool = True
) -> dict[str, dict[str, str]]:
    path = root / "data" / "diarization" / "speaker-name-detections-2026.csv"
    result: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("confidence") not in {"High", "Medium"}:
                continue
            if (
                not include_voice_matches
                and row.get("method", "").strip() == "cross_recording_voice_match"
            ):
                continue
            episode_id = row.get("episode_id", "").strip()
            speaker = row.get("speaker_label", "").strip()
            name = row.get("suggested_name", "").strip()
            if episode_id and speaker and name:
                result[episode_id][speaker] = name
    return result


def _programme_roster(path: Path) -> dict[str, list[str]]:
    roster: dict[str, list[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            names: list[str] = []
            moderator = row.get("programme_moderator", "") or ""
            match = re.match(r"\s*([^,(]+?)(?:\s*\(|,|$)", moderator)
            if match and len(match.group(1).split()) >= 2:
                names.append(match.group(1).strip())
            panelists = row.get("programme_panelists", "") or ""
            for match in re.finditer(r"([^,()]+?)\s*\(([^)]*)\)", panelists):
                name = match.group(1).strip()
                if len(name.split()) >= 2 and name not in names:
                    names.append(name)
            roster[row["episode_id"]] = names
    return roster


def _duration_by_speaker(diarization: dict[str, Any]) -> Counter[str]:
    duration: Counter[str] = Counter()
    for cue in diarization.get("cues", []):
        try:
            duration[str(cue["speaker"])] += max(
                0.0, float(cue["end_seconds"]) - float(cue["start_seconds"])
            )
        except (KeyError, TypeError, ValueError):
            continue
    return duration


def _duplicate_groups(metadata_path: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            group = set(json.loads(row.get("duplicate_episode_ids", "[]")))
            if not group:
                group = {row["episode_id"]}
            for episode_id in group:
                groups[episode_id] = group
    return groups


def _speech_targets(
    root: Path,
    roster: dict[str, list[str]],
    names_by_episode: dict[str, dict[str, str]],
    minimum_seconds: float,
) -> list[tuple[str, str, float]]:
    targets: list[tuple[str, str, float]] = []
    for path in sorted((root / "data" / "diarization" / "2026").glob("*.json")):
        episode_id = path.stem
        diarization = _load_json(path)
        durations = _duration_by_speaker(diarization)
        known = names_by_episode.get(episode_id, {})
        if not roster.get(episode_id):
            continue
        for speaker, seconds in durations.items():
            if speaker not in known and seconds >= minimum_seconds:
                targets.append((episode_id, speaker, seconds))
    return sorted(targets, key=lambda item: item[2], reverse=True)


def _reference_index(
    root: Path,
    names_by_episode: dict[str, dict[str, str]],
    duplicate_groups: dict[str, set[str]],
) -> dict[str, list[tuple[str, str, float]]]:
    references: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for episode_id, speakers in names_by_episode.items():
        path = _diarization_path(root, episode_id)
        if not path.is_file():
            continue
        durations = _duration_by_speaker(_load_json(path))
        for speaker, name in speakers.items():
            seconds = durations.get(speaker, 0.0)
            if seconds >= 60:
                references[name].append((episode_id, speaker, seconds))
    for name in references:
        references[name].sort(key=lambda item: item[2], reverse=True)
    return references


def _candidate_references_for_names(
    episode_id: str,
    target_names: list[str],
    references: dict[str, list[tuple[str, str, float]]],
    duplicate_groups: dict[str, set[str]],
    max_names: int,
    max_recordings_per_name: int,
) -> dict[str, list[tuple[str, str, float]]]:
    target_duplicate_group = set(duplicate_groups.get(episode_id, {episode_id}))
    selected: dict[str, list[tuple[str, str, float]]] = {}
    for name in target_names:
        blocked = set(target_duplicate_group)
        name_references = []
        for reference in references.get(name, []):
            reference_id = reference[0]
            duplicate_group = duplicate_groups.get(reference_id, {reference_id})
            if duplicate_group & blocked:
                continue
            name_references.append(reference)
            blocked.update(duplicate_group)
            if len(name_references) >= max_recordings_per_name:
                break
        if name_references:
            selected[name] = name_references
        if len(selected) >= max_names:
            break
    return selected


def _load_or_download_audio(
    client: httpx.Client,
    root: Path,
    episode_id: str,
) -> Path:
    episode = _audio_episode(root, episode_id)
    path = audio_path(root / "data" / "audio-cache", episode)
    if path.is_file() and (
        episode.audio_bytes is None or path.stat().st_size == episode.audio_bytes
    ):
        return path
    audio = download_episode(client, episode, root / "data" / "audio-cache")
    return audio.path


def _read_audio_window(path: Path, start: float, duration: float) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode cached MP3 audio")
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(path),
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1000:])
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size < SAMPLE_RATE:
        raise RuntimeError(f"audio window at {start:.1f}s is shorter than one second")
    return samples


class WavLMEmbedder:
    def __init__(self, cache_dir: Path):
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
        except ImportError as error:
            raise RuntimeError(
                "Install the optional speaker-match dependencies from pyproject.toml"
            ) from error
        self.torch = torch
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        self.model = AutoModelForAudioXVector.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        self.model.eval()

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        inputs = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with self.torch.inference_mode():
            output = self.model(**inputs)
        vector = output.embeddings[0].detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm == 0:
            raise RuntimeError("speaker model returned an invalid embedding")
        return vector / norm


def _embedding_cache_path(cache_dir: Path, episode_id: str, speaker: str) -> Path:
    token = hashlib.sha256(f"{episode_id}/{speaker}/{MODEL_ID}".encode()).hexdigest()[:24]
    return cache_dir / f"{token}.npy"


def _speaker_embedding(
    embedder: WavLMEmbedder,
    audio_path_value: Path,
    diarization: dict[str, Any],
    speaker: str,
    cache_dir: Path,
) -> tuple[np.ndarray, int]:
    episode_id = str(diarization["episode_id"])
    cache_path = _embedding_cache_path(cache_dir, episode_id, speaker)
    cues = [cue for cue in diarization.get("cues", []) if cue.get("speaker") == speaker]
    selected = select_speaker_cues(cues)
    if not selected:
        raise ValueError(f"no clean speech windows for {episode_id} {speaker}")
    if cache_path.is_file():
        return np.load(cache_path), len(selected)
    vectors = []
    for cue in selected:
        start = float(cue["start_seconds"])
        duration = min(8.0, float(cue["end_seconds"]) - start)
        vectors.append(embedder.embed(_read_audio_window(audio_path_value, start, duration)))
    vector = np.mean(vectors, axis=0)
    vector /= np.linalg.norm(vector)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vector)
    return vector, len(vectors)


def _target_episode_ids(
    explicit: list[str],
    targets: list[tuple[str, str, float]],
    roster: dict[str, list[str]],
    references: dict[str, list[tuple[str, str, float]]],
    duplicate_groups: dict[str, set[str]],
    limit: int,
    max_reference_recordings: int,
    all_eligible: bool,
) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))
    weights: Counter[str] = Counter()
    for episode_id, _speaker, seconds in targets:
        candidates = _candidate_references_for_names(
            episode_id,
            roster.get(episode_id, []),
            references,
            duplicate_groups,
            max_names=99,
            max_recordings_per_name=max_reference_recordings,
        )
        if candidates:
            weights[episode_id] += min(seconds, 1800) * len(candidates)
    ranked = weights.most_common()
    if all_eligible:
        return [episode_id for episode_id, _ in ranked]
    return [episode_id for episode_id, _ in ranked[:limit]]


def run(
    root: Path,
    episode_ids: list[str],
    *,
    min_speech_seconds: float = 600,
    max_target_speakers: int = 3,
    max_reference_names: int = 8,
    max_reference_recordings: int = 3,
    all_eligible: bool = False,
) -> Path:
    if max_reference_recordings < 1:
        raise ValueError("max_reference_recordings must be at least one")
    metadata_path = root / "data" / "topic-analysis" / "results" / "talk-metadata" / "talks.csv"
    roster = _programme_roster(metadata_path)
    names_by_episode = _speaker_names(root)
    reference_names_by_episode = _speaker_names(root, include_voice_matches=False)
    duplicate_groups = _duplicate_groups(metadata_path)
    targets = _speech_targets(root, roster, names_by_episode, min_speech_seconds)
    references = _reference_index(root, reference_names_by_episode, duplicate_groups)
    selected_episodes = _target_episode_ids(
        episode_ids,
        targets,
        roster,
        references,
        duplicate_groups,
        limit=1,
        max_reference_recordings=max_reference_recordings,
        all_eligible=all_eligible,
    )
    if not selected_episodes:
        raise RuntimeError("No long unnamed speakers have known programme participants to compare")

    work: dict[str, dict[str, Any]] = {}
    for episode_id in selected_episodes:
        target_path = _diarization_path(root, episode_id)
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        diarization = _load_json(target_path)
        durations = _duration_by_speaker(diarization)
        unknown = [
            (speaker, seconds)
            for speaker, seconds in durations.items()
            if speaker not in names_by_episode.get(episode_id, {})
            and seconds >= min_speech_seconds
        ]
        unknown.sort(key=lambda item: item[1], reverse=True)
        if not unknown:
            continue
        candidate_references = _candidate_references_for_names(
            episode_id,
            roster.get(episode_id, []),
            references,
            duplicate_groups,
            max_names=max_reference_names,
            max_recordings_per_name=max_reference_recordings,
        )
        if not candidate_references:
            continue
        work[episode_id] = {
            "diarization": diarization,
            "unknown": unknown[:max_target_speakers],
            "references": candidate_references,
        }
    if not work:
        raise RuntimeError("Selected discussions have no eligible known names to compare")

    cache_dir = root / "data" / "diarization" / "speaker-voice-cache" / "wavlm-base-plus-sv"
    model_cache = root / "data" / "diarization" / "model-cache"
    embedder = WavLMEmbedder(model_cache)
    output_rows: list[dict[str, Any]] = []
    timeout = httpx.Timeout(connect=30, read=120, write=30, pool=30)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for episode_id, entry in work.items():
            print(f"Target recording {episode_id}: {len(entry['unknown'])} unnamed speakers")
            target_audio = _load_or_download_audio(client, root, episode_id)
            target_vectors = {}
            sample_counts = {}
            for speaker, seconds in entry["unknown"]:
                target_vectors[speaker], sample_counts[speaker] = _speaker_embedding(
                    embedder,
                    target_audio,
                    entry["diarization"],
                    speaker,
                    cache_dir,
                )
                print(f"  embedded {speaker} ({seconds / 60:.1f} min)")

            reference_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
            reference_details: dict[str, list[str]] = defaultdict(list)
            for name, references_for_name in entry["references"].items():
                for reference_id, reference_speaker, _seconds in references_for_name:
                    reference_diarization = _load_json(_diarization_path(root, reference_id))
                    reference_audio = _load_or_download_audio(client, root, reference_id)
                    vector, sample_count = _speaker_embedding(
                        embedder,
                        reference_audio,
                        reference_diarization,
                        reference_speaker,
                        cache_dir,
                    )
                    reference_vectors[name].append(vector)
                    reference_details[name].append(
                        f"{reference_id}/{reference_speaker}:windows={sample_count}"
                    )
                    print(f"  reference {name}: {reference_id}/{reference_speaker}")

            title = _load_json(_transcript_path(root, episode_id))["episode"]["title"]
            for speaker, seconds in entry["unknown"]:
                ranked = rank_named_candidates(target_vectors[speaker], reference_vectors)
                for rank, candidate in enumerate(ranked[:5], start=1):
                    output_rows.append(
                        {
                            "episode_id": episode_id,
                            "episode_title": title,
                            "speaker_label": speaker,
                            "speech_minutes": round(seconds / 60, 1),
                            "rank": rank,
                            "candidate_name": candidate["name"],
                            "cosine_similarity": round(candidate["similarity"], 4),
                            "reference_recordings": ";".join(reference_details[candidate["name"]]),
                            "reference_recording_count": len(reference_details[candidate["name"]]),
                            "target_samples": sample_counts[speaker],
                            "model": MODEL_ID,
                            "review_status": "manual_review_required",
                        }
                    )

    output_path = root / "data" / "diarization" / "speaker-voice-match-candidates.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} ranked candidates to {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--min-speech-minutes", type=float, default=10)
    parser.add_argument("--max-target-speakers", type=int)
    parser.add_argument("--max-reference-names", type=int, default=8)
    parser.add_argument("--max-reference-recordings", type=int, default=3)
    parser.add_argument(
        "--all-eligible",
        action="store_true",
        help="process every eligible discussion and all eligible unnamed speakers",
    )
    args = parser.parse_args()
    try:
        run(
            args.root.resolve(),
            args.episode_id,
            min_speech_seconds=args.min_speech_minutes * 60,
            max_target_speakers=args.max_target_speakers
            if args.max_target_speakers is not None
            else (999 if args.all_eligible else 3),
            max_reference_names=args.max_reference_names,
            max_reference_recordings=args.max_reference_recordings,
            all_eligible=args.all_eligible,
        )
    except (OSError, RuntimeError, ValueError, httpx.HTTPError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
