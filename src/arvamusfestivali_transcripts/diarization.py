"""One-episode speaker diarization pilot and timestamp reconciliation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

MODEL_NAME = "pyannote/speaker-diarization-community-1"


def assign_speakers(
    cues: Sequence[Mapping[str, Any]], turns: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Attach the speaker with the greatest temporal overlap to each transcript cue."""
    assigned: list[dict[str, Any]] = []
    for cue in cues:
        start = float(cue["start_seconds"])
        end = float(cue["end_seconds"])
        overlap_by_speaker: dict[str, float] = {}
        for turn in turns:
            overlap = max(
                0.0,
                min(end, float(turn["end_seconds"]))
                - max(start, float(turn["start_seconds"])),
            )
            if overlap:
                speaker = str(turn["speaker"])
                overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

        speaker = None
        speaker_overlap = 0.0
        if overlap_by_speaker:
            speaker, speaker_overlap = max(
                overlap_by_speaker.items(), key=lambda item: (item[1], item[0])
            )
        duration = max(0.0, end - start)
        item = dict(cue)
        item.update(
            {
                "speaker": speaker,
                "speaker_overlap_seconds": round(speaker_overlap, 3),
                "speaker_share": round(speaker_overlap / duration, 4) if duration else 0.0,
            }
        )
        assigned.append(item)
    return assigned


def download_audio(
    client: httpx.Client,
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str,
) -> Path:
    """Download audio and verify it against provenance retained in the transcript archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        _validate_audio(partial, expected_bytes, expected_sha256)
        partial.replace(destination)
        return destination
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _validate_audio(path: Path, expected_bytes: int | None, expected_sha256: str) -> None:
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(
            f"audio size mismatch: expected {expected_bytes} bytes, got {actual_bytes} bytes"
        )
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"audio SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )


def run_pyannote(
    audio_path: Path,
    token: str,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run the Community-1 model and return exclusive, anonymous speaker turns."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(MODEL_NAME, token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    options = {
        key: value
        for key, value in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if value is not None
    }
    result = pipeline(str(audio_path), **options)
    turns = [
        {
            "start_seconds": round(float(segment.start), 3),
            "end_seconds": round(float(segment.end), 3),
            "speaker": str(speaker),
        }
        for segment, speaker in result.exclusive_speaker_diarization
    ]
    return turns, device


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("speaker count must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diarize one archived festival episode")
    parser.add_argument("transcript", type=Path, help="canonical transcript JSON")
    parser.add_argument("--audio", type=Path, help="local source audio; otherwise download it")
    parser.add_argument("--output", type=Path, help="output JSON path")
    parser.add_argument("--num-speakers", type=_positive_integer)
    parser.add_argument("--min-speakers", type=_positive_integer)
    parser.add_argument("--max-speakers", type=_positive_integer)
    return parser


def _default_output(transcript: Path, episode: Mapping[str, Any]) -> Path:
    year = str(episode["published_at"])[:4]
    if transcript.parent.name == year and transcript.parent.parent.name == "transcripts":
        data_root = transcript.parent.parent.parent
        return data_root / "diarization" / year / f"{episode['id']}.json"
    return transcript.with_suffix(".diarization.json")


def _load_archive(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("transcript archive must contain a JSON object")
    episode = payload.get("episode")
    transcription = payload.get("transcription")
    if not isinstance(episode, dict) or not isinstance(transcription, dict):
        raise ValueError("transcript archive is missing episode or transcription metadata")
    if not isinstance(transcription.get("cues"), list):
        raise ValueError("transcript archive is missing timestamped cues")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        archive = _load_archive(args.transcript)
        episode = archive["episode"]
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for the gated pyannote model")
        output = args.output or _default_output(args.transcript, episode)

        with tempfile.TemporaryDirectory(prefix="arvamusfestival-diarization-") as temporary:
            if args.audio is not None:
                audio_path = args.audio
                _validate_audio(
                    audio_path,
                    episode.get("audio_bytes"),
                    str(episode["audio_sha256"]),
                )
            else:
                audio_path = Path(temporary) / f"{episode['id']}.mp3"
                with httpx.Client(follow_redirects=True, timeout=120) as client:
                    download_audio(
                        client,
                        str(episode["audio_url"]),
                        audio_path,
                        expected_bytes=episode.get("audio_bytes"),
                        expected_sha256=str(episode["audio_sha256"]),
                    )

            turns, device = run_pyannote(
                audio_path,
                token,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )

        payload = {
            "schema_version": 1,
            "episode_id": str(episode["id"]),
            "model": MODEL_NAME,
            "device": device,
            "speakers": sorted({turn["speaker"] for turn in turns}),
            "turns": turns,
            "cues": assign_speakers(archive["transcription"]["cues"], turns),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {output} ({len(payload['speakers'])} speakers, {len(turns)} turns)")
        return 0
    except (httpx.HTTPError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
