"""Reuse saved speaker turns when retranscription leaves source audio unchanged."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .archive import validate_archive
from .diarization import assign_speakers
from .retranscription import sha256_file


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reassign_saved_turns(
    old_transcript_path: Path,
    new_transcript_path: Path,
    diarization_path: Path,
    output_path: Path,
) -> Path:
    """Assign new cues to saved turns after proving that the source audio is identical."""
    old = _read_object(old_transcript_path)
    new = _read_object(new_transcript_path)
    saved = _read_object(diarization_path)
    old_year = int(str(old["episode"]["published_at"])[:4])
    new_year = int(str(new["episode"]["published_at"])[:4])
    validate_archive(old, old_year)
    validate_archive(new, new_year)
    old_episode = old["episode"]
    new_episode = new["episode"]
    episode_id = str(new_episode["id"])
    if str(old_episode["id"]) != episode_id or str(saved.get("episode_id")) != episode_id:
        raise ValueError("episode identity differs across transcript and diarization inputs")
    if old_episode["audio_sha256"] != new_episode["audio_sha256"]:
        raise ValueError("audio SHA-256 changed; saved speaker turns cannot be reused safely")
    turns = saved.get("turns")
    if not isinstance(turns, list):
        raise ValueError("saved diarization has no speaker turns")
    for turn in turns:
        if not isinstance(turn, dict) or not {
            "start_seconds",
            "end_seconds",
            "speaker",
        }.issubset(turn):
            raise ValueError("saved diarization contains an invalid speaker turn")

    payload = {
        "schema_version": 2,
        "episode_id": episode_id,
        "audio_sha256": new_episode["audio_sha256"],
        "transcript_sha256": sha256_file(new_transcript_path),
        "model": saved.get("model"),
        "device": saved.get("device"),
        "speakers": sorted({str(turn["speaker"]) for turn in turns}),
        "turns": turns,
        "cues": assign_speakers(new["transcription"]["cues"], turns),
    }
    _atomic_json(output_path, payload)
    return output_path


def reassign_tree(
    old_transcript_dir: Path,
    new_transcript_dir: Path,
    saved_diarization_dir: Path,
    output_dir: Path,
    *,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    """Reassign every new transcript, continuing after per-episode failures."""
    from .diarization_review import build_review_page

    new_paths = sorted(new_transcript_dir.glob("*.json"), key=lambda path: path.name)
    failures: dict[str, str] = {}
    files: dict[str, dict[str, str]] = {}
    completed = 0
    for new_path in new_paths:
        episode_id = new_path.stem
        output = output_dir / new_path.name
        try:
            reassign_saved_turns(
                old_transcript_dir / new_path.name,
                new_path,
                saved_diarization_dir / new_path.name,
                output,
            )
            if review_dir is not None:
                build_review_page(new_path, output, review_dir / f"{episode_id}.html")
            files[new_path.name] = {
                "sha256": sha256_file(output),
                "transcript_sha256": sha256_file(new_path),
            }
            completed += 1
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures[episode_id] = str(error)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "counts": {
            "selected": len(new_paths),
            "completed": completed,
            "failed": len(failures),
        },
        "failures": failures,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reassign new transcript cues to unchanged saved pyannote turns"
    )
    parser.add_argument("--old-transcripts", type=Path, required=True)
    parser.add_argument("--new-transcripts", type=Path, required=True)
    parser.add_argument("--saved-diarization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = reassign_tree(
        args.old_transcripts,
        args.new_transcripts,
        args.saved_diarization,
        args.output,
        review_dir=args.review_output,
    )
    _atomic_json(args.manifest, manifest)
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 1 if manifest["counts"]["failed"] else 0
