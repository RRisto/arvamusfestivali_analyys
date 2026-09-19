"""Adapter for the external Estonian transcription skill."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def run_transcriber(
    audio_path: Path,
    output_dir: Path,
    script_path: Path,
    uv_executable: str = "uv",
) -> dict[str, Any]:
    """Run the transcription skill and return its one newly-created JSON object."""
    _require_file(audio_path, "audio file")
    _require_file(script_path, "transcription script")
    _require_uv(uv_executable)
    if audio_path.suffix.lower() == ".mp3" and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg executable is required for MP3 transcription")

    resolved_audio = audio_path.resolve()
    resolved_output = output_dir.resolve()
    skill_root = script_path.parent.parent
    previous_outputs = _json_outputs(resolved_output)
    command = [
        uv_executable,
        "run",
        "scripts/transcribe.py",
        str(resolved_audio),
        "--output-dir",
        str(resolved_output),
        "--formats",
        "json",
    ]
    try:
        subprocess.run(
            command,
            cwd=skill_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(_subprocess_error(error)) from error

    new_outputs = _json_outputs(resolved_output) - previous_outputs
    if len(new_outputs) != 1:
        raise RuntimeError(
            f"expected exactly one new JSON engine output, found {len(new_outputs)}"
        )
    return _load_json_object(new_outputs.pop())


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def _require_uv(uv_executable: str) -> None:
    configured_path = Path(uv_executable)
    if configured_path.is_absolute() and configured_path.is_file():
        return
    if shutil.which(uv_executable) is None:
        raise RuntimeError(f"uv executable is unavailable: {uv_executable}")


def _json_outputs(output_dir: Path) -> set[Path]:
    if not output_dir.is_dir():
        return set()
    return {path.resolve() for path in output_dir.glob("*.json") if path.is_file()}


def _subprocess_error(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr if isinstance(error.stderr, str) else ""
    last_line = next((line for line in reversed(stderr.splitlines()) if line.strip()), None)
    detail = f": {last_line}" if last_line else ""
    return f"transcription skill failed with exit code {error.returncode}{detail}"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"engine output contains invalid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"engine JSON output must be an object: {path}")
    return dict(payload)
