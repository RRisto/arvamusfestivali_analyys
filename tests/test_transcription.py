from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from arvamusfestivali_transcripts.transcription import run_transcriber


@pytest.fixture
def engine_payload() -> dict[str, object]:
    return {"transcript": "Tere tulemast.", "cues": []}


@pytest.fixture
def skill_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "2400217815.mp3"
    source.write_bytes(b"audio")
    script = tmp_path / "skill" / "scripts" / "transcribe.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('fixture')", encoding="utf-8")
    return source, tmp_path / "engine", script


@pytest.fixture
def available_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: f"/tools/{executable}")


def test_run_transcriber_requests_json_only(
    skill_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    engine_payload: dict[str, object],
    available_dependencies: None,
) -> None:
    """Omitting JSON-only output would mix human-readable engine output into the archive path."""
    source, output_dir, script = skill_paths
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command, cwd, check, capture_output, text):
        calls.append((command, cwd))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2400217815.json").write_text(
            json.dumps(engine_payload), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_transcriber(source, output_dir, script, uv_executable="uv")

    assert calls == [
        (
            [
                "uv",
                "run",
                "scripts/transcribe.py",
                str(source.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--formats",
                "json",
            ],
            script.parent.parent,
        )
    ]
    assert result == engine_payload


def test_run_transcriber_rejects_missing_audio(
    skill_paths: tuple[Path, Path, Path], available_dependencies: None
) -> None:
    """Launching the skill without source audio would create a misleading engine error."""
    source, output_dir, script = skill_paths
    source.unlink()

    with pytest.raises(FileNotFoundError, match="audio file"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_rejects_missing_script(
    skill_paths: tuple[Path, Path, Path], available_dependencies: None
) -> None:
    """A stale skill installation must fail before subprocess launch."""
    source, output_dir, script = skill_paths
    script.unlink()

    with pytest.raises(FileNotFoundError, match="transcription script"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_rejects_missing_uv(
    skill_paths: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable uv command cannot launch the external skill."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    source, output_dir, script = skill_paths

    with pytest.raises(RuntimeError, match="uv executable"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_rejects_missing_ffmpeg(
    skill_paths: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MP3 transcription depends on FFmpeg conversion support."""
    monkeypatch.setattr(
        "shutil.which", lambda executable: None if executable == "ffmpeg" else "/uv"
    )
    source, output_dir, script = skill_paths

    with pytest.raises(RuntimeError, match="ffmpeg"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_surfaces_last_stderr_line(
    skill_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    available_dependencies: None,
) -> None:
    """Discarding the final engine diagnostic makes a failed transcription hard to diagnose."""
    source, output_dir, script = skill_paths

    def fake_run(command, **_):
        raise subprocess.CalledProcessError(23, command, stderr="first problem\nactual problem\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"exit code 23.*actual problem"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_rejects_missing_new_json_output(
    skill_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    available_dependencies: None,
) -> None:
    """A successful engine exit without JSON cannot produce an archive payload."""
    source, output_dir, script = skill_paths
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
    )

    with pytest.raises(RuntimeError, match="exactly one new JSON"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_ignores_json_that_existed_before_invocation(
    skill_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    engine_payload: dict[str, object],
    available_dependencies: None,
) -> None:
    """Reading a previous JSON file would return stale transcription data after a no-op run."""
    source, output_dir, script = skill_paths
    output_dir.mkdir()
    (output_dir / "previous.json").write_text('{"stale": true}', encoding="utf-8")

    def fake_run(command, **_):
        (output_dir / "2400217815.json").write_text(json.dumps(engine_payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_transcriber(source, output_dir, script) == engine_payload


def test_run_transcriber_rejects_multiple_new_json_outputs(
    skill_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    available_dependencies: None,
) -> None:
    """Choosing an arbitrary output would attach a transcript to the wrong episode."""
    source, output_dir, script = skill_paths

    def fake_run(command, **_):
        output_dir.mkdir()
        (output_dir / "first.json").write_text("{}", encoding="utf-8")
        (output_dir / "second.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exactly one new JSON"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_rejects_invalid_json(
    skill_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    available_dependencies: None,
) -> None:
    """Malformed engine JSON must not reach archive conversion."""
    source, output_dir, script = skill_paths

    def fake_run(command, **_):
        output_dir.mkdir()
        (output_dir / "2400217815.json").write_text("not json", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        run_transcriber(source, output_dir, script)


def test_run_transcriber_rejects_non_object_json(
    skill_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    available_dependencies: None,
) -> None:
    """An array is valid JSON but cannot be the engine payload mapping."""
    source, output_dir, script = skill_paths

    def fake_run(command, **_):
        output_dir.mkdir()
        (output_dir / "2400217815.json").write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="must be an object"):
        run_transcriber(source, output_dir, script)
