from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import arvamusfestivali_transcripts.__main__ as cli
from arvamusfestivali_transcripts.pipeline import RunSummary
from arvamusfestivali_transcripts.schema import CatalogSnapshot


def test_run_command_accepts_year_and_acceptance_limit() -> None:
    args = cli.build_parser().parse_args(
        [
            "run",
            "--year",
            "2026",
            "--limit",
            "1",
            "--transcriber-script",
            "C:/skills/transcribe.py",
        ]
    )

    assert args.command == "run"
    assert args.year == 2026
    assert args.limit == 1


def test_main_returns_one_when_any_episode_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_year", lambda **kwargs: RunSummary(selected=1, failed=1))
    monkeypatch.setattr(cli, "_validate_transcription_configuration", lambda **kwargs: None)

    code = cli.main(
        ["run", "--year", "2026", "--transcriber-script", "C:/skills/transcribe.py"]
    )

    assert code == 1


def test_main_returns_two_when_transcriber_script_is_missing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing_script = tmp_path / "missing.py"

    assert (
        cli.main(
            ["run", "--year", "2026", "--transcriber-script", str(missing_script)]
        )
        == 2
    )
    assert "transcription script does not exist" in capsys.readouterr().err


def test_main_returns_two_when_uv_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = tmp_path / "transcribe.py"
    script.touch()
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)

    assert (
        cli.main(
            [
                "run",
                "--year",
                "2026",
                "--transcriber-script",
                str(script),
                "--uv-executable",
                "missing-uv",
            ]
        )
        == 2
    )
    assert "uv executable is unavailable: missing-uv" in capsys.readouterr().err


def test_main_returns_two_when_ffmpeg_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = tmp_path / "transcribe.py"
    script.touch()
    uv_executable = tmp_path / "uv.exe"
    uv_executable.touch()
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)

    assert (
        cli.main(
            [
                "run",
                "--year",
                "2026",
                "--transcriber-script",
                str(script),
                "--uv-executable",
                str(uv_executable),
            ]
        )
        == 2
    )
    assert "ffmpeg executable is required" in capsys.readouterr().err


def test_main_returns_two_for_global_discovery_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fetch(**_: object) -> RunSummary:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(cli, "fetch_year", fetch)

    assert cli.main(["fetch", "--year", "2026"]) == 2
    assert "offline" in capsys.readouterr().err


def test_main_returns_two_for_malformed_catalog_snapshot(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    snapshot = tmp_path / "malformed.json"
    snapshot.write_text("{}", encoding="utf-8")

    assert (
        cli.main(["fetch", "--year", "2026", "--catalog-snapshot", str(snapshot)]) == 2
    )
    assert "error:" in capsys.readouterr().err


def test_main_dispatches_fetch_and_prints_the_complete_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    received: dict[str, object] = {}

    def fetch(**kwargs: object) -> RunSummary:
        received.update(kwargs)
        return RunSummary(
            selected=1,
            skipped=2,
            completed=1,
            downloaded_bytes=86222137,
            deleted_cache=0,
        )

    monkeypatch.setattr(cli, "fetch_year", fetch)

    assert cli.main(["fetch", "--year", "2026", "--root", str(tmp_path), "--dry-run"]) == 0
    assert received["year"] == 2026
    assert received["paths"].root == tmp_path
    assert capsys.readouterr().out == (
        "selected=1 skipped=2 completed=1 failed=0 downloaded_bytes=86222137 deleted_cache=0\n"
    )


def test_main_dispatches_transcribe_with_validated_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: dict[str, object] = {}
    script = tmp_path / "transcribe.py"
    script.touch()
    uv_executable = tmp_path / "uv.exe"
    uv_executable.touch()

    def transcribe(**kwargs: object) -> RunSummary:
        received.update(kwargs)
        return RunSummary(selected=1, completed=1)

    monkeypatch.setattr(cli, "transcribe_year", transcribe)
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda executable: "/ffmpeg" if executable == "ffmpeg" else None,
    )

    assert (
        cli.main(
            [
                "transcribe",
                "--year",
                "2026",
                "--root",
                str(tmp_path),
                "--transcriber-script",
                str(script),
                "--uv-executable",
                str(uv_executable),
            ]
        )
        == 0
    )
    assert received["transcriber_script"] == script
    assert received["paths"].root == tmp_path


def test_main_forwards_frozen_snapshot_to_run_without_cli_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: dict[str, object] = {}
    script = tmp_path / "transcribe.py"
    script.touch()
    uv_executable = tmp_path / "uv.exe"
    uv_executable.touch()
    snapshot = CatalogSnapshot(
        apple_collection_id=1477431807,
        apple_page_url="https://podcasts.apple.com/id1477431807",
        feed_url="https://example.test/feed.xml",
        resolved_at=datetime(2026, 9, 19, tzinfo=UTC),
        year=2026,
        episodes=(),
    )
    snapshot_path = tmp_path / "frozen-2026.json"
    snapshot_path.write_text(snapshot.to_json(), encoding="utf-8")

    def run(**kwargs: object) -> RunSummary:
        received.update(kwargs)
        return RunSummary()

    monkeypatch.setattr(cli, "run_year", run)
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda executable: "/ffmpeg" if executable == "ffmpeg" else None,
    )

    assert (
        cli.main(
            [
                "run",
                "--year",
                "2026",
                "--catalog-snapshot",
                str(snapshot_path),
                "--transcriber-script",
                str(script),
                "--uv-executable",
                str(uv_executable),
            ]
        )
        == 0
    )
    assert received["catalog_snapshot"] == snapshot_path


@pytest.mark.parametrize("argument", ["0", "-1"])
def test_main_rejects_nonpositive_limits(argument: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["fetch", "--year", "2026", "--limit", argument]) == 2
    assert "limit must be positive" in capsys.readouterr().err


@pytest.mark.parametrize("episode_id", ["abc", "12-34", "../12"])
def test_main_rejects_non_numeric_episode_ids(
    episode_id: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["fetch", "--year", "2026", "--episode-id", episode_id]) == 2
    assert "episode ID must be numeric" in capsys.readouterr().err


def test_main_rejects_year_outside_supported_range(capsys: pytest.CaptureFixture[str]) -> None:
    next_year = datetime.now(UTC).year + 1

    assert cli.main(["fetch", "--year", str(next_year)]) == 2
    assert "year must be between 2019 and" in capsys.readouterr().err


def test_run_rejects_a_frozen_snapshot_for_a_different_year(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    snapshot = CatalogSnapshot(
        apple_collection_id=1477431807,
        apple_page_url="https://podcasts.apple.com/id1477431807",
        feed_url="https://example.test/feed.xml",
        resolved_at=datetime(2026, 9, 19, tzinfo=UTC),
        year=2025,
        episodes=(),
    )
    path = tmp_path / "frozen-2025.json"
    path.write_text(snapshot.to_json(), encoding="utf-8")

    assert (
        cli.main(
            [
                "run",
                "--year",
                "2026",
                "--catalog-snapshot",
                str(path),
                "--transcriber-script",
                "C:/skills/transcribe.py",
            ]
        )
        == 2
    )
    assert "catalog snapshot year does not match requested year 2026" in capsys.readouterr().err
