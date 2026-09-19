"""Command-line entry point for the Arvamusfestival transcript archive."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import httpx

from .pipeline import (
    PipelineDependencies,
    PipelinePaths,
    RunSummary,
    fetch_year,
    run_year,
    transcribe_year,
)
from .schema import CatalogSnapshot
from .transcription import run_transcriber


def _year(value: str) -> int:
    year = int(value)
    current_year = datetime.now(UTC).year
    if not 2019 <= year <= current_year:
        raise argparse.ArgumentTypeError(f"year must be between 2019 and {current_year}")
    return year


def _positive_limit(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return limit


def _episode_id(value: str) -> str:
    if not value.isascii() or not value.isdigit():
        raise argparse.ArgumentTypeError("episode ID must be numeric")
    return value


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--year", required=True, type=_year, help="publication year")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="archive root directory")
    parser.add_argument("--episode-id", type=_episode_id, help="one numeric SoundCloud episode ID")
    parser.add_argument("--limit", type=_positive_limit, help="maximum pending episodes to process")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing transcript archive"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="select episodes without changing files"
    )
    parser.add_argument(
        "--catalog-snapshot",
        type=Path,
        help="frozen catalog JSON snapshot; avoids Apple and RSS lookup when supplied",
    )
    parser.add_argument(
        "--uv-executable",
        default="uv",
        help="uv executable used by the external transcriber (default: uv)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the parser used by both the console script and tests."""
    parser = argparse.ArgumentParser(prog="arvamusfestivali-transcripts")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download audio into the verified cache")
    _add_common_arguments(fetch)

    transcribe = commands.add_parser("transcribe", help="transcribe verified cached audio")
    _add_common_arguments(transcribe)
    transcribe.add_argument(
        "--transcriber-script",
        required=True,
        type=Path,
        help="path to the Estonian transcription skill script",
    )

    run = commands.add_parser("run", help="download, transcribe, and archive episodes")
    _add_common_arguments(run)
    run.add_argument(
        "--transcriber-script",
        required=True,
        type=Path,
        help="path to the Estonian transcription skill script",
    )
    return parser


def _validate_snapshot_year(snapshot_path: Path, year: int) -> None:
    snapshot = CatalogSnapshot.from_json(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.year != year:
        raise ValueError(f"catalog snapshot year does not match requested year {year}")


def _validate_transcription_configuration(*, script: Path, uv_executable: str) -> None:
    """Fail before the workflow can classify missing global dependencies as episode errors."""
    if not script.is_file():
        raise FileNotFoundError(f"transcription script does not exist: {script}")
    uv_path = Path(uv_executable)
    if not (uv_path.is_absolute() and uv_path.is_file()) and shutil.which(uv_executable) is None:
        raise RuntimeError(f"uv executable is unavailable: {uv_executable}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg executable is required for MP3 transcription")


def _dependencies(uv_executable: str) -> PipelineDependencies:
    return PipelineDependencies(transcriber=partial(run_transcriber, uv_executable=uv_executable))


def _summary_line(summary: RunSummary) -> str:
    return (
        f"selected={summary.selected} skipped={summary.skipped} completed={summary.completed} "
        f"failed={summary.failed} downloaded_bytes={summary.downloaded_bytes} "
        f"deleted_cache={summary.deleted_cache}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one archive operation and return its process exit status."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        if args.catalog_snapshot is not None:
            _validate_snapshot_year(args.catalog_snapshot, args.year)
        if args.command != "fetch":
            _validate_transcription_configuration(
                script=args.transcriber_script,
                uv_executable=args.uv_executable,
            )

        common = {
            "year": args.year,
            "paths": PipelinePaths.from_root(args.root),
            "catalog_snapshot": args.catalog_snapshot,
            "episode_id": args.episode_id,
            "limit": args.limit,
            "force": args.force,
            "dry_run": args.dry_run,
        }
        if args.command == "fetch":
            summary = fetch_year(**common)
        else:
            workflow = run_year if args.command == "run" else transcribe_year
            summary = workflow(
                **common,
                transcriber_script=args.transcriber_script,
                dependencies=_dependencies(args.uv_executable),
            )
    except (httpx.HTTPError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(_summary_line(summary))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
