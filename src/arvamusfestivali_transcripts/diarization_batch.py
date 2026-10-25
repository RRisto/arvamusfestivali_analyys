"""Resumable batch speaker diarization and review-page generation."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .diarization import MODEL_NAME, _load_archive, assign_speakers, download_audio
from .diarization_review import build_review_page


@dataclass(frozen=True)
class BatchJob:
    episode_id: str
    year: str
    title: str
    transcript: Path
    output: Path
    review: Path


@dataclass(frozen=True)
class BatchPlan:
    total: int
    skipped: int
    jobs: list[BatchJob]
    remaining_after_limit: int


@dataclass(frozen=True)
class BatchSummary:
    completed: list[str]
    failed: dict[str, str]


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _valid_output(path: Path, episode_id: str) -> bool:
    try:
        payload = _read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        str(payload.get("episode_id")) == episode_id
        and payload.get("schema_version") == 1
        and isinstance(payload.get("turns"), list)
        and isinstance(payload.get("cues"), list)
        and isinstance(payload.get("speakers"), list)
    )


def plan_batch(
    project_root: Path,
    *,
    year: str | None = None,
    limit: int | None = None,
) -> BatchPlan:
    """Discover transcripts and return only unfinished or invalid jobs."""
    transcript_root = project_root / "data" / "transcripts"
    candidates = sorted(transcript_root.glob(f"{year or '*'}/*.json"))
    pending: list[BatchJob] = []
    skipped = 0
    for transcript in candidates:
        archive = _load_archive(transcript)
        episode = archive["episode"]
        episode_id = str(episode["id"])
        published_year = str(episode.get("published_at", transcript.parent.name))[:4]
        output = project_root / "data" / "diarization" / published_year / f"{episode_id}.json"
        if _valid_output(output, episode_id):
            skipped += 1
            continue
        pending.append(
            BatchJob(
                episode_id=episode_id,
                year=published_year,
                title=str(episode.get("title", episode_id)),
                transcript=transcript,
                output=output,
                review=project_root / "site" / "diarization" / f"{episode_id}.html",
            )
        )
    selected = pending if limit is None else pending[:limit]
    return BatchPlan(
        total=len(candidates),
        skipped=skipped,
        jobs=selected,
        remaining_after_limit=len(pending) - len(selected),
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class PyannoteBatchProcessor:
    """Load Community-1 once and reuse it for every episode."""

    def __init__(self, token: str, *, require_cuda: bool = True) -> None:
        import torch
        from pyannote.audio import Pipeline

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required; pass --allow-cpu to override")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = Pipeline.from_pretrained(MODEL_NAME, token=token)
        self.pipeline.to(torch.device(self.device))
        self.client = httpx.Client(follow_redirects=True, timeout=120)

    def close(self) -> None:
        self.client.close()

    def __call__(self, job: BatchJob) -> None:
        archive = _load_archive(job.transcript)
        episode = archive["episode"]
        with tempfile.TemporaryDirectory(prefix="arvamusfestival-diarization-") as temporary:
            audio_path = Path(temporary) / f"{job.episode_id}.mp3"
            download_audio(
                self.client,
                str(episode["audio_url"]),
                audio_path,
                expected_bytes=episode.get("audio_bytes"),
                expected_sha256=str(episode["audio_sha256"]),
            )
            result = self.pipeline(str(audio_path))
        turns = [
            {
                "start_seconds": round(float(segment.start), 3),
                "end_seconds": round(float(segment.end), 3),
                "speaker": str(speaker),
            }
            for segment, speaker in result.exclusive_speaker_diarization
        ]
        payload = {
            "schema_version": 1,
            "episode_id": job.episode_id,
            "model": MODEL_NAME,
            "device": self.device,
            "speakers": sorted({turn["speaker"] for turn in turns}),
            "turns": turns,
            "cues": assign_speakers(archive["transcription"]["cues"], turns),
        }
        _write_json_atomic(job.output, payload)


def _status_path(project_root: Path) -> Path:
    return project_root / "data" / "diarization" / "batch-status.json"


def _write_status(
    project_root: Path, plan: BatchPlan, completed: list[str], failed: Mapping[str, str]
) -> None:
    _write_json_atomic(
        _status_path(project_root),
        {
            "total_transcripts": plan.total,
            "already_complete": plan.skipped,
            "selected": len(plan.jobs),
            "remaining_after_limit": plan.remaining_after_limit,
            "completed": completed,
            "failed": dict(failed),
        },
    )


def run_batch(
    project_root: Path,
    *,
    year: str | None = None,
    limit: int | None = None,
    process: Callable[[BatchJob], None] | None = None,
    token: str | None = None,
    require_cuda: bool = True,
) -> BatchSummary:
    """Run pending jobs sequentially, checkpointing after every episode."""
    plan = plan_batch(project_root, year=year, limit=limit)
    owned_processor: PyannoteBatchProcessor | None = None
    if process is None and plan.jobs:
        if not token:
            raise RuntimeError("HF_TOKEN is required for the gated pyannote model")
        owned_processor = PyannoteBatchProcessor(token, require_cuda=require_cuda)
        process = owned_processor

    completed: list[str] = []
    failed: dict[str, str] = {}
    try:
        for position, job in enumerate(plan.jobs, start=1):
            print(f"[{position}/{len(plan.jobs)}] {job.episode_id} {job.title}", flush=True)
            try:
                assert process is not None
                process(job)
                if not _valid_output(job.output, job.episode_id):
                    raise ValueError("processor did not write a valid diarization output")
                build_review_page(job.transcript, job.output, job.review)
                completed.append(job.episode_id)
                print(f"completed {job.episode_id}", flush=True)
            except Exception as error:
                failed[job.episode_id] = str(error)
                print(f"failed {job.episode_id}: {error}", file=sys.stderr, flush=True)
            _write_status(project_root, plan, completed, failed)
            build_review_index(project_root, failures=failed)
    finally:
        if owned_processor is not None:
            owned_processor.close()
    _write_status(project_root, plan, completed, failed)
    build_review_index(project_root, failures=failed)
    required_stats_inputs = (
        project_root / "data/topic-analysis/results/talk-metadata/talks.parquet",
        project_root / "data/topic-analysis/results/jev-semantic-all/assignments.parquet",
        project_root / "data/topic-analysis/results/segment-dataset/all-segments.parquet",
    )
    if all(path.is_file() for path in required_stats_inputs):
        try:
            from .discussion_stats import build_discussion_stats_site

            pages, _ = build_discussion_stats_site(project_root)
            print(f"built {pages} discussion statistics pages", flush=True)
        except ModuleNotFoundError as error:
            if error.name not in {"pandas", "pyarrow"}:
                raise
            print("skipped statistics pages; install the topic-analysis dependencies", flush=True)
    return BatchSummary(completed=completed, failed=failed)


def build_review_index(
    project_root: Path, *, failures: Mapping[str, str] | None = None
) -> Path:
    """Write an index of all transcript episodes and their batch state."""
    failures = failures or {}
    rows: list[str] = []
    completed = 0
    transcripts = sorted((project_root / "data" / "transcripts").glob("*/*.json"))
    for transcript in transcripts:
        archive = _load_archive(transcript)
        episode = archive["episode"]
        episode_id = str(episode["id"])
        title = html.escape(str(episode.get("title", episode_id)))
        review = project_root / "site" / "diarization" / f"{episode_id}.html"
        if review.is_file():
            completed += 1
            state = '<span class="done">Completed</span>'
            stats_page = project_root / "site" / "diarization" / f"{episode_id}-stats.html"
            stats_link = (
                f' · <a href="{episode_id}-stats.html">Statistics</a>'
                if stats_page.is_file()
                else ""
            )
            title_markup = f'<a href="{episode_id}.html">{title}</a>{stats_link}'
        elif episode_id in failures:
            message = html.escape(str(failures[episode_id]))
            state = f'<span class="failed">Failed: {message}</span>'
            title_markup = title
        else:
            state = '<span class="waiting">Waiting</span>'
            title_markup = title
        rows.append(
            f"<li><span>{title_markup}</span><small>{episode_id}</small>{state}</li>"
        )
    output = project_root / "site" / "diarization" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speaker diarization review</title>
<style>
body{font:16px system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 18px;
color:#172033;background:#f5f7fb}
h1{margin-bottom:6px}p{color:#59657a}
ul{list-style:none;padding:0;display:grid;gap:8px}
li{display:grid;grid-template-columns:1fr 130px 260px;gap:14px;align-items:center;
background:#fff;padding:13px 16px;border-radius:9px;border:1px solid #dde3ed}
a{font-weight:700;color:#2458c6}small{color:#6a7486}.done{color:#087443}
.failed{color:#b42318}.waiting{color:#6a7486}
@media(max-width:700px){li{grid-template-columns:1fr}small{display:none}}
</style></head><body>
"""
        + f"<h1>Speaker diarization review</h1><p>{completed} / {len(transcripts)} completed</p>"
        + "<ul>"
        + "\n".join(rows)
        + "</ul></body></html>\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diarize all unfinished episodes")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    parser.add_argument("--year", help="only process one publication year")
    parser.add_argument("--limit", type=int, help="process at most this many pending episodes")
    parser.add_argument(
        "--dry-run", action="store_true", help="print selected episodes without processing"
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow the batch to run without CUDA (not recommended)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("error: --limit must be positive", file=sys.stderr)
        return 2
    try:
        plan = plan_batch(args.root.resolve(), year=args.year, limit=args.limit)
        print(
            f"found {plan.total} transcripts; {plan.skipped} complete; "
            f"{len(plan.jobs)} selected"
        )
        if args.dry_run:
            for job in plan.jobs:
                print(f"{job.episode_id} {job.title}")
            return 0
        summary = run_batch(
            args.root.resolve(),
            year=args.year,
            limit=args.limit,
            token=os.environ.get("HF_TOKEN"),
            require_cuda=not args.allow_cpu,
        )
        print(f"batch finished: {len(summary.completed)} completed, {len(summary.failed)} failed")
        return 1 if summary.failed else 0
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
