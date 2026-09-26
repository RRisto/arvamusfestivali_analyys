"""Resumable orchestration for source audio and canonical transcripts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .archive import build_archive, validate_archive, write_archive
from .catalog import resolve_catalog
from .download import audio_path, download_episode
from .feed import fetch_feed, parse_feed, write_snapshot
from .schema import CatalogSnapshot, DownloadedAudio, Episode
from .state import EpisodeStatus, PipelineState
from .transcription import run_transcriber

DEFAULT_COLLECTION_ID = 1477431807
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    root: Path
    catalog: Path
    audio_cache: Path
    transcripts: Path
    engine_output: Path
    state_db: Path

    @classmethod
    def from_root(cls, root: Path) -> PipelinePaths:
        data = root / "data"
        return cls(
            root,
            data / "catalog",
            data / "audio-cache",
            data / "transcripts",
            data / "engine-output",
            data / "state" / "pipeline.sqlite3",
        )


@dataclass(slots=True)
class RunSummary:
    selected: int = 0
    skipped: int = 0
    completed: int = 0
    failed: int = 0
    downloaded_bytes: int = 0
    deleted_cache: int = 0


def _http_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=60)


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """Inject external I/O while retaining real archive, download and state logic."""

    client_factory: Callable[[], httpx.Client] = _http_client
    transcriber: Callable[[Path, Path, Path], dict[str, Any]] = run_transcriber
    now: Callable[[], datetime] = lambda: datetime.now(UTC)


def run_year(
    *,
    year: int,
    paths: PipelinePaths,
    transcriber_script: Path,
    dependencies: PipelineDependencies | None = None,
    collection_id: int = DEFAULT_COLLECTION_ID,
    catalog_snapshot: Path | None = None,
    episode_id: str | None = None,
    limit: int | None = None,
    parallelism: int = 1,
    force: bool = False,
    dry_run: bool = False,
) -> RunSummary:
    """Download and transcribe each pending episode before starting the next."""
    return _workflow(
        "run",
        year,
        paths,
        dependencies,
        collection_id,
        catalog_snapshot,
        episode_id,
        limit,
        parallelism,
        force,
        dry_run,
        transcriber_script,
    )


def fetch_year(
    *,
    year: int,
    paths: PipelinePaths,
    dependencies: PipelineDependencies | None = None,
    collection_id: int = DEFAULT_COLLECTION_ID,
    catalog_snapshot: Path | None = None,
    episode_id: str | None = None,
    limit: int | None = None,
    parallelism: int = 1,
    force: bool = False,
    dry_run: bool = False,
) -> RunSummary:
    """Cache pending audio; completed counts successful downloads, skipped includes cache hits."""
    return _workflow(
        "fetch",
        year,
        paths,
        dependencies,
        collection_id,
        catalog_snapshot,
        episode_id,
        limit,
        parallelism,
        force,
        dry_run,
        None,
    )


def transcribe_year(
    *,
    year: int,
    paths: PipelinePaths,
    transcriber_script: Path,
    dependencies: PipelineDependencies | None = None,
    collection_id: int = DEFAULT_COLLECTION_ID,
    catalog_snapshot: Path | None = None,
    episode_id: str | None = None,
    limit: int | None = None,
    parallelism: int = 1,
    force: bool = False,
    dry_run: bool = False,
) -> RunSummary:
    """Transcribe cached audio using a frozen or newest local catalog, without HTTP."""
    return _workflow(
        "transcribe",
        year,
        paths,
        dependencies,
        collection_id,
        catalog_snapshot,
        episode_id,
        limit,
        parallelism,
        force,
        dry_run,
        transcriber_script,
    )


def refresh_catalog(
    *,
    year: int,
    paths: PipelinePaths,
    collection_id: int = DEFAULT_COLLECTION_ID,
    dependencies: PipelineDependencies | None = None,
) -> CatalogSnapshot:
    """Resolve Apple and RSS, saving source snapshots for a nonempty publication year."""
    dependencies = dependencies or PipelineDependencies()
    with dependencies.client_factory() as client:
        catalog = resolve_catalog(client, collection_id)
        xml = fetch_feed(client, catalog.feed_url)
    snapshot = CatalogSnapshot(
        apple_collection_id=catalog.collection_id,
        apple_page_url=catalog.apple_page_url,
        feed_url=catalog.feed_url,
        resolved_at=dependencies.now(),
        year=year,
        episodes=parse_feed(xml, year),
    )
    _validate_snapshot(snapshot, year)
    write_snapshot(paths.root, snapshot, xml)
    return snapshot


def _load_snapshot(path: Path) -> CatalogSnapshot:
    return CatalogSnapshot.from_json(path.read_text(encoding="utf-8"))


def _newest_snapshot(paths: PipelinePaths, year: int) -> CatalogSnapshot:
    snapshots = [
        (snapshot.resolved_at, path.name, snapshot)
        for path in paths.catalog.glob("*.json")
        if (snapshot := _load_snapshot(path)).year == year
    ]
    if not snapshots:
        raise FileNotFoundError(f"no catalog snapshot for {year}; run catalog or fetch first")
    return max(snapshots, key=lambda item: (item[0], item[1]))[2]


def _validate_snapshot(snapshot: CatalogSnapshot, year: int) -> None:
    if snapshot.year != year or any(e.published_at.year != year for e in snapshot.episodes):
        raise ValueError(f"catalog snapshot year does not match requested year {year}")
    if not snapshot.episodes:
        raise ValueError(f"catalog contains no episodes for {year}")
    for episode in snapshot.episodes:
        if not episode.id.isascii() or not episode.id.isdigit():
            raise ValueError(f"invalid SoundCloud episode ID: {episode.id!r}")


def _archive_path(paths: PipelinePaths, episode: Episode) -> Path:
    return paths.transcripts / str(episode.published_at.year) / f"{episode.id}.json"


def _valid_archive(path: Path, episode: Episode, collection_id: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        validate_archive(payload, episode.published_at.year)
        return (
            payload["episode"]["id"] == episode.id
            and payload["episode"]["apple_collection_id"] == collection_id
        )
    except (ValueError, OSError):
        return False


def _valid_cache(path: Path, episode: Episode) -> bool:
    return _verified_audio(path, episode) is not None


def _cache_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _read_cache_metadata(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


def _write_cache_metadata(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verified_audio(path: Path, episode: Episode) -> DownloadedAudio | None:
    metadata = _read_cache_metadata(_cache_sidecar(path))
    if metadata is None or not path.is_file():
        return None
    try:
        size = path.stat().st_size
        if (
            size <= 0
            or (episode.audio_bytes is not None and size != episode.audio_bytes)
            or metadata.get("episode_id") != episode.id
            or metadata.get("audio_url") != episode.audio_url
            or metadata.get("byte_count") != size
        ):
            return None
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if metadata.get("sha256") != digest:
            return None
        return DownloadedAudio(path, size, digest)
    except OSError:
        return None


def _cached_audio(paths: PipelinePaths, episode: Episode) -> DownloadedAudio:
    path = audio_path(paths.audio_cache, episode)
    audio = _verified_audio(path, episode)
    if audio is None:
        raise ValueError(f"missing or unverified cached audio: {path}; run fetch or run to refresh")
    return audio


def _prepare_download(path: Path, episode: Episode) -> Path:
    # Unverified finals must not reach download_episode's size-only cache shortcut.
    if path.exists() or path.is_symlink():
        path.unlink()
    _cache_sidecar(path).unlink(missing_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    provenance = _cache_sidecar(partial)
    expected = {
        "episode_id": episode.id,
        "audio_url": episode.audio_url,
        "expected_bytes": episode.audio_bytes,
    }
    if _read_cache_metadata(provenance) != expected:
        partial.unlink(missing_ok=True)
        provenance.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_cache_metadata(provenance, expected)
    return provenance


def _workflow(
    mode: str,
    year: int,
    paths: PipelinePaths,
    dependencies: PipelineDependencies | None,
    collection_id: int,
    catalog_snapshot: Path | None,
    episode_id: str | None,
    limit: int | None,
    parallelism: int,
    force: bool,
    dry_run: bool,
    transcriber_script: Path | None,
) -> RunSummary:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    dependencies = dependencies or PipelineDependencies()
    if catalog_snapshot is not None:
        snapshot = _load_snapshot(catalog_snapshot)
    elif mode == "transcribe":
        snapshot = _newest_snapshot(paths, year)
    else:
        snapshot = refresh_catalog(
            year=year,
            paths=paths,
            collection_id=collection_id,
            dependencies=dependencies,
        )
    _validate_snapshot(snapshot, year)
    candidates = sorted(snapshot.episodes, key=lambda episode: (episode.published_at, episode.id))
    if episode_id is not None:
        candidates = [episode for episode in candidates if episode.id == episode_id]
        if not candidates:
            raise ValueError(f"episode {episode_id} is absent from the {year} catalog")
    summary = RunSummary()
    pending = []
    for episode in candidates:
        if (
            not force
            and _valid_archive(
                _archive_path(paths, episode),
                episode,
                snapshot.apple_collection_id,
            )
        ) or (mode == "fetch" and _valid_cache(audio_path(paths.audio_cache, episode), episode)):
            summary.skipped += 1
        else:
            pending.append(episode)
    selected = pending[:limit]
    summary.selected = len(selected)
    if dry_run:
        return summary
    paths.state_db.parent.mkdir(parents=True, exist_ok=True)
    with PipelineState(paths.state_db) as state:
        for episode in snapshot.episodes:
            state.record_discovered(episode.id, year)
    if not selected:
        return summary

    def process(episode: Episode) -> RunSummary:
        episode_summary = RunSummary()
        with PipelineState(paths.state_db) as state:
            client_context = (
                dependencies.client_factory() if mode != "transcribe" else nullcontext()
            )
            with client_context as client:
                _process_episode(
                    episode,
                    snapshot,
                    mode,
                    paths,
                    state,
                    client,
                    dependencies,
                    transcriber_script,
                    force,
                    episode_summary,
                )
        return episode_summary

    if parallelism == 1:
        results = map(process, selected)
        for result in results:
            _merge_summary(summary, result)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            for result in executor.map(process, selected):
                _merge_summary(summary, result)
    return summary


def _merge_summary(total: RunSummary, episode: RunSummary) -> None:
    total.completed += episode.completed
    total.failed += episode.failed
    total.downloaded_bytes += episode.downloaded_bytes
    total.deleted_cache += episode.deleted_cache


def _process_episode(
    episode: Episode,
    snapshot: CatalogSnapshot,
    mode: str,
    paths: PipelinePaths,
    state: PipelineState,
    client: httpx.Client | None,
    dependencies: PipelineDependencies,
    transcriber_script: Path | None,
    force: bool,
    summary: RunSummary,
) -> None:
    stage = "cache" if mode == "transcribe" else "download"
    try:
        if mode == "transcribe":
            audio = _cached_audio(paths, episode)
        else:
            cached = audio_path(paths.audio_cache, episode)
            audio = _verified_audio(cached, episode)
            if audio is None:
                state.mark_started(episode.id, EpisodeStatus.DOWNLOADING)
                provenance = _prepare_download(cached, episode)
                audio = download_episode(client, episode, paths.audio_cache)
                summary.downloaded_bytes += audio.byte_count
                if audio.byte_count <= 0:
                    raise ValueError(f"download produced invalid cached audio: {audio.path}")
                _write_cache_metadata(
                    _cache_sidecar(audio.path),
                    {
                        "episode_id": episode.id,
                        "audio_url": episode.audio_url,
                        "byte_count": audio.byte_count,
                        "sha256": audio.sha256,
                    },
                )
                provenance.unlink()
        state.mark_started(episode.id, EpisodeStatus.DOWNLOADED)
        if mode == "fetch":
            summary.completed += 1
            return
        stage = "transcribe"
        state.mark_started(episode.id, EpisodeStatus.TRANSCRIBING)
        engine_root = paths.engine_output / str(snapshot.year) / episode.id
        engine_root.mkdir(parents=True, exist_ok=True)
        # A fresh attempt directory prevents retained failed outputs from being reused.
        output = Path(tempfile.mkdtemp(prefix="attempt-", dir=engine_root))
        payload = dependencies.transcriber(audio.path, output, transcriber_script)
        stage = "archive"
        archive = build_archive(episode, audio, payload, snapshot.apple_collection_id)
        validate_archive(archive, snapshot.year)
        write_archive(_archive_path(paths, episode), archive, force=force)
        stage = "complete"
        state.mark_complete(episode.id)
        stage = "cleanup"
        shutil.rmtree(output)
        audio.path.unlink()
        summary.deleted_cache += 1
        _cache_sidecar(audio.path).unlink()
        summary.completed += 1
    except Exception as error:
        message = str(error) or type(error).__name__
        state.mark_failed(episode.id, stage, message)
        LOGGER.error("episode %s failed during %s: %s", episode.id, stage, message)
        summary.failed += 1
