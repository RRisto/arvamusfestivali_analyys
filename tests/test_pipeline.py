from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from arvamusfestivali_transcripts import pipeline
from arvamusfestivali_transcripts.pipeline import (
    PipelineDependencies,
    PipelinePaths,
    run_year,
)
from arvamusfestivali_transcripts.schema import CatalogSnapshot, Episode
from arvamusfestivali_transcripts.state import EpisodeStatus, PipelineState

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def episode() -> Episode:
    return Episode(
        id="123",
        rss_guid="tag:soundcloud,2010:tracks/123",
        title="Arutelu",
        published_at=datetime(2026, 9, 14, tzinfo=UTC),
        published_raw="Mon, 14 Sep 2026 00:00:00 +0000",
        page_url="https://soundcloud.com/arvamusfestival/123",
        audio_url="https://example.test/123.mp3",
        audio_bytes=5,
        duration_seconds=60,
    )


def save_catalog(paths, episodes, *, resolved_at=None, name="frozen-2026.json", year=2026):
    snapshot = CatalogSnapshot(
        apple_collection_id=1477431807,
        apple_page_url="https://podcasts.apple.com/id1477431807",
        feed_url="https://example.test/feed.xml",
        resolved_at=resolved_at or datetime(2026, 9, 19, tzinfo=UTC),
        year=year,
        episodes=tuple(episodes),
    )
    paths.catalog.mkdir(parents=True, exist_ok=True)
    path = paths.catalog / name
    path.write_text(snapshot.to_json(), encoding="utf-8")
    return path


class ExternalServices:
    def __init__(self):
        self.requests = []
        self.events = []
        self.payload = json.loads((FIXTURES / "engine_output.json").read_text(encoding="utf-8"))
        self.failure_ids = set()

    def request(self, request):
        self.requests.append(str(request.url))
        self.events.append(("download", request.url.path))
        return httpx.Response(200, content=b"audio")

    def transcribe(self, audio_path, output_dir, script_path):
        assert audio_path.read_bytes() == b"audio"
        self.events.append(("transcribe", audio_path.stem))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(json.dumps(self.payload), encoding="utf-8")
        if audio_path.stem in self.failure_ids:
            raise RuntimeError("engine exploded")
        return self.payload

    def dependencies(self):
        return PipelineDependencies(
            client_factory=lambda: httpx.Client(transport=httpx.MockTransport(self.request)),
            transcriber=self.transcribe,
            now=lambda: datetime(2026, 9, 19, tzinfo=UTC),
        )


@pytest.fixture
def services():
    return ExternalServices()


def test_run_year_deletes_audio_only_after_archive_success(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    summary = run_year(
        year=2026,
        paths=paths,
        transcriber_script=tmp_path / "scripts/transcribe.py",
        dependencies=services.dependencies(),
        catalog_snapshot=snapshot,
        limit=1,
    )
    archive_path = paths.transcripts / "2026/123.json"
    assert json.loads(archive_path.read_text(encoding="utf-8"))["episode"]["id"] == "123"
    assert not (paths.audio_cache / "2026/123.mp3").exists()
    assert not list(paths.engine_output.rglob("*.json"))
    assert (summary.selected, summary.completed, summary.failed) == (1, 1, 0)
    assert (summary.downloaded_bytes, summary.deleted_cache) == (5, 1)
    with PipelineState(paths.state_db) as state:
        assert state.status_for("123").status == EpisodeStatus.COMPLETE


def run_frozen(paths, services, snapshot, **options):
    return run_year(
        year=2026,
        paths=paths,
        transcriber_script=paths.root / "scripts/transcribe.py",
        dependencies=services.dependencies(),
        catalog_snapshot=snapshot,
        **options,
    )


def test_validation_failure_preserves_audio_and_records_message(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    services.payload["cues"] = []
    summary = run_frozen(paths, services, snapshot)
    assert (summary.completed, summary.failed, summary.deleted_cache) == (0, 1, 0)
    assert (paths.audio_cache / "2026/123.mp3").read_bytes() == b"audio"
    assert not (paths.transcripts / "2026/123.json").exists()
    with PipelineState(paths.state_db) as state:
        record = state.status_for("123")
        assert record.status == EpisodeStatus.FAILED
        assert record.failure_stage == "archive"
        assert "cue" in record.failure_message


def test_completed_archives_skip_and_force_regenerates(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    run_frozen(paths, services, snapshot)
    services.payload["transcript"] = "Uus tekst."
    skipped = run_frozen(paths, services, snapshot)
    assert (skipped.selected, skipped.skipped, skipped.completed) == (0, 1, 0)
    assert skipped.downloaded_bytes == 0
    assert len(services.requests) == 1
    forced = run_frozen(paths, services, snapshot, force=True)
    assert (forced.selected, forced.skipped, forced.completed) == (1, 0, 1)
    archive = json.loads((paths.transcripts / "2026/123.json").read_text(encoding="utf-8"))
    assert archive["transcription"]["text"] == "Uus tekst."


def test_limit_counts_pending_after_skips_and_uses_stable_order(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    run_frozen(paths, services, snapshot)
    second = replace(episode, id="456", audio_url="https://example.test/456.mp3")
    third = replace(episode, id="789", audio_url="https://example.test/789.mp3")
    snapshot = save_catalog(paths, [third, second, episode])
    summary = run_frozen(paths, services, snapshot, limit=1)
    assert (summary.selected, summary.skipped, summary.completed) == (1, 1, 1)
    assert (paths.transcripts / "2026/456.json").exists()
    assert not (paths.transcripts / "2026/789.json").exists()
    with PipelineState(paths.state_db) as state:
        assert state.status_for("789").status == EpisodeStatus.DISCOVERED


def test_episode_filter_precedes_skip_and_limit(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    second = replace(episode, id="456", audio_url="https://example.test/456.mp3")
    snapshot = save_catalog(paths, [episode, second])
    summary = run_frozen(paths, services, snapshot, episode_id="456", limit=1)
    assert (summary.selected, summary.skipped, summary.completed) == (1, 0, 1)
    assert not (paths.transcripts / "2026/123.json").exists()
    with pytest.raises(ValueError, match="999"):
        run_frozen(paths, services, snapshot, episode_id="999")


def test_streaming_continues_after_failure_and_retries_cached_audio(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    second = replace(episode, id="456", audio_url="https://example.test/456.mp3")
    snapshot = save_catalog(paths, [episode, second])
    services.failure_ids.add("123")
    summary = run_frozen(paths, services, snapshot)
    assert (summary.completed, summary.failed, summary.downloaded_bytes) == (1, 1, 10)
    assert services.events == [
        ("download", "/123.mp3"),
        ("transcribe", "123"),
        ("download", "/456.mp3"),
        ("transcribe", "456"),
    ]
    services.failure_ids.clear()
    retried = run_frozen(paths, services, snapshot)
    assert (retried.selected, retried.skipped, retried.completed) == (1, 1, 1)
    assert (retried.downloaded_bytes, retried.deleted_cache) == (0, 1)
    assert len(services.requests) == 2


def test_dry_run_selects_but_does_not_create_state_or_audio(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    summary = run_frozen(paths, services, snapshot, dry_run=True, limit=1)
    assert (summary.selected, summary.completed, summary.failed) == (1, 0, 0)
    assert not paths.state_db.exists()
    assert not paths.audio_cache.exists()
    assert not paths.engine_output.exists()
    assert not services.requests


@pytest.mark.parametrize("failure", ["write", "complete", "cleanup", "unlink"])
def test_failure_boundaries_keep_audio_and_failure_state(
    tmp_path,
    services,
    episode,
    monkeypatch,
    failure,
):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])

    def fail(*args, **kwargs):
        raise OSError(f"failed at {failure}")

    if failure == "write":
        monkeypatch.setattr(pipeline, "write_archive", fail)
    elif failure == "complete":
        monkeypatch.setattr(PipelineState, "mark_complete", fail)
    elif failure == "cleanup":
        monkeypatch.setattr(pipeline.shutil, "rmtree", fail)
    else:
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path.suffix == ".mp3":
                fail()
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", unlink)
    summary = run_frozen(paths, services, snapshot)
    assert (summary.failed, summary.completed, summary.deleted_cache) == (1, 0, 0)
    assert (paths.audio_cache / "2026/123.mp3").exists()
    with PipelineState(paths.state_db) as state:
        assert state.status_for("123").status == EpisodeStatus.FAILED
        assert f"failed at {failure}" in state.status_for("123").failure_message


def test_cleanup_happens_after_commit_and_before_audio_deletion(
    tmp_path,
    services,
    episode,
    monkeypatch,
):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    original_cleanup = pipeline.shutil.rmtree

    def cleanup(path):
        assert (paths.audio_cache / "2026/123.mp3").exists()
        assert (paths.transcripts / "2026/123.json").exists()
        with PipelineState(paths.state_db) as state:
            assert state.status_for("123").status == EpisodeStatus.COMPLETE
        original_cleanup(path)

    monkeypatch.setattr(pipeline.shutil, "rmtree", cleanup)
    assert run_frozen(paths, services, snapshot).completed == 1


@pytest.mark.parametrize("year,episodes", [(2025, True), (2026, False)])
def test_frozen_catalog_rejects_wrong_year_or_empty_selection(
    tmp_path,
    services,
    episode,
    year,
    episodes,
):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode] if episodes else [], year=year)
    with pytest.raises(ValueError, match="year|episodes"):
        run_frozen(paths, services, snapshot)
    assert not paths.state_db.exists()
    assert not services.requests


@pytest.mark.parametrize("limit", [-1, 0])
def test_limit_must_be_positive(tmp_path, services, episode, limit):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    with pytest.raises(ValueError, match="limit"):
        run_frozen(paths, services, snapshot, limit=limit)


def live_services(services):
    def request(request):
        services.requests.append(str(request.url))
        if request.url.host == "itunes.apple.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "collectionId": 1477431807,
                            "collectionName": "Arvamusfestival",
                            "collectionViewUrl": "https://podcasts.apple.com/id1477431807",
                            "feedUrl": "https://example.test/feed.xml",
                        }
                    ]
                },
            )
        if request.url.path == "/feed.xml":
            return httpx.Response(200, content=(FIXTURES / "feed.xml").read_bytes())
        raise AssertionError("dry-run must not request audio")

    return replace(
        services.dependencies(),
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(request),
        ),
    )


@pytest.mark.parametrize("workflow", ["refresh", "fetch", "run"])
def test_live_catalog_writes_snapshots_without_state_in_dry_run(tmp_path, services, workflow):
    paths = PipelinePaths.from_root(tmp_path)
    arguments = dict(year=2026, paths=paths, dependencies=live_services(services))
    if workflow == "refresh":
        snapshot = pipeline.refresh_catalog(**arguments)
        assert len(snapshot.episodes) == 2
    elif workflow == "fetch":
        assert pipeline.fetch_year(**arguments, dry_run=True, limit=1).selected == 1
    else:
        assert (
            run_year(
                **arguments,
                transcriber_script=tmp_path / "script.py",
                dry_run=True,
                limit=1,
            ).selected
            == 1
        )
    assert len(list(paths.catalog.glob("*.xml"))) == 1
    assert len(list(paths.catalog.glob("*.json"))) == 1
    assert len(services.requests) == 2
    assert not paths.state_db.exists()
    assert not paths.audio_cache.exists()


def test_refresh_rejects_empty_year(tmp_path, services):
    paths = PipelinePaths.from_root(tmp_path)
    with pytest.raises(ValueError, match="episodes"):
        pipeline.refresh_catalog(year=2024, paths=paths, dependencies=live_services(services))


def test_fetch_is_idempotent_and_offline_transcribe_completes(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    fetched = pipeline.fetch_year(
        year=2026,
        paths=paths,
        dependencies=services.dependencies(),
        catalog_snapshot=snapshot,
    )
    assert (fetched.selected, fetched.completed, fetched.downloaded_bytes) == (1, 1, 5)
    with PipelineState(paths.state_db) as state:
        assert state.status_for("123").status == EpisodeStatus.DOWNLOADED
    again = pipeline.fetch_year(
        year=2026,
        paths=paths,
        dependencies=services.dependencies(),
        catalog_snapshot=snapshot,
    )
    assert (again.selected, again.skipped, again.downloaded_bytes) == (0, 1, 0)
    transcribed = pipeline.transcribe_year(
        year=2026,
        paths=paths,
        transcriber_script=tmp_path / "script.py",
        dependencies=services.dependencies(),
    )
    assert (transcribed.completed, transcribed.downloaded_bytes) == (1, 0)
    assert len(services.requests) == 1


def test_transcribe_missing_audio_fails_without_network(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    save_catalog(paths, [episode])
    result = pipeline.transcribe_year(
        year=2026,
        paths=paths,
        transcriber_script=tmp_path / "script.py",
        dependencies=services.dependencies(),
    )
    assert (result.selected, result.failed, result.completed) == (1, 1, 0)
    assert not services.requests
    with PipelineState(paths.state_db) as state:
        assert "cache" in state.status_for("123").failure_message


def test_transcribe_selects_newest_matching_snapshot_by_resolved_time(
    tmp_path,
    services,
    episode,
):
    paths = PipelinePaths.from_root(tmp_path)
    save_catalog(
        paths, [episode], name="zzz-old-2026.json", resolved_at=datetime(2026, 9, 17, tzinfo=UTC)
    )
    newer = replace(episode, id="456")
    save_catalog(paths, [newer], name="aaa-new-2026.json")
    save_catalog(paths, [], name="other-2025.json", year=2025)
    summary = pipeline.transcribe_year(
        year=2026,
        paths=paths,
        transcriber_script=tmp_path / "script.py",
        dependencies=services.dependencies(),
        episode_id="456",
        dry_run=True,
    )
    assert summary.selected == 1
    assert not paths.state_db.exists()
    assert not services.requests


def test_transcribe_without_snapshot_explains_missing_catalog(tmp_path, services):
    with pytest.raises(FileNotFoundError, match="snapshot.*2026"):
        pipeline.transcribe_year(
            year=2026,
            paths=PipelinePaths.from_root(tmp_path),
            transcriber_script=tmp_path / "script.py",
            dependencies=services.dependencies(),
        )


def test_fetch_download_failure_records_original_error_and_continues(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    second = replace(episode, id="456", audio_url="https://example.test/456.mp3")
    snapshot = save_catalog(paths, [episode, second])

    def request(request):
        return httpx.Response(404 if request.url.path == "/123.mp3" else 200, content=b"audio")

    dependencies = replace(
        services.dependencies(),
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(request),
        ),
    )
    result = pipeline.fetch_year(
        year=2026,
        paths=paths,
        dependencies=dependencies,
        catalog_snapshot=snapshot,
    )
    assert (result.selected, result.completed, result.failed, result.downloaded_bytes) == (
        2,
        1,
        1,
        5,
    )
    with PipelineState(paths.state_db) as state:
        assert state.status_for("123").failure_stage == "download"
        assert "404" in state.status_for("123").failure_message


@pytest.mark.parametrize("content", [b"", b"truncated"])
def test_fetch_replaces_invalid_cache(tmp_path, services, episode, content):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    cached = paths.audio_cache / "2026/123.mp3"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(content)
    result = pipeline.fetch_year(
        year=2026,
        paths=paths,
        dependencies=services.dependencies(),
        catalog_snapshot=snapshot,
    )
    assert (result.completed, result.downloaded_bytes, result.skipped) == (1, 5, 0)
    assert cached.read_bytes() == b"audio"


def test_fetch_rejects_empty_download_with_unknown_expected_size(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [replace(episode, audio_bytes=None)])
    dependencies = replace(
        services.dependencies(),
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"")),
        ),
    )
    result = pipeline.fetch_year(
        year=2026,
        paths=paths,
        dependencies=dependencies,
        catalog_snapshot=snapshot,
    )
    assert (result.completed, result.failed, result.downloaded_bytes) == (0, 1, 0)
    with PipelineState(paths.state_db) as state:
        assert state.status_for("123").status == EpisodeStatus.FAILED
    retried = pipeline.fetch_year(
        year=2026,
        paths=paths,
        dependencies=services.dependencies(),
        catalog_snapshot=snapshot,
    )
    assert (retried.completed, retried.failed, retried.downloaded_bytes) == (1, 0, 5)


def test_invalid_existing_archive_is_not_skipped_or_overwritten_without_force(
    tmp_path,
    services,
    episode,
):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    archive = paths.transcripts / "2026/123.json"
    archive.parent.mkdir(parents=True)
    archive.write_text("corrupt existing archive", encoding="utf-8")
    result = run_frozen(paths, services, snapshot)
    assert (result.selected, result.skipped, result.failed) == (1, 0, 1)
    assert archive.read_text(encoding="utf-8") == "corrupt existing archive"
    assert (paths.audio_cache / "2026/123.mp3").exists()
    assert run_frozen(paths, services, snapshot, force=True).completed == 1


def test_valid_archive_for_different_episode_does_not_cause_skip(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    run_frozen(paths, services, snapshot)
    archive = paths.transcripts / "2026/123.json"
    payload = json.loads(archive.read_text(encoding="utf-8"))
    payload["episode"]["id"] = "456"
    archive.write_text(json.dumps(payload), encoding="utf-8")
    result = run_frozen(paths, services, snapshot, dry_run=True)
    assert (result.selected, result.skipped) == (1, 0)


def test_dry_run_preserves_existing_failure_record(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [episode])
    services.failure_ids.add("123")
    run_frozen(paths, services, snapshot)
    with PipelineState(paths.state_db) as state:
        before = state.status_for("123")
    assert run_frozen(paths, services, snapshot, dry_run=True).selected == 1
    with PipelineState(paths.state_db) as state:
        assert state.status_for("123") == before


def test_publication_date_precedes_id_in_pending_order(tmp_path, services, episode):
    paths = PipelinePaths.from_root(tmp_path)
    earlier = replace(episode, id="999", published_at=datetime(2026, 1, 1, tzinfo=UTC))
    snapshot = save_catalog(paths, [episode, earlier])
    result = run_frozen(paths, services, snapshot, limit=1)
    assert result.completed == 1
    assert (paths.transcripts / "2026/999.json").exists()
    assert not (paths.transcripts / "2026/123.json").exists()


@pytest.mark.parametrize("identifier", ["../outside", "..\\outside", "C:/outside"])
def test_frozen_catalog_rejects_episode_ids_that_escape_storage(
    tmp_path,
    services,
    episode,
    identifier,
):
    paths = PipelinePaths.from_root(tmp_path)
    snapshot = save_catalog(paths, [replace(episode, id=identifier)])
    with pytest.raises(ValueError, match="episode ID"):
        run_frozen(paths, services, snapshot, dry_run=True)
    assert not paths.state_db.exists()
