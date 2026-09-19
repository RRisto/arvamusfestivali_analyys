# Arvamusfestival Transcription Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable CLI that discovers Arvamusfestival episodes by year, downloads each MP3, transcribes it with timestamped Estonian speech recognition, commits one compact canonical JSON file per episode, and deletes verified audio cache files.

**Architecture:** A feed-driven Python application resolves the Apple collection to SoundCloud RSS, snapshots and parses the feed into immutable episode records, tracks mutable progress in SQLite, and processes episodes through downloader, transcription adapter, archive validator, and atomic writer boundaries. Canonical JSON and feed snapshots are durable Git data; MP3s, model files, engine output, and SQLite state remain local and reproducible.

**Tech Stack:** Python 3.12, uv, argparse, dataclasses, sqlite3, ElementTree, httpx, pytest, Ruff, FFmpeg, TalTech Zipformer through the existing `estonian-audio-transcription` skill.

**Spec:** `docs/superpowers/specs/2026-09-19-arvamusfestivali-transcription-design.md`

## Global Constraints

- Python version is `>=3.12`; use `uv` for locking, execution, testing, and builds.
- Apple Podcasts collection ID is `1477431807`.
- The current feed URL resolves to `https://feeds.soundcloud.com/users/soundcloud:users:96052562/sounds.rss`; resolve it through Apple instead of treating it as permanent configuration.
- Initial processing year is `2026`; every catalog and pipeline interface accepts another year unchanged.
- Canonical transcript format is JSON only and must retain cue-level timestamps, complete text, source URLs, SHA-256, duration, model revision, precision, and runtime version.
- Do not retain raw token arrays in canonical JSON.
- Audio is a Git-ignored cache and is deleted only after the canonical JSON validates and is atomically written.
- Existing valid transcripts are immutable unless `--force` is supplied.
- Do not add diarization, translation, manual editing, a web UI, or historical discovery beyond the RSS window.
- The first live acceptance run processes exactly one public 2026 episode before bulk processing.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/arvamusfestivali_transcripts/schema.py` | Immutable episode/download models plus catalog serialization. |
| `src/arvamusfestivali_transcripts/catalog.py` | Apple lookup, RSS retrieval, and durable source snapshots. |
| `src/arvamusfestivali_transcripts/feed.py` | RSS parsing, SoundCloud ID extraction, UTC normalization, year filtering, and duplicate policy. |
| `src/arvamusfestivali_transcripts/state.py` | SQLite state machine for resumable episode processing and failures. |
| `src/arvamusfestivali_transcripts/download.py` | `.part` downloads, HTTP range resume, size checks, SHA-256, and atomic rename. |
| `src/arvamusfestivali_transcripts/transcription.py` | Invocation and parsing of the existing Estonian transcription skill. |
| `src/arvamusfestivali_transcripts/archive.py` | Engine-output compaction, canonical JSON creation, validation, timestamp links, and atomic writes. |
| `src/arvamusfestivali_transcripts/pipeline.py` | Fetch, transcribe, and one-at-a-time run orchestration plus summary counts. |
| `src/arvamusfestivali_transcripts/__main__.py` | argparse CLI and exit codes. |
| `tests/fixtures/apple_lookup.json` | Fixed Apple lookup response. |
| `tests/fixtures/feed.xml` | Feed with valid, duplicate, cross-year, and malformed items. |
| `tests/fixtures/engine_output.json` | Representative transcription-skill JSON. |
| `tests/test_schema.py` | Model and catalog serialization tests. |
| `tests/test_catalog.py` | Resolver, snapshot, and network error tests. |
| `tests/test_feed.py` | RSS parsing, IDs, dates, filtering, and duplicates. |
| `tests/test_state.py` | State transitions and retry selection. |
| `tests/test_download.py` | Fresh downloads, range resume, size mismatch, and cleanup behavior. |
| `tests/test_transcription.py` | Exact subprocess command and output parsing. |
| `tests/test_archive.py` | Compaction, validation, links, immutability, and atomic writes. |
| `tests/test_pipeline.py` | Idempotence, summaries, cache deletion, and failure retention. |
| `tests/test_cli.py` | CLI parsing, dependency errors, selection flags, and exit codes. |
| `.gitignore` | Ignore audio cache, partials, state DB, and engine intermediates while retaining transcripts and snapshots. |
| `README.md` | Installation, commands, data policy, acceptance run, and later-year instructions. |

---

### Task 1: Lock dependencies and define core data contracts

**Files:**
- Modify: `pyproject.toml`
- Create: `src/arvamusfestivali_transcripts/schema.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Consumes: no application interfaces.
- Produces: `Episode`, `CatalogInfo`, `CatalogSnapshot`, `DownloadedAudio`, `episode_to_dict()`, `episode_from_dict()`, `CatalogSnapshot.to_json()`, and `CatalogSnapshot.from_json()`.

- [ ] **Step 1: Write failing serialization tests**

```python
from datetime import UTC, datetime

from arvamusfestivali_transcripts.schema import (
    CatalogSnapshot,
    Episode,
    episode_from_dict,
    episode_to_dict,
)


def sample_episode() -> Episode:
    return Episode(
        id="2400217815",
        rss_guid="tag:soundcloud,2010:tracks/2400217815",
        title="Kelle vastutus on ennetus tervishoius_",
        published_at=datetime(2026, 9, 14, 13, 40, 21, tzinfo=UTC),
        published_raw="Mon, 14 Sep 2026 13:40:21 +0000",
        page_url="https://soundcloud.com/arvamusfestival/example-episode",
        audio_url="https://feeds.soundcloud.com/stream/2400217815-example.mp3",
        audio_bytes=86222137,
        duration_seconds=5400.0,
    )


def test_episode_round_trip() -> None:
    episode = sample_episode()
    assert episode_from_dict(episode_to_dict(episode)) == episode


def test_snapshot_json_is_stable() -> None:
    snapshot = CatalogSnapshot(
        apple_collection_id=1477431807,
        apple_page_url="https://podcasts.apple.com/us/podcast/arvamusfestival/id1477431807",
        feed_url="https://feeds.soundcloud.com/users/soundcloud:users:96052562/sounds.rss",
        resolved_at=datetime(2026, 9, 19, 10, 0, tzinfo=UTC),
        year=2026,
        episodes=(sample_episode(),),
    )
    rendered = snapshot.to_json()
    assert '"year": 2026' in rendered
    assert rendered.endswith("\n")
    assert CatalogSnapshot.from_json(rendered) == snapshot
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `uv run pytest tests/test_schema.py -v`

Expected: FAIL because `arvamusfestivali_transcripts.schema` does not exist.

- [ ] **Step 3: Add the runtime dependency and bound the build backend**

Update `pyproject.toml` to contain:

```toml
dependencies = [
    "httpx>=0.28,<0.29",
]

[build-system]
requires = ["uv_build>=0.8.15,<0.13"]
build-backend = "uv_build"
```

Run: `uv lock && uv sync --dev`

Expected: lock and environment update succeed without the current unbounded-backend warning.

- [ ] **Step 4: Implement immutable models and explicit serialization**

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Episode:
    id: str
    rss_guid: str
    title: str
    published_at: datetime
    published_raw: str
    page_url: str
    audio_url: str
    audio_bytes: int | None
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CatalogInfo:
    collection_id: int
    collection_name: str
    apple_page_url: str
    feed_url: str


@dataclass(frozen=True, slots=True)
class DownloadedAudio:
    path: Path
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    apple_collection_id: int
    apple_page_url: str
    feed_url: str
    resolved_at: datetime
    year: int
    episodes: tuple[Episode, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["resolved_at"] = self.resolved_at.isoformat().replace("+00:00", "Z")
        payload["episodes"] = [episode_to_dict(item) for item in self.episodes]
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "CatalogSnapshot":
        payload = json.loads(text)
        payload["resolved_at"] = datetime.fromisoformat(
            payload["resolved_at"].replace("Z", "+00:00")
        )
        payload["episodes"] = tuple(
            episode_from_dict(item) for item in payload["episodes"]
        )
        return cls(**payload)


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    payload = asdict(episode)
    payload["published_at"] = episode.published_at.isoformat().replace("+00:00", "Z")
    return payload


def episode_from_dict(payload: dict[str, Any]) -> Episode:
    values = dict(payload)
    values["published_at"] = datetime.fromisoformat(values["published_at"].replace("Z", "+00:00"))
    return Episode(**values)
```

- [ ] **Step 5: Verify contracts, lint, and commit**

Run: `uv run pytest tests/test_schema.py -v && uv run ruff check .`

Expected: all schema tests pass and Ruff reports no errors.

```bash
git add pyproject.toml uv.lock src/arvamusfestivali_transcripts/schema.py tests/test_schema.py
git commit -m "feat: define archive data contracts"
```

---

### Task 2: Resolve Apple metadata and parse yearly RSS catalogs

**Files:**
- Create: `src/arvamusfestivali_transcripts/catalog.py`
- Create: `src/arvamusfestivali_transcripts/feed.py`
- Create: `tests/fixtures/apple_lookup.json`
- Create: `tests/fixtures/feed.xml`
- Create: `tests/test_catalog.py`
- Create: `tests/test_feed.py`

**Interfaces:**
- Consumes: `CatalogInfo`, `CatalogSnapshot`, and `Episode` from Task 1.
- Produces: `resolve_catalog(client: httpx.Client, collection_id: int) -> CatalogInfo`, `fetch_feed(client: httpx.Client, feed_url: str) -> bytes`, `parse_feed(xml_bytes: bytes, year: int) -> tuple[Episode, ...]`, and `write_snapshot(root: Path, snapshot: CatalogSnapshot, xml_bytes: bytes) -> tuple[Path, Path]`.

- [ ] **Step 1: Add representative fixtures**

`tests/fixtures/apple_lookup.json` must contain one result with collection ID
`1477431807`, the Apple page URL, collection name, and the SoundCloud RSS URL.
`tests/fixtures/feed.xml` must contain five items: two unique 2026 episodes, a
duplicate of one 2026 track ID, one 2025 episode, and one malformed item without
an enclosure URL. Use SoundCloud GUIDs in the form
`tag:soundcloud,2010:tracks/2400217815` and include `itunes:duration` values.

- [ ] **Step 2: Write failing resolver and feed tests**

```python
from pathlib import Path

import httpx

from arvamusfestivali_transcripts.catalog import resolve_catalog
from arvamusfestivali_transcripts.feed import parse_feed

FIXTURES = Path(__file__).parent / "fixtures"


def test_resolve_catalog_extracts_feed_url() -> None:
    payload = (FIXTURES / "apple_lookup.json").read_bytes()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload)))
    info = resolve_catalog(client, 1477431807)
    assert info.collection_id == 1477431807
    assert info.feed_url.endswith("sounds.rss")


def test_parse_feed_filters_year_and_deduplicates() -> None:
    episodes = parse_feed((FIXTURES / "feed.xml").read_bytes(), 2026)
    assert [episode.id for episode in episodes] == ["2400217815", "2400217000"]
    assert all(episode.published_at.year == 2026 for episode in episodes)
```

- [ ] **Step 3: Run tests and confirm missing-function failures**

Run: `uv run pytest tests/test_catalog.py tests/test_feed.py -v`

Expected: FAIL because catalog and feed modules do not exist.

- [ ] **Step 4: Implement strict Apple lookup handling**

```python
APPLE_LOOKUP_URL = "https://itunes.apple.com/lookup"


def resolve_catalog(client: httpx.Client, collection_id: int) -> CatalogInfo:
    response = client.get(
        APPLE_LOOKUP_URL,
        params={"id": collection_id, "media": "podcast", "entity": "podcast"},
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    matches = [item for item in results if item.get("collectionId") == collection_id]
    if len(matches) != 1 or not matches[0].get("feedUrl"):
        raise ValueError(f"Apple lookup did not return one feed for {collection_id}")
    item = matches[0]
    return CatalogInfo(
        collection_id=collection_id,
        collection_name=str(item["collectionName"]),
        apple_page_url=str(item["collectionViewUrl"]),
        feed_url=str(item["feedUrl"]),
    )
```

`fetch_feed()` must use `client.get()`, `raise_for_status()`, and return raw bytes.
Use a client configured by the caller with a 60-second timeout and follow redirects.

- [ ] **Step 5: Implement namespace-aware RSS parsing**

```python
SOUNDCLOUD_ID = re.compile(r"tracks/(\d+)$")


def parse_duration(value: str) -> float:
    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"invalid duration: {value}")


def parse_feed(xml_bytes: bytes, year: int) -> tuple[Episode, ...]:
    root = ElementTree.fromstring(xml_bytes)
    by_id: dict[str, Episode] = {}
    for item in root.findall("./channel/item"):
        guid = required_text(item, "guid")
        match = SOUNDCLOUD_ID.search(guid)
        enclosure = item.find("enclosure")
        if match is None or enclosure is None or not enclosure.get("url"):
            continue
        published_raw = required_text(item, "pubDate")
        published_at = parsedate_to_datetime(published_raw).astimezone(UTC)
        if published_at.year != year:
            continue
        episode = Episode(
            id=match.group(1),
            rss_guid=guid,
            title=required_text(item, "title"),
            published_at=published_at,
            published_raw=published_raw,
            page_url=required_text(item, "link"),
            audio_url=enclosure.get("url", ""),
            audio_bytes=int(enclosure.get("length")) if enclosure.get("length") else None,
            duration_seconds=parse_duration(required_text(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")),
        )
        by_id.setdefault(episode.id, episode)
    return tuple(sorted(by_id.values(), key=lambda item: (item.published_at, item.id)))
```

Add `required_text()` with a clear `ValueError` for missing required elements.
Document that first occurrence wins for duplicate stable IDs.

- [ ] **Step 6: Implement atomic source snapshots**

`write_snapshot()` must create `data/catalog/`, use a UTC filename such as
`20260919T100000Z-2026`, write XML and JSON through sibling `.tmp` files, call
`Path.replace()`, and return both final paths. Inject `resolved_at` into the
snapshot from the pipeline so tests never depend on wall-clock time.

- [ ] **Step 7: Verify catalog behavior and commit**

Run: `uv run pytest tests/test_catalog.py tests/test_feed.py -v && uv run ruff check .`

Expected: resolver, year filter, duplicate, malformed-item, duration, and snapshot tests pass.

```bash
git add src/arvamusfestivali_transcripts/catalog.py src/arvamusfestivali_transcripts/feed.py tests/fixtures tests/test_catalog.py tests/test_feed.py
git commit -m "feat: resolve and snapshot podcast catalog"
```

---

### Task 3: Add the resumable SQLite state machine

**Files:**
- Create: `src/arvamusfestivali_transcripts/state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes: episode IDs from `Episode.id`.
- Produces: `PipelineState`, `EpisodeRecord`, `EpisodeStatus`, `record_discovered()`, `mark_started()`, `mark_complete()`, `mark_failed()`, `status_for()`, and `retryable_ids()`.

- [ ] **Step 1: Write state-transition tests**

```python
from arvamusfestivali_transcripts.state import EpisodeStatus, PipelineState


def test_state_tracks_failure_and_retry(tmp_path) -> None:
    with PipelineState(tmp_path / "pipeline.sqlite3") as state:
        state.record_discovered("2400217815", 2026)
        state.mark_started("2400217815", EpisodeStatus.DOWNLOADING)
        state.mark_failed("2400217815", "download", "connection reset")
        record = state.status_for("2400217815")
        assert record.status is EpisodeStatus.FAILED
        assert record.failure_stage == "download"
        assert state.retryable_ids(2026) == ("2400217815",)
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `uv run pytest tests/test_state.py -v`

Expected: FAIL because `state.py` does not exist.

- [ ] **Step 3: Implement schema creation and guarded transitions**

Use a table with primary key `episode_id`, columns `year`, `status`,
`failure_stage`, `failure_message`, and `updated_at`. Define statuses as a string
enum: `discovered`, `downloading`, `downloaded`, `transcribing`, `complete`, and
`failed`. Use UTC ISO timestamps. `mark_complete()` clears failure fields;
`mark_failed()` requires a nonempty stage and message; unknown IDs raise
`KeyError`. `status_for()` returns an immutable `EpisodeRecord` containing those
six columns. Commit every public mutation before returning.

```python
class EpisodeStatus(StrEnum):
    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    TRANSCRIBING = "transcribing"
    COMPLETE = "complete"
    FAILED = "failed"
```

- [ ] **Step 4: Verify state behavior and commit**

Run: `uv run pytest tests/test_state.py -v && uv run ruff check .`

Expected: transition, persistence, retry, and failure-clearing tests pass.

```bash
git add src/arvamusfestivali_transcripts/state.py tests/test_state.py
git commit -m "feat: track resumable pipeline state"
```

---

### Task 4: Implement verified resumable audio downloads

**Files:**
- Create: `src/arvamusfestivali_transcripts/download.py`
- Create: `tests/test_download.py`

**Interfaces:**
- Consumes: `Episode`.
- Produces: `download_episode(client: httpx.Client, episode: Episode, cache_root: Path, retries: int = 3) -> DownloadedAudio` and `audio_path(cache_root: Path, episode: Episode) -> Path`.

- [ ] **Step 1: Write fresh-download and resume tests with `httpx.MockTransport`**

```python
def test_download_writes_hash_and_atomic_file(tmp_path, episode) -> None:
    body = b"complete-audio"
    client = client_returning(body, status=200, headers={"Content-Length": str(len(body))})
    result = download_episode(client, replace(episode, audio_bytes=len(body)), tmp_path)
    assert result.path.read_bytes() == body
    assert result.sha256 == hashlib.sha256(body).hexdigest()
    assert not result.path.with_suffix(result.path.suffix + ".part").exists()


def test_download_resumes_partial_file(tmp_path, episode) -> None:
    final = audio_path(tmp_path, episode)
    final.parent.mkdir(parents=True)
    part = final.with_suffix(final.suffix + ".part")
    part.write_bytes(b"first-")
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("Range"))
        return httpx.Response(206, content=b"second", headers={"Content-Range": "bytes 6-11/12"})

    result = download_episode(httpx.Client(transport=httpx.MockTransport(handler)), replace(episode, audio_bytes=12), tmp_path)
    assert seen_range == ["bytes=6-"]
    assert result.path.read_bytes() == b"first-second"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_download.py -v`

Expected: FAIL because download functions do not exist.

- [ ] **Step 3: Implement stable cache paths and response rules**

Use `data/audio-cache/<year>/<episode-id>.mp3`. If a `.part` file exists, send
`Range: bytes=<size>-`. Append only when the response is `206` and its
`Content-Range` starts at the requested byte. If the server returns `200`,
truncate the partial and restart. Reject other statuses through
`raise_for_status()`.

- [ ] **Step 4: Implement bounded retries, verification, and hashing**

Stream response bytes to the `.part` file. Retry connection, timeout, and 5xx
failures at most three attempts with delays of 1, 2, and 4 seconds; inject a
`sleep` callable in tests. Do not retry 4xx responses. After download, require
the actual length to equal `Episode.audio_bytes` when present. Hash the complete
`.part` contents with SHA-256, atomically replace the final path, and return
`DownloadedAudio`. Preserve `.part` after an interrupted transfer or size
mismatch; never create the final file on failure.

- [ ] **Step 5: Verify download edge cases and commit**

Run: `uv run pytest tests/test_download.py -v && uv run ruff check .`

Expected: fresh, resume, range-ignored, mismatch, retry, and 4xx tests pass.

```bash
git add src/arvamusfestivali_transcripts/download.py tests/test_download.py
git commit -m "feat: download podcast audio safely"
```

---

### Task 5: Convert and validate canonical transcript JSON

**Files:**
- Create: `src/arvamusfestivali_transcripts/archive.py`
- Create: `tests/fixtures/engine_output.json`
- Create: `tests/test_archive.py`

**Interfaces:**
- Consumes: `Episode`, `DownloadedAudio`, and the transcription skill's parsed JSON payload.
- Produces: `build_archive() -> dict[str, Any]`, `validate_archive(payload: Mapping[str, Any], expected_year: int) -> None`, `write_archive(path: Path, payload: Mapping[str, Any], force: bool = False) -> Path`, and `timestamp_link(audio_url: str, start_seconds: float) -> str`.

- [ ] **Step 1: Add a realistic engine fixture**

Create `engine_output.json` with `source`, `audio_duration_seconds`, complete
`transcript`, parallel `tokens` and `timestamps`, two `cues` using engine keys
`start`, `end`, and `text`, plus complete `model` and `runtime` objects matching
the transcription skill's current output.

- [ ] **Step 2: Write failing compaction and validation tests**

```python
def test_build_archive_removes_raw_tokens(engine_payload, episode, downloaded) -> None:
    archive = build_archive(episode, downloaded, engine_payload, 1477431807)
    assert archive["schema_version"] == 1
    assert "tokens" not in archive["transcription"]
    assert "timestamps" not in archive["transcription"]
    assert archive["transcription"]["cues"][0] == {
        "start_seconds": 0.0,
        "end_seconds": 4.2,
        "text": "Tere tulemast.",
    }
    validate_archive(archive, 2026)


def test_timestamp_link_uses_media_fragment() -> None:
    assert timestamp_link("https://example.test/audio.mp3", 754.2) == "https://example.test/audio.mp3#t=754.2"
```

- [ ] **Step 3: Run tests and confirm failure**

Run: `uv run pytest tests/test_archive.py -v`

Expected: FAIL because archive functions do not exist.

- [ ] **Step 4: Implement canonical conversion and strict validation**

`build_archive()` must map engine cues to `start_seconds`, `end_seconds`, and
trimmed text; copy the full transcript; preserve model/runtime dictionaries;
and build the exact episode object approved in the spec. It must not copy
`source`, `tokens`, or `timestamps` from engine output.

`validate_archive()` must enforce schema version 1, requested publication year,
HTTP(S) page/audio URLs, positive duration, a lowercase 64-character SHA-256,
nonempty text and cues, finite numeric times, `0 <= start < end`, monotonic cue
starts, cue ends no more than 0.25 seconds beyond duration, and nonempty model
name/repository/revision/precision and runtime version/decoder fields.

- [ ] **Step 5: Implement immutable atomic writes**

Serialize with `ensure_ascii=False`, `indent=2`, and a trailing newline. Refuse
an existing destination with `FileExistsError` unless `force=True`. Write a
sibling `.tmp`, flush and `os.fsync()`, then call `Path.replace()`. Remove the
temporary file if serialization or validation raises.

- [ ] **Step 6: Verify archive behavior and commit**

Run: `uv run pytest tests/test_archive.py -v && uv run ruff check .`

Expected: compaction, validation failures, Unicode output, immutability, force,
atomicity, and timestamp-link tests pass.

```bash
git add src/arvamusfestivali_transcripts/archive.py tests/fixtures/engine_output.json tests/test_archive.py
git commit -m "feat: create canonical transcript archives"
```

---

### Task 6: Integrate the Estonian transcription skill

**Files:**
- Create: `src/arvamusfestivali_transcripts/transcription.py`
- Create: `tests/test_transcription.py`

**Interfaces:**
- Consumes: local MP3 path, output directory, the transcription skill's `transcribe.py` path, and a uv executable.
- Produces: `run_transcriber(audio_path: Path, output_dir: Path, script_path: Path, uv_executable: str = "uv") -> dict[str, Any]`.

- [ ] **Step 1: Write a failing exact-command test**

```python
def test_run_transcriber_requests_json_only(tmp_path, monkeypatch, engine_payload) -> None:
    source = tmp_path / "2400217815.mp3"
    source.write_bytes(b"audio")
    script = tmp_path / "skill" / "scripts" / "transcribe.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('fixture')", encoding="utf-8")
    output_dir = tmp_path / "engine"
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command, cwd, check, capture_output, text):
        calls.append((command, cwd))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2400217815.json").write_text(json.dumps(engine_payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_transcriber(source, output_dir, script, uv_executable="uv")
    assert calls[0][0] == [
        "uv", "run", "scripts/transcribe.py", str(source.resolve()),
        "--output-dir", str(output_dir.resolve()), "--formats", "json",
    ]
    assert calls[0][1] == script.parent.parent
    assert result == engine_payload
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_transcription.py -v`

Expected: FAIL because `run_transcriber` does not exist.

- [ ] **Step 3: Implement dependency checks and invocation**

Require the audio file and script to exist. Require `shutil.which(uv_executable)`
unless the value is an existing absolute path. Require `shutil.which("ffmpeg")`
before MP3 input. Run from the skill root (the parent of `scripts/`) with
`check=True`, `capture_output=True`, and `text=True`. On failure, raise a
`RuntimeError` containing the exit code and the last nonempty stderr line.

- [ ] **Step 4: Parse exactly one engine JSON output**

Snapshot pre-existing `*.json` paths before invocation. After success, require
exactly one new JSON file, parse it as UTF-8, and return the mapping. Reject
missing, multiple, non-object, or invalid JSON output with a descriptive
`RuntimeError`. The caller owns engine-output cleanup.

- [ ] **Step 5: Verify adapter behavior and commit**

Run: `uv run pytest tests/test_transcription.py -v && uv run ruff check .`

Expected: command, missing dependency, subprocess failure, missing output,
multiple output, and invalid JSON tests pass.

```bash
git add src/arvamusfestivali_transcripts/transcription.py tests/test_transcription.py
git commit -m "feat: integrate Estonian transcription skill"
```

---

### Task 7: Orchestrate idempotent fetch, transcribe, and run workflows

**Files:**
- Create: `src/arvamusfestivali_transcripts/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: all interfaces from Tasks 1-6.
- Produces: `PipelinePaths`, `RunSummary`, `refresh_catalog()`, `fetch_year()`, `transcribe_year()`, and `run_year()`.

- [ ] **Step 1: Write a failing one-at-a-time success test**

```python
def test_run_year_deletes_audio_only_after_archive_success(tmp_path, dependencies, episode) -> None:
    dependencies.catalog_episodes = (episode,)
    summary = run_year(
        year=2026,
        paths=PipelinePaths.from_root(tmp_path),
        transcriber_script=tmp_path / "skill" / "scripts" / "transcribe.py",
        dependencies=dependencies,
        limit=1,
    )
    transcript = tmp_path / "data" / "transcripts" / "2026" / f"{episode.id}.json"
    audio = tmp_path / "data" / "audio-cache" / "2026" / f"{episode.id}.mp3"
    assert transcript.exists()
    assert not audio.exists()
    assert summary.completed == 1
    assert summary.deleted_cache == 1
    assert summary.failed == 0
```

Also test that archive validation failure leaves the MP3 in place, records a
failed state, and returns a nonzero failure count; an existing valid transcript
is skipped; `force=True` regenerates it; `episode_id` and pending-work `limit`
selection are deterministic.

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_pipeline.py -v`

Expected: FAIL because pipeline interfaces do not exist.

- [ ] **Step 3: Implement paths and summary values**

```python
@dataclass(frozen=True, slots=True)
class PipelinePaths:
    root: Path
    catalog: Path
    audio_cache: Path
    transcripts: Path
    engine_output: Path
    state_db: Path

    @classmethod
    def from_root(cls, root: Path) -> "PipelinePaths":
        data = root / "data"
        return cls(root, data / "catalog", data / "audio-cache", data / "transcripts", data / "engine-output", data / "state" / "pipeline.sqlite3")


@dataclass(slots=True)
class RunSummary:
    selected: int = 0
    skipped: int = 0
    completed: int = 0
    failed: int = 0
    downloaded_bytes: int = 0
    deleted_cache: int = 0
```

- [ ] **Step 4: Implement refresh and selection**

`refresh_catalog()` resolves Apple, fetches RSS, parses the year, writes XML and
JSON snapshots, and returns `CatalogSnapshot`. Reject an empty year selection.
`fetch_year()` and `run_year()` call `record_discovered()` only when not in dry
run mode. `transcribe_year()` loads the newest matching JSON snapshot through
`CatalogSnapshot.from_json()` and fails clearly when none exists. Apply optional
`episode_id` first, classify already-valid transcripts as skipped, and apply
`limit` to the remaining work items in stable `(published_at, id)` order.
`--dry-run` writes source snapshots but does not download or change episode
status; its `selected` count is the number of pending work items after `limit`.
An optional `catalog_snapshot` path loads an already committed JSON snapshot and
skips Apple/RSS network access, allowing many transcript batches to share one
frozen catalog.

- [ ] **Step 5: Implement fetch and transcribe workflows**

`fetch_year()` downloads selected episodes that lack a valid cache file and
marks downloaded or failed states. `transcribe_year()` requires cached audio,
invokes the adapter, builds and validates archive JSON, writes atomically, marks
complete, removes engine output, then unlinks MP3. If any step after download
fails, mark failed and preserve MP3. Skip an existing valid transcript unless
force is true.

- [ ] **Step 6: Implement streaming `run_year()`**

Process each selected episode through download and transcription before moving
to the next episode, limiting cache growth to one active MP3 plus retained
failures. Continue after individual failures and accumulate exact summary
counts. Return the summary; do not suppress the underlying failure message in
state or logs.

- [ ] **Step 7: Verify orchestration and commit**

Run: `uv run pytest tests/test_pipeline.py -v && uv run ruff check .`

Expected: success, failure retention, idempotent skip, force, retry selection,
limit, dry-run, summary, and cleanup-order tests pass.

```bash
git add src/arvamusfestivali_transcripts/pipeline.py tests/test_pipeline.py
git commit -m "feat: orchestrate resumable transcription runs"
```

---

### Task 8: Expose the CLI and document operations

**Files:**
- Modify: `src/arvamusfestivali_transcripts/__main__.py`
- Modify: `.gitignore`
- Modify: `README.md`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `fetch_year()`, `transcribe_year()`, `run_year()`, and `RunSummary`.
- Produces: `build_parser() -> argparse.ArgumentParser` and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing CLI parsing and exit-code tests**

```python
def test_run_command_accepts_year_and_acceptance_limit() -> None:
    args = build_parser().parse_args([
        "run", "--year", "2026", "--limit", "1",
        "--transcriber-script", "C:/skills/transcribe.py",
    ])
    assert args.command == "run"
    assert args.year == 2026
    assert args.limit == 1


def test_main_returns_one_when_any_episode_fails(monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_year", lambda **kwargs: RunSummary(selected=1, failed=1))
    code = main(["run", "--year", "2026", "--transcriber-script", "C:/skills/transcribe.py"])
    assert code == 1
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_cli.py -v`

Expected: FAIL because parser and command dispatch do not exist.

- [ ] **Step 3: Implement the command surface**

Create `fetch`, `transcribe`, and `run` subcommands. All accept `--year`,
`--root`, `--episode-id`, `--limit`, `--force`, and `--dry-run` where relevant.
`transcribe` and `run` require `--transcriber-script`; all commands accept
`--uv-executable` defaulting to `uv`. `run` also accepts
`--catalog-snapshot PATH`; when set, require the snapshot year to match `--year`
and do not contact Apple or RSS. Validate years from 2019 through the current
UTC year, positive limits, and a numeric episode ID. Print summary as one line:

```text
selected=1 skipped=0 completed=1 failed=0 downloaded_bytes=86222137 deleted_cache=1
```

Return 0 when no failures occurred, 1 for episode failures, and 2 for invalid
configuration or dependency errors.

- [ ] **Step 4: Apply precise Git ignore rules**

Append:

```gitignore
data/audio-cache/
data/engine-output/
data/state/
*.part
```

Do not ignore `data/catalog/` or `data/transcripts/`.

- [ ] **Step 5: Replace the scaffold README with operational documentation**

Document the Apple and RSS sources, 2026 scope, prerequisites, the exact local
transcriber script path configuration, dry run, one-episode acceptance run,
full run, retry by episode ID, adding a later year, JSON schema, timestamp-link
construction, cache deletion guarantee, Git policy, and the lack of
diarization. Include these commands:

```powershell
uv sync --dev
uv run arvamusfestivali-transcripts run --year 2026 --dry-run --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
uv run arvamusfestivali-transcripts run --year 2026 --limit 1 --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
uv run arvamusfestivali-transcripts run --year 2026 --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
```

- [ ] **Step 6: Verify CLI, full suite, build, and commit**

Run:

```powershell
uv run pytest -q
uv run ruff check .
uv build
git status --short
```

Expected: all tests pass, Ruff is clean, source/wheel builds succeed without the
`uv_build` upper-bound warning, and only intended Task 8 files are modified.

```bash
git add src/arvamusfestivali_transcripts/__main__.py tests/test_cli.py .gitignore README.md
git commit -m "feat: expose transcription archive CLI"
```

---

### Task 9: Run the live 2026 acceptance gate

**Files:**
- Create through the CLI: `data/catalog/<UTC timestamp>-2026.xml`
- Create through the CLI: `data/catalog/<UTC timestamp>-2026.json`
- Create through the CLI: `data/transcripts/2026/<selected-track-id>.json`
- Modify if a real defect is found: the owning source and test file from Tasks 2-8.

**Interfaces:**
- Consumes: the complete CLI, installed FFmpeg, existing uv, public Apple/RSS/audio endpoints, and the existing transcription skill.
- Produces: one validated committed transcript and evidence that its audio cache was deleted.

- [ ] **Step 1: Run a live dry run and capture the selected count**

Run:

```powershell
uv run arvamusfestivali-transcripts run --year 2026 --dry-run --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
```

Expected: exit 0, a nonzero 2026 selection, and new XML/JSON catalog snapshots.
Compare the count with the live feed; do not hard-code 264 because the feed can change.

- [ ] **Step 2: Run exactly one episode end to end**

Run the same command with `--limit 1` and without `--dry-run`.

Expected: one MP3 downloads, one transcription JSON is produced and compacted,
the canonical archive validates, and the MP3 is removed. The first run may
download checksum-verified TalTech model assets.

- [ ] **Step 3: Inspect the generated archive mechanically**

Run a short `uv run python -c` check that loads the only new transcript, asserts
schema version 1, HTTP(S) page/audio URLs, nonempty text and cues, ordered cue
times, absent `tokens`/`timestamps`, and a 64-character SHA-256. Print the first
cue's generated `<audio_url>#t=<start_seconds>` link.

- [ ] **Step 4: Confirm cleanup and idempotence**

Verify `data/audio-cache/2026/<selected-track-id>.mp3` is absent. Repeat the run
with `--episode-id <selected-track-id>` and expect `skipped=1`, `completed=0`,
and no audio download.

- [ ] **Step 5: Run final verification and commit acceptance data**

Run:

```powershell
uv run pytest -q
uv run ruff check .
uv build
git status --short
```

Expected: all checks pass; Git shows catalog snapshots and exactly one new
canonical transcript, with no MP3, `.part`, SQLite, engine output, or model file.

```bash
git add data/catalog data/transcripts/2026
git commit -m "data: add first verified 2026 transcript"
```

If the live run exposes a defect, first add a fixture-based failing regression
test, verify the failure, implement the minimal fix, verify the suite, and commit
the fix separately before repeating the acceptance gate.

---

### Task 10: Process and commit the remaining 2026 archive in batches

**Files:**
- Create through the CLI: `data/transcripts/2026/<soundcloud-track-id>.json`
- Create through catalog refresh: `data/catalog/<UTC timestamp>-2026.xml`
- Create through catalog refresh: `data/catalog/<UTC timestamp>-2026.json`

**Interfaces:**
- Consumes: the acceptance-tested `run --year 2026` workflow.
- Produces: the complete currently discoverable 2026 transcript set in Git.

- [ ] **Step 1: Process a bounded batch**

Run `run --year 2026 --limit 10 --catalog-snapshot <accepted snapshot JSON>`
with the approved transcriber script, substituting the exact snapshot path
printed by Task 9. Confirm the summary has no failures before staging output. If
failures occur, use the state database and `--episode-id` to retry only those
episodes.

- [ ] **Step 2: Verify the batch before committing**

Run the test, Ruff, and build commands from Task 9. Run a JSON validation command
over every untracked or modified transcript in the batch and confirm no ignored
audio or runtime file appears in `git status --short --ignored` as trackable.

- [ ] **Step 3: Commit the verified batch**

```bash
git add data/catalog data/transcripts/2026
git commit -m "data: add 2026 transcript batch"
```

- [ ] **Step 4: Repeat bounded batches until no pending episodes remain**

Repeat Steps 1-3 with batches sized for the machine's transcription throughput.
The terminal condition is a dry run with `selected=0`, zero failures, every
catalog episode represented by a valid canonical transcript, and no cached MP3
files for completed episodes.

- [ ] **Step 5: Record the final archive audit**

Add a README status section containing the feed snapshot timestamp, selected
2026 episode count, committed transcript count, failed count, and the command
used for validation. Run the full verification suite and commit the audit:

```bash
git add README.md
git commit -m "docs: record 2026 archive completion"
```
