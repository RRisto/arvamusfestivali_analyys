from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import pytest

from arvamusfestivali_transcripts.download import audio_path, download_episode
from arvamusfestivali_transcripts.schema import Episode


@pytest.fixture
def episode() -> Episode:
    return Episode(
        id="2400217815",
        rss_guid="tag:soundcloud,2010:tracks/2400217815",
        title="Example episode",
        published_at=datetime(2026, 9, 14, 13, 40, 21, tzinfo=UTC),
        published_raw="Mon, 14 Sep 2026 13:40:21 +0000",
        page_url="https://soundcloud.com/arvamusfestival/example-episode",
        audio_url="https://example.com/2400217815.mp3",
        audio_bytes=None,
        duration_seconds=60.0,
    )


def client_returning(
    body: bytes, *, status: int, headers: dict[str, str] | None = None
) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, content=body, headers=headers)
        )
    )


def test_download_writes_hash_and_atomic_file(tmp_path, episode: Episode) -> None:
    """Dropping finalization leaves the completed data only at the partial path."""
    body = b"complete-audio"
    client = client_returning(body, status=200, headers={"Content-Length": str(len(body))})

    result = download_episode(client, replace(episode, audio_bytes=len(body)), tmp_path)

    assert result.path == tmp_path / "2026" / "2400217815.mp3"
    assert result.path.read_bytes() == body
    assert result.byte_count == len(body)
    assert result.sha256 == hashlib.sha256(body).hexdigest()
    assert not result.path.with_suffix(result.path.suffix + ".part").exists()


def test_download_resumes_partial_file(tmp_path, episode: Episode) -> None:
    """Ignoring a saved offset would corrupt a recovered transfer."""
    final = audio_path(tmp_path, episode)
    final.parent.mkdir(parents=True)
    part = final.with_suffix(final.suffix + ".part")
    part.write_bytes(b"first-")
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("Range"))
        return httpx.Response(206, content=b"second", headers={"Content-Range": "bytes 6-11/12"})

    result = download_episode(
        httpx.Client(transport=httpx.MockTransport(handler)),
        replace(episode, audio_bytes=12),
        tmp_path,
    )

    assert seen_range == ["bytes=6-"]
    assert result.path.read_bytes() == b"first-second"


def test_download_requests_zero_offset_for_empty_partial(tmp_path, episode: Episode) -> None:
    """Ignoring an empty partial file loses the explicit resume contract."""
    final = audio_path(tmp_path, episode)
    final.parent.mkdir(parents=True)
    final.with_suffix(final.suffix + ".part").touch()
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("Range"))
        return httpx.Response(206, content=b"audio", headers={"Content-Range": "bytes 0-4/5"})

    result = download_episode(
        httpx.Client(transport=httpx.MockTransport(handler)),
        replace(episode, audio_bytes=5),
        tmp_path,
    )

    assert seen_range == ["bytes=0-"]
    assert result.path.read_bytes() == b"audio"


def test_download_restarts_when_server_ignores_range(tmp_path, episode: Episode) -> None:
    """Appending a 200 response to a partial file would duplicate its prefix."""
    final = audio_path(tmp_path, episode)
    final.parent.mkdir(parents=True)
    final.with_suffix(final.suffix + ".part").write_bytes(b"stale-")
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("Range"))
        return httpx.Response(200, content=b"complete", headers={"Content-Length": "8"})

    result = download_episode(
        httpx.Client(transport=httpx.MockTransport(handler)),
        replace(episode, audio_bytes=8),
        tmp_path,
    )

    assert seen_range == ["bytes=6-"]
    assert result.path.read_bytes() == b"complete"


def test_download_preserves_partial_file_after_size_mismatch(tmp_path, episode: Episode) -> None:
    """Finalizing a truncated response would make a corrupt cache indistinguishable from audio."""
    body = b"truncated"
    final = audio_path(tmp_path, episode)

    with pytest.raises(ValueError, match="expected 10 bytes"):
        download_episode(
            client_returning(body, status=200), replace(episode, audio_bytes=10), tmp_path
        )

    assert not final.exists()
    assert final.with_suffix(".mp3.part").read_bytes() == body


def test_download_retries_transient_server_errors(tmp_path, episode: Episode) -> None:
    """Treating a 5xx response as final makes a temporary upstream failure permanent."""
    attempts = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            return httpx.Response(503)
        return httpx.Response(200, content=b"audio")

    result = download_episode(
        httpx.Client(transport=httpx.MockTransport(handler)),
        replace(episode, audio_bytes=5),
        tmp_path,
        sleep=sleeps.append,
    )

    assert result.path.read_bytes() == b"audio"
    assert attempts == 4
    assert sleeps == [1, 2, 4]


def test_download_does_not_retry_client_errors(tmp_path, episode: Episode) -> None:
    """Retrying a rejected URL needlessly repeats a non-transient request."""
    attempts = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        download_episode(
            httpx.Client(transport=httpx.MockTransport(handler)),
            episode,
            tmp_path,
            sleep=sleeps.append,
        )

    assert attempts == 1
    assert sleeps == []
