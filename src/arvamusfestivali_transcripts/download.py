from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from .schema import DownloadedAudio, Episode


def audio_path(cache_root: Path, episode: Episode) -> Path:
    """Return the stable cache location for an episode's source audio."""
    return cache_root / str(episode.published_at.year) / f"{episode.id}.mp3"


def download_episode(
    client: httpx.Client,
    episode: Episode,
    cache_root: Path,
    retries: int = 3,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadedAudio:
    """Download one episode, resuming only when the server proves the offset."""
    if retries < 0:
        raise ValueError("retries must not be negative")

    final_path = audio_path(cache_root, episode)
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists() and _has_expected_size(final_path, episode.audio_bytes):
        return _downloaded_audio(final_path)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        try:
            _stream_to_part(client, episode, part_path)
            break
        except httpx.HTTPStatusError as error:
            if not _is_server_error(error) or attempt == retries:
                raise
        except httpx.RequestError:
            if attempt == retries:
                raise

        sleep(2**attempt)

    actual_bytes = part_path.stat().st_size
    if episode.audio_bytes is not None and actual_bytes != episode.audio_bytes:
        raise ValueError(
            "download size mismatch: "
            f"expected {episode.audio_bytes} bytes, got {actual_bytes} bytes"
        )

    part_path.replace(final_path)
    return _downloaded_audio(final_path)


def _stream_to_part(client: httpx.Client, episode: Episode, part_path: Path) -> None:
    has_partial = part_path.exists()
    offset = part_path.stat().st_size if has_partial else 0
    headers = {"Range": f"bytes={offset}-"} if has_partial else {}

    with client.stream("GET", episode.audio_url, headers=headers) as response:
        response.raise_for_status()
        mode = _part_write_mode(response, offset)
        with part_path.open(mode) as output:
            for chunk in response.iter_bytes():
                output.write(chunk)


def _part_write_mode(response: httpx.Response, offset: int) -> str:
    if offset == 0:
        return "wb"
    if response.status_code == 200:
        return "wb"
    expected_range = f"bytes {offset}-"
    if response.status_code == 206 and response.headers.get("Content-Range", "").startswith(
        expected_range
    ):
        return "ab"
    raise ValueError(f"server returned an invalid range response for byte {offset}")


def _has_expected_size(path: Path, expected_bytes: int | None) -> bool:
    return expected_bytes is None or path.stat().st_size == expected_bytes


def _downloaded_audio(path: Path) -> DownloadedAudio:
    with path.open("rb") as audio:
        digest = hashlib.file_digest(audio, "sha256").hexdigest()
    return DownloadedAudio(path=path, byte_count=path.stat().st_size, sha256=digest)


def _is_server_error(error: httpx.HTTPStatusError) -> bool:
    return 500 <= error.response.status_code < 600
