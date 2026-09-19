from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .schema import DownloadedAudio, Episode

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def build_archive(
    episode: Episode,
    downloaded_audio: DownloadedAudio,
    engine_payload: Mapping[str, Any],
    apple_collection_id: int,
) -> dict[str, Any]:
    """Convert transcription-engine JSON to the compact archive schema."""
    cues = _require_sequence(engine_payload, "cues")
    model = _require_mapping(engine_payload, "model")
    runtime = _require_mapping(engine_payload, "runtime")
    transcription_cues = [_archive_cue(cue) for cue in cues]

    return {
        "schema_version": 1,
        "episode": {
            "id": episode.id,
            "rss_guid": episode.rss_guid,
            "title": episode.title,
            "published_at": _utc_timestamp(episode.published_at),
            "apple_collection_id": apple_collection_id,
            "page_url": episode.page_url,
            "audio_url": episode.audio_url,
            "audio_bytes": downloaded_audio.byte_count,
            "audio_sha256": downloaded_audio.sha256,
            "duration_seconds": episode.duration_seconds,
        },
        "transcription": {
            "model": dict(model),
            "runtime": dict(runtime),
            "text": _require_text(engine_payload, "transcript"),
            "cues": transcription_cues,
        },
    }


def validate_archive(payload: Mapping[str, Any], expected_year: int) -> None:
    """Raise ValueError unless *payload* is a complete version-one archive."""
    _reject_unrecognized(payload, {"schema_version", "episode", "transcription"}, "archive")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported schema version")

    episode = _require_mapping(payload, "episode")
    transcription = _require_mapping(payload, "transcription")
    _reject_unrecognized(
        episode,
        {
            "id",
            "rss_guid",
            "title",
            "published_at",
            "apple_collection_id",
            "page_url",
            "audio_url",
            "audio_bytes",
            "audio_sha256",
            "duration_seconds",
        },
        "episode",
    )
    _reject_unrecognized(transcription, {"model", "runtime", "text", "cues"}, "transcription")
    _validate_episode(episode, expected_year)
    _validate_transcription(transcription, float(episode["duration_seconds"]))


def write_archive(path: Path, payload: Mapping[str, Any], force: bool = False) -> Path:
    """Validate and atomically write a UTF-8 archive without overwriting by default."""
    if os.path.lexists(path) and not force:
        raise FileExistsError(f"refusing to overwrite existing archive: {path}")

    episode = _require_mapping(payload, "episode")
    published_at = _parse_timestamp(_require_text(episode, "published_at"))
    validate_archive(payload, published_at.year)

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"refusing to overwrite existing archive: {path}") from error
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def timestamp_link(audio_url: str, start_seconds: float) -> str:
    """Return a W3C media-fragment URL for a point in an audio recording."""
    return f"{audio_url}#t={start_seconds}"


def _archive_cue(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("engine cue must be an object")
    return {
        "start_seconds": value.get("start"),
        "end_seconds": value.get("end"),
        "text": _require_text(value, "text").strip(),
    }


def _validate_episode(episode: Mapping[str, Any], expected_year: int) -> None:
    for key in ("id", "rss_guid", "title"):
        _require_text(episode, key)

    published_at = _parse_timestamp(_require_text(episode, "published_at"))
    if published_at.year != expected_year:
        raise ValueError("episode publication year does not match requested year")

    collection_id = episode.get("apple_collection_id")
    if not isinstance(collection_id, int) or isinstance(collection_id, bool) or collection_id <= 0:
        raise ValueError("apple collection id must be positive")
    for key, description in (("page_url", "page URL"), ("audio_url", "audio URL")):
        if not _is_http_url(episode.get(key)):
            raise ValueError(f"{description} must be an HTTP(S) URL")
    byte_count = episode.get("audio_bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
        raise ValueError("audio byte count must be positive")
    digest = episode.get("audio_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("audio SHA-256 must be a lowercase 64-character hexadecimal digest")
    duration = episode.get("duration_seconds")
    if not _is_finite_number(duration) or duration <= 0:
        raise ValueError("audio duration must be positive and finite")


def _validate_transcription(transcription: Mapping[str, Any], duration: float) -> None:
    _require_text(transcription, "text")
    model = _require_mapping(transcription, "model")
    runtime = _require_mapping(transcription, "runtime")
    for key in ("name", "repository", "revision", "precision"):
        _require_text(model, key, description="model metadata")
    for key in ("sherpa_onnx", "decoding_method"):
        _require_text(runtime, key, description="runtime metadata")

    cues = _require_sequence(transcription, "cues")
    if not cues:
        raise ValueError("transcription must contain at least one cue")
    previous_start = -math.inf
    for value in cues:
        cue = _require_mapping_value(value, "cue")
        _reject_unrecognized(cue, {"start_seconds", "end_seconds", "text"}, "cue")
        start = cue.get("start_seconds")
        end = cue.get("end_seconds")
        if not _is_finite_number(start) or not _is_finite_number(end):
            raise ValueError("cue times must be finite")
        if start < 0 or start >= end:
            raise ValueError("cue start must be non-negative and before its end")
        if start < previous_start:
            raise ValueError("cue starts must be monotonic")
        if end > duration + 0.25:
            raise ValueError("cue end exceeds audio duration")
        _require_text(cue, "text", description="cue text")
        previous_start = start


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _require_mapping_value(payload.get(key), key)


def _require_mapping_value(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _reject_unrecognized(
    payload: Mapping[str, Any], allowed: set[str], description: str
) -> None:
    unrecognized = set(payload) - allowed
    if unrecognized:
        names = ", ".join(sorted(unrecognized))
        raise ValueError(f"{description} contains unrecognized fields: {names}")


def _require_sequence(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _require_text(payload: Mapping[str, Any], key: str, *, description: str | None = None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        label = description or key
        raise ValueError(f"{label} must be nonempty text")
    return value


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _parse_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("published_at must be an ISO 8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("published_at must include a timezone")
    return timestamp.astimezone(UTC)


def _utc_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
