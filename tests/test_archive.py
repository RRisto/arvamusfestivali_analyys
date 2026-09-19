from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arvamusfestivali_transcripts.archive import (
    build_archive,
    timestamp_link,
    validate_archive,
    write_archive,
)
from arvamusfestivali_transcripts.schema import DownloadedAudio, Episode


@pytest.fixture
def engine_payload() -> dict[str, object]:
    return json.loads(
        (Path(__file__).parent / "fixtures" / "engine_output.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def episode() -> Episode:
    return Episode(
        id="2400217815",
        rss_guid="tag:soundcloud,2010:tracks/2400217815",
        title="Kelle vastutus on ennetus tervishoius_",
        published_at=datetime(2026, 9, 14, 13, 40, 21, tzinfo=UTC),
        published_raw="Mon, 14 Sep 2026 13:40:21 +0000",
        page_url="https://soundcloud.com/arvamusfestival/example-episode",
        audio_url="https://feeds.soundcloud.com/stream/2400217815-example.mp3",
        audio_bytes=60,
        duration_seconds=60.0,
    )


@pytest.fixture
def downloaded(tmp_path: Path) -> DownloadedAudio:
    path = tmp_path / "2400217815.mp3"
    path.write_bytes(b"archive audio")
    return DownloadedAudio(
        path=path,
        byte_count=13,
        sha256="a" * 64,
    )


@pytest.fixture
def archive(
    engine_payload: dict[str, object], episode: Episode, downloaded: DownloadedAudio
) -> dict[str, object]:
    return build_archive(episode, downloaded, engine_payload, 1477431807)


def test_build_archive_removes_raw_tokens(
    engine_payload: dict[str, object], episode: Episode, downloaded: DownloadedAudio
) -> None:
    archive = build_archive(episode, downloaded, engine_payload, 1477431807)

    assert archive["schema_version"] == 1
    assert archive["episode"] == {
        "id": "2400217815",
        "rss_guid": "tag:soundcloud,2010:tracks/2400217815",
        "title": "Kelle vastutus on ennetus tervishoius_",
        "published_at": "2026-09-14T13:40:21Z",
        "apple_collection_id": 1477431807,
        "page_url": "https://soundcloud.com/arvamusfestival/example-episode",
        "audio_url": "https://feeds.soundcloud.com/stream/2400217815-example.mp3",
        "audio_bytes": 13,
        "audio_sha256": "a" * 64,
        "duration_seconds": 60.0,
    }
    assert "source" not in archive["transcription"]
    assert "tokens" not in archive["transcription"]
    assert "timestamps" not in archive["transcription"]
    assert archive["transcription"]["cues"][0] == {
        "start_seconds": 0.0,
        "end_seconds": 4.2,
        "text": "Tere tulemast.",
    }
    validate_archive(archive, 2026)


def test_build_archive_preserves_complete_transcript_outer_whitespace(
    engine_payload: dict[str, object], episode: Episode, downloaded: DownloadedAudio
) -> None:
    engine_payload["transcript"] = "  Tere tulemast.\n"

    archive = build_archive(episode, downloaded, engine_payload, 1477431807)

    assert archive["transcription"]["text"] == "  Tere tulemast.\n"


def test_validate_archive_rejects_invalid_contract_values(archive: dict[str, object]) -> None:
    invalid_values = [
        (lambda data: data.update(schema_version=2), "schema version"),
        (lambda data: data["episode"].update(page_url="ftp://example.test"), "page URL"),
        (lambda data: data["episode"].update(audio_url="not a URL"), "audio URL"),
        (lambda data: data["episode"].update(audio_sha256="A" * 64), "SHA-256"),
        (lambda data: data["episode"].update(duration_seconds=0), "duration"),
        (lambda data: data["transcription"].update(text=" "), "text"),
        (lambda data: data["transcription"].update(cues=[]), "cue"),
        (
            lambda data: data["transcription"]["cues"][0].update(start_seconds=float("nan")),
            "finite",
        ),
        (
            lambda data: data["transcription"]["cues"][0].update(end_seconds=0.0),
            "start",
        ),
        (
            lambda data: data["transcription"]["cues"][1].update(start_seconds=-1.0),
            "start",
        ),
        (
            lambda data: data["transcription"]["cues"][1].update(end_seconds=60.3),
            "duration",
        ),
        (
            lambda data: data["transcription"]["cues"][1].update(start_seconds=-0.1),
            "start",
        ),
        (lambda data: data["transcription"]["model"].update(name=""), "model"),
        (lambda data: data["transcription"]["runtime"].update(sherpa_onnx=""), "runtime"),
    ]

    for mutate, message in invalid_values:
        candidate = deepcopy(archive)
        mutate(candidate)
        with pytest.raises(ValueError, match=message):
            validate_archive(candidate, 2026)


def test_validate_archive_rejects_wrong_year_and_non_monotonic_cues(
    archive: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="year"):
        validate_archive(archive, 2025)

    archive["transcription"]["cues"][0]["start_seconds"] = 1.0
    archive["transcription"]["cues"][1]["start_seconds"] = 0.0
    with pytest.raises(ValueError, match="monotonic"):
        validate_archive(archive, 2026)


@pytest.mark.parametrize("key", ["source", "tokens", "timestamps"])
def test_validate_archive_rejects_uncompacted_engine_fields(
    archive: dict[str, object], key: str
) -> None:
    archive["transcription"][key] = []

    with pytest.raises(ValueError, match="unrecognized"):
        validate_archive(archive, 2026)


def test_write_archive_is_immutable_atomic_and_unicode(
    tmp_path: Path, archive: dict[str, object]
) -> None:
    destination = tmp_path / "2026" / "2400217815.json"
    written = write_archive(destination, archive)

    assert written == destination
    rendered = destination.read_text(encoding="utf-8")
    assert "tervishoius" in rendered
    assert rendered.endswith("\n")
    assert json.loads(rendered) == archive
    assert not destination.with_suffix(".json.tmp").exists()

    with pytest.raises(FileExistsError):
        write_archive(destination, archive)

    replacement = deepcopy(archive)
    replacement["transcription"]["text"] = "Uuendatud tekst."
    write_archive(destination, replacement, force=True)
    assert json.loads(destination.read_text(encoding="utf-8")) == replacement


def test_write_archive_removes_temp_when_validation_fails(
    tmp_path: Path, archive: dict[str, object]
) -> None:
    destination = tmp_path / "archive.json"
    archive["episode"]["duration_seconds"] = 0

    with pytest.raises(ValueError, match="duration"):
        write_archive(destination, archive)

    assert not destination.exists()
    assert not destination.with_suffix(".json.tmp").exists()


def test_write_archive_preserves_preexisting_legacy_temp_file(
    tmp_path: Path, archive: dict[str, object]
) -> None:
    destination = tmp_path / "archive.json"
    legacy_temp = destination.with_suffix(".json.tmp")
    legacy_temp.write_text("keep me", encoding="utf-8")

    write_archive(destination, archive)

    assert destination.exists()
    assert legacy_temp.read_text(encoding="utf-8") == "keep me"


def test_write_archive_refuses_dangling_symlink_destination(
    tmp_path: Path, archive: dict[str, object]
) -> None:
    destination = tmp_path / "archive.json"
    try:
        destination.symlink_to(tmp_path / "missing.json")
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(FileExistsError):
        write_archive(destination, archive)

    assert destination.is_symlink()


def test_timestamp_link_uses_media_fragment() -> None:
    assert (
        timestamp_link("https://example.test/audio.mp3", 754.2)
        == "https://example.test/audio.mp3#t=754.2"
    )
