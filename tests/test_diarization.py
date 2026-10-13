from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest


def test_assign_speakers_uses_the_largest_time_overlap() -> None:
    from arvamusfestivali_transcripts.diarization import assign_speakers

    cues = [{"start_seconds": 10.0, "end_seconds": 14.0, "text": "Tere."}]
    turns = [
        {"start_seconds": 9.0, "end_seconds": 11.0, "speaker": "SPEAKER_00"},
        {"start_seconds": 11.0, "end_seconds": 15.0, "speaker": "SPEAKER_01"},
    ]

    assert assign_speakers(cues, turns) == [
        {
            "start_seconds": 10.0,
            "end_seconds": 14.0,
            "text": "Tere.",
            "speaker": "SPEAKER_01",
            "speaker_overlap_seconds": 3.0,
            "speaker_share": 0.75,
        }
    ]


def test_assign_speakers_leaves_a_cue_unassigned_when_speech_does_not_overlap() -> None:
    from arvamusfestivali_transcripts.diarization import assign_speakers

    cues = [{"start_seconds": 20.0, "end_seconds": 22.0, "text": "Aplaus."}]

    assert assign_speakers(cues, []) == [
        {
            "start_seconds": 20.0,
            "end_seconds": 22.0,
            "text": "Aplaus.",
            "speaker": None,
            "speaker_overlap_seconds": 0.0,
            "speaker_share": 0.0,
        }
    ]


def test_download_audio_checks_the_archived_size_and_hash(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.diarization import download_audio

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/episode.mp3"
        return httpx.Response(200, content=b"audio")

    destination = tmp_path / "episode.mp3"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = download_audio(
            client,
            "https://example.test/episode.mp3",
            destination,
            expected_bytes=5,
            expected_sha256="6ed8919ce20490a5e3ad8630a4fab69475297abd07db73918dd5f36fcfaeb11b",
        )

    assert result == destination
    assert destination.read_bytes() == b"audio"


def test_download_audio_removes_a_file_with_the_wrong_hash(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.diarization import download_audio

    destination = tmp_path / "episode.mp3"
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"audio"))

    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            download_audio(
                client,
                "https://example.test/episode.mp3",
                destination,
                expected_bytes=5,
                expected_sha256="0" * 64,
            )

    assert not destination.exists()


def test_pilot_writes_turns_and_cue_assignments_for_local_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arvamusfestivali_transcripts.diarization as diarization

    transcript = tmp_path / "2258666735.json"
    transcript.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episode": {
                    "id": "2258666735",
                    "published_at": "2026-01-01T00:00:00Z",
                    "audio_url": "https://example.test/episode.mp3",
                    "audio_bytes": 5,
                    "audio_sha256": (
                        "6ed8919ce20490a5e3ad8630a4fab694"
                        "75297abd07db73918dd5f36fcfaeb11b"
                    ),
                },
                "transcription": {
                    "cues": [
                        {"start_seconds": 0.0, "end_seconds": 2.0, "text": "Tere."}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    output = tmp_path / "diarization.json"
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        diarization,
        "run_pyannote",
        lambda *args, **kwargs: (
            [{"start_seconds": 0.0, "end_seconds": 2.0, "speaker": "SPEAKER_00"}],
            "cpu",
        ),
    )

    assert diarization.main(
        [str(transcript), "--audio", str(audio), "--output", str(output)]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "episode_id": "2258666735",
        "model": "pyannote/speaker-diarization-community-1",
        "device": "cpu",
        "speakers": ["SPEAKER_00"],
        "turns": [
            {"start_seconds": 0.0, "end_seconds": 2.0, "speaker": "SPEAKER_00"}
        ],
        "cues": [
            {
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "text": "Tere.",
                "speaker": "SPEAKER_00",
                "speaker_overlap_seconds": 2.0,
                "speaker_share": 1.0,
            }
        ],
    }
