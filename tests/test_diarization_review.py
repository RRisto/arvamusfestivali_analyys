from __future__ import annotations

import json
from pathlib import Path


def test_merge_speaker_cues_keeps_separate_turns_and_marks_low_confidence() -> None:
    from arvamusfestivali_transcripts.diarization_review import merge_speaker_cues

    cues = [
        {
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "text": "Tere",
            "speaker": "SPEAKER_00",
            "speaker_share": 0.9,
        },
        {
            "start_seconds": 2.0,
            "end_seconds": 4.0,
            "text": "tulemast.",
            "speaker": "SPEAKER_00",
            "speaker_share": 0.5,
        },
        {
            "start_seconds": 4.0,
            "end_seconds": 5.0,
            "text": "Aitäh.",
            "speaker": "SPEAKER_01",
            "speaker_share": 1.0,
        },
        {
            "start_seconds": 5.0,
            "end_seconds": 6.0,
            "text": "Jätkan.",
            "speaker": "SPEAKER_00",
            "speaker_share": 1.0,
        },
    ]

    assert merge_speaker_cues(cues) == [
        {
            "id": "segment-0001",
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 4.0,
            "text": "Tere tulemast.",
            "cue_count": 2,
            "minimum_speaker_share": 0.5,
            "needs_review": True,
        },
        {
            "id": "segment-0002",
            "speaker": "SPEAKER_01",
            "start": 4.0,
            "end": 5.0,
            "text": "Aitäh.",
            "cue_count": 1,
            "minimum_speaker_share": 1.0,
            "needs_review": False,
        },
        {
            "id": "segment-0003",
            "speaker": "SPEAKER_00",
            "start": 5.0,
            "end": 6.0,
            "text": "Jätkan.",
            "cue_count": 1,
            "minimum_speaker_share": 1.0,
            "needs_review": False,
        },
    ]


def test_build_review_page_embeds_safe_data_and_clickable_audio_segments(
    tmp_path: Path,
) -> None:
    from arvamusfestivali_transcripts.diarization_review import build_review_page

    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "episode": {
                    "id": "123",
                    "title": "Raamatud <lastele>",
                    "audio_url": "https://audio.example/episode.mp3",
                    "page_url": "https://example.test/episode",
                    "duration_seconds": 30,
                },
                "transcription": {"cues": []},
            }
        ),
        encoding="utf-8",
    )
    diarization = tmp_path / "diarization.json"
    diarization.write_text(
        json.dumps(
            {
                "episode_id": "123",
                "model": "pyannote/test",
                "speakers": ["SPEAKER_00"],
                "turns": [],
                "cues": [
                    {
                        "start_seconds": 1.25,
                        "end_seconds": 3.5,
                        "text": "Tere </script> maailm.",
                        "speaker": "SPEAKER_00",
                        "speaker_share": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review.html"

    result = build_review_page(transcript, diarization, output)
    html = output.read_text(encoding="utf-8")

    assert result == output
    assert "Raamatud &lt;lastele&gt;" in html
    assert '<audio id="episode-audio" controls preload="metadata"' in html
    assert 'data-segment-id="segment-0001"' in html
    assert 'data-start="1.25"' in html
    assert "Tere &lt;/script&gt; maailm." in html
    assert "player.currentTime = start" in html
    assert "player.play()" in html
    assert "Tere </script> maailm." not in html


def test_review_cli_writes_the_default_episode_page(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.diarization_review import main

    transcript = tmp_path / "data" / "transcripts" / "2026" / "123.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "episode": {
                    "id": "123",
                    "title": "Test",
                    "audio_url": "https://audio.example/episode.mp3",
                    "page_url": "https://example.test/episode",
                    "duration_seconds": 30,
                },
                "transcription": {"cues": []},
            }
        ),
        encoding="utf-8",
    )
    diarization = tmp_path / "data" / "diarization" / "2026" / "123.json"
    diarization.parent.mkdir(parents=True)
    diarization.write_text(
        json.dumps(
            {
                "episode_id": "123",
                "model": "pyannote/test",
                "speakers": ["SPEAKER_00"],
                "turns": [],
                "cues": [
                    {
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "text": "Tere.",
                        "speaker": "SPEAKER_00",
                        "speaker_share": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main([str(transcript), "--diarization", str(diarization)]) == 0
    output = tmp_path / "site" / "diarization" / "123.html"
    assert output.is_file()
    assert "Tere." in output.read_text(encoding="utf-8")
