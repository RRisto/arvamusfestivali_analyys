from __future__ import annotations

import json
from pathlib import Path


def _write_transcript(root: Path, episode_id: str, title: str) -> Path:
    path = root / "data" / "transcripts" / "2026" / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "episode": {
                    "id": episode_id,
                    "title": title,
                    "published_at": "2026-01-01T00:00:00Z",
                    "audio_url": f"https://audio.example/{episode_id}.mp3",
                    "audio_bytes": 5,
                    "audio_sha256": "0" * 64,
                    "duration_seconds": 60,
                },
                "transcription": {
                    "cues": [
                        {
                            "start_seconds": 0.0,
                            "end_seconds": 1.0,
                            "text": title,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_valid_output(path: Path, episode_id: str, text: str = "done") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episode_id": episode_id,
                "model": "pyannote/test",
                "device": "cuda",
                "speakers": ["SPEAKER_00"],
                "turns": [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                        "speaker": "SPEAKER_00",
                    }
                ],
                "cues": [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                        "text": text,
                        "speaker": "SPEAKER_00",
                        "speaker_share": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_plan_batch_skips_valid_outputs_and_retries_invalid_ones(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.diarization_batch import plan_batch

    _write_transcript(tmp_path, "100", "Already complete")
    _write_transcript(tmp_path, "200", "Retry me")
    _write_transcript(tmp_path, "300", "Waiting")
    _write_valid_output(
        tmp_path / "data" / "diarization" / "2026" / "100.json", "100"
    )
    invalid = tmp_path / "data" / "diarization" / "2026" / "200.json"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text('{"episode_id": "wrong"}', encoding="utf-8")

    plan = plan_batch(tmp_path, limit=1)

    assert plan.total == 3
    assert plan.skipped == 1
    assert [job.episode_id for job in plan.jobs] == ["200"]
    assert plan.remaining_after_limit == 1


def test_run_batch_continues_after_failure_and_records_resumable_status(
    tmp_path: Path,
) -> None:
    from arvamusfestivali_transcripts.diarization_batch import run_batch

    _write_transcript(tmp_path, "100", "Fails")
    _write_transcript(tmp_path, "200", "Works")

    def process(job: object) -> None:
        episode_id = str(getattr(job, "episode_id"))
        if episode_id == "100":
            raise RuntimeError("GPU out of memory")
        _write_valid_output(Path(getattr(job, "output")), episode_id, "Works")

    summary = run_batch(tmp_path, process=process)

    assert summary.completed == ["200"]
    assert summary.failed == {"100": "GPU out of memory"}
    status = json.loads(
        (tmp_path / "data" / "diarization" / "batch-status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["completed"] == ["200"]
    assert status["failed"] == {"100": "GPU out of memory"}
    assert (tmp_path / "site" / "diarization" / "200.html").is_file()


def test_review_index_links_completed_pages_and_shows_failures(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.diarization_batch import build_review_index

    transcript = _write_transcript(tmp_path, "100", "Books & children")
    _write_transcript(tmp_path, "200", "Broken episode")
    output = tmp_path / "data" / "diarization" / "2026" / "100.json"
    _write_valid_output(output, "100")
    from arvamusfestivali_transcripts.diarization_review import build_review_page

    review = tmp_path / "site" / "diarization" / "100.html"
    build_review_page(transcript, output, review)

    index = build_review_index(tmp_path, failures={"200": "download failed"})
    page = index.read_text(encoding="utf-8")

    assert 'href="100.html"' in page
    assert "Books &amp; children" in page
    assert "Broken episode" in page
    assert "download failed" in page
    assert "1 / 2 completed" in page


def test_cli_dry_run_prints_plan_without_requiring_token(
    tmp_path: Path, capsys: object
) -> None:
    from arvamusfestivali_transcripts.diarization_batch import main

    _write_transcript(tmp_path, "100", "First")
    _write_transcript(tmp_path, "200", "Second")

    assert main(["--root", str(tmp_path), "--limit", "1", "--dry-run"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "1 selected" in output
    assert "100 First" in output
    assert "200 Second" not in output
