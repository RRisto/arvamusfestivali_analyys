from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL = {
    "name": "large",
    "repository": "TalTechNLP/streaming-zipformer-large.et-en",
    "revision": "2812262bd4fd5d4327dc09ca0e6c19e8a32a03d6",
    "precision": "int8",
}


def _write_catalog(path: Path, episode_ids: tuple[str, ...] = ("100", "200")) -> Path:
    payload = {
        "apple_collection_id": 1477431807,
        "apple_page_url": "https://example.test/podcast",
        "feed_url": "https://example.test/feed.xml",
        "resolved_at": "2026-09-20T11:25:40Z",
        "year": 2026,
        "episodes": [
            {
                "id": episode_id,
                "rss_guid": f"guid-{episode_id}",
                "title": f"Episode {episode_id}",
                "published_at": "2026-01-01T00:00:00Z",
                "published_raw": "Thu, 01 Jan 2026 00:00:00 +0000",
                "page_url": f"https://example.test/{episode_id}",
                "audio_url": f"https://example.test/{episode_id}.mp3",
                "audio_bytes": 5,
                "duration_seconds": 60,
            }
            for episode_id in episode_ids
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_archive(directory: Path, episode_id: str, audio_sha256: str, text: str = "Text") -> Path:
    path = directory / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "episode": {
            "id": episode_id,
            "rss_guid": f"guid-{episode_id}",
            "title": f"Episode {episode_id}",
            "published_at": "2026-01-01T00:00:00Z",
            "apple_collection_id": 1477431807,
            "page_url": f"https://example.test/{episode_id}",
            "audio_url": f"https://example.test/{episode_id}.mp3",
            "audio_bytes": 5,
            "audio_sha256": audio_sha256,
            "duration_seconds": 60,
        },
        "transcription": {
            "model": MODEL,
            "runtime": {
                "sherpa_onnx": "1.13.7",
                "decoding_method": "modified_beam_search",
                "threads": 4,
            },
            "text": text,
            "cues": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": text}],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_create_run_lock_pins_catalog_script_model_and_audio_identity(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.retranscription import create_run_lock

    catalog = _write_catalog(tmp_path / "catalog.json")
    baseline = tmp_path / "baseline"
    _write_archive(baseline, "100", "1" * 64)
    _write_archive(baseline, "200", "2" * 64)
    script = tmp_path / "transcribe.py"
    script.write_text("# pinned transcriber\n", encoding="utf-8")

    lock = create_run_lock(catalog, baseline, script, expected_model=MODEL)

    assert lock["catalog_sha256"] == hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert lock["transcriber_script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert lock["expected_model"] == MODEL
    assert lock["episodes"] == {
        "100": {"audio_sha256": "1" * 64, "audio_url": "https://example.test/100.mp3"},
        "200": {"audio_sha256": "2" * 64, "audio_url": "https://example.test/200.mp3"},
    }


def test_validate_stage_reports_missing_mismatch_and_unexpected_files(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.retranscription import create_run_lock, validate_stage

    catalog = _write_catalog(tmp_path / "catalog.json")
    baseline = tmp_path / "baseline"
    _write_archive(baseline, "100", "1" * 64)
    _write_archive(baseline, "200", "2" * 64)
    script = tmp_path / "transcribe.py"
    script.write_text("# pinned transcriber\n", encoding="utf-8")
    lock = create_run_lock(catalog, baseline, script, expected_model=MODEL)
    stage = tmp_path / "stage"
    _write_archive(stage, "100", "9" * 64)
    _write_archive(stage, "999", "3" * 64)

    manifest = validate_stage(catalog, stage, lock, year=2026)

    assert manifest["counts"] == {
        "expected": 2,
        "valid": 0,
        "missing": 1,
        "failed": 2,
        "unexpected": 1,
    }
    assert {failure["episode_id"] for failure in manifest["failures"]} == {"100", "200"}
    assert manifest["unexpected_files"] == ["999.json"]


def test_validate_stage_records_checksums_only_after_complete_validation(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.retranscription import create_run_lock, validate_stage

    catalog = _write_catalog(tmp_path / "catalog.json")
    baseline = tmp_path / "baseline"
    stage = tmp_path / "stage"
    for episode_id, digest in (("100", "1" * 64), ("200", "2" * 64)):
        _write_archive(baseline, episode_id, digest, text=f"old {episode_id}")
        _write_archive(stage, episode_id, digest, text=f"new {episode_id}")
    script = tmp_path / "transcribe.py"
    script.write_text("# pinned transcriber\n", encoding="utf-8")
    lock = create_run_lock(catalog, baseline, script, expected_model=MODEL)

    manifest = validate_stage(catalog, stage, lock, year=2026)

    assert manifest["counts"]["valid"] == 2
    assert manifest["failures"] == []
    assert set(manifest["files"]) == {"100.json", "200.json"}
    assert manifest["files"]["100.json"]["sha256"] == hashlib.sha256(
        (stage / "100.json").read_bytes()
    ).hexdigest()


def test_promote_stage_keeps_recoverable_backup_and_refuses_changed_files(tmp_path: Path) -> None:
    import pytest

    from arvamusfestivali_transcripts.retranscription import (
        create_run_lock,
        promote_stage,
        validate_stage,
    )

    catalog = _write_catalog(tmp_path / "catalog.json", ("100",))
    target = tmp_path / "transcripts" / "2026"
    _write_archive(target, "100", "1" * 64, text="old")
    stage = tmp_path / "staging" / "2026"
    _write_archive(stage, "100", "1" * 64, text="new")
    script = tmp_path / "transcribe.py"
    script.write_text("# pinned transcriber\n", encoding="utf-8")
    lock = create_run_lock(catalog, target, script, expected_model=MODEL)
    manifest = validate_stage(catalog, stage, lock, year=2026)
    backup = tmp_path / "backups" / "before-retranscription"

    (stage / "100.json").write_text("changed after validation", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after validation"):
        promote_stage(stage, target, backup, manifest)
    assert target.is_dir()
    assert not backup.exists()

    _write_archive(stage, "100", "1" * 64, text="new")
    manifest = validate_stage(catalog, stage, lock, year=2026)
    promote_stage(stage, target, backup, manifest)
    assert json.loads((target / "100.json").read_text())["transcription"]["text"] == "new"
    assert json.loads((backup / "100.json").read_text())["transcription"]["text"] == "old"
    assert not stage.exists()


def test_cli_create_lock_and_validate_write_operator_artifacts(tmp_path: Path) -> None:
    from arvamusfestivali_transcripts.retranscription import main

    catalog = _write_catalog(tmp_path / "catalog.json", ("100",))
    baseline = tmp_path / "baseline"
    stage = tmp_path / "stage"
    _write_archive(baseline, "100", "1" * 64, text="old")
    _write_archive(stage, "100", "1" * 64, text="new")
    script = tmp_path / "transcribe.py"
    script.write_text("# pinned transcriber\n", encoding="utf-8")
    lock_path = tmp_path / "run-lock.json"
    manifest_path = tmp_path / "manifest.json"

    assert main(
        [
            "create-lock",
            "--catalog",
            str(catalog),
            "--baseline",
            str(baseline),
            "--transcriber-script",
            str(script),
            "--output",
            str(lock_path),
        ]
    ) == 0
    assert main(
        [
            "validate",
            "--catalog",
            str(catalog),
            "--stage",
            str(stage),
            "--lock",
            str(lock_path),
            "--manifest",
            str(manifest_path),
            "--year",
            "2026",
        ]
    ) == 0
    assert json.loads(manifest_path.read_text())["counts"]["valid"] == 1
