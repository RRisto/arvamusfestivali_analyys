from __future__ import annotations

import json
from pathlib import Path


def _write_transcript(path: Path, *, audio_sha256: str, text: str) -> Path:
    payload = {
        "schema_version": 1,
        "episode": {
            "id": "100",
            "rss_guid": "guid-100",
            "title": "Episode",
            "published_at": "2026-01-01T00:00:00Z",
            "apple_collection_id": 1477431807,
            "page_url": "https://example.test/100",
            "audio_url": "https://example.test/100.mp3",
            "audio_bytes": 5,
            "audio_sha256": audio_sha256,
            "duration_seconds": 60,
        },
        "transcription": {
            "model": {
                "name": "large",
                "repository": "TalTechNLP/model",
                "revision": "revision",
                "precision": "int8",
            },
            "runtime": {"sherpa_onnx": "1.13.7", "decoding_method": "beam"},
            "text": text,
            "cues": [{"start_seconds": 0.0, "end_seconds": 2.0, "text": text}],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_diarization(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "episode_id": "100",
        "model": "pyannote/speaker-diarization-community-1",
        "device": "cuda",
        "speakers": ["SPEAKER_00"],
        "turns": [{"start_seconds": 0.0, "end_seconds": 2.0, "speaker": "SPEAKER_00"}],
        "cues": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reassign_saved_turns_requires_unchanged_audio_identity(tmp_path: Path) -> None:
    import pytest

    from arvamusfestivali_transcripts.diarization_reassign import reassign_saved_turns

    old = _write_transcript(tmp_path / "old.json", audio_sha256="1" * 64, text="old")
    new = _write_transcript(tmp_path / "new.json", audio_sha256="2" * 64, text="new")
    diarization = _write_diarization(tmp_path / "diarization.json")

    with pytest.raises(ValueError, match="audio SHA-256 changed"):
        reassign_saved_turns(old, new, diarization, tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()


def test_reassign_saved_turns_reuses_turns_and_records_provenance(tmp_path: Path) -> None:
    import hashlib

    from arvamusfestivali_transcripts.diarization_reassign import reassign_saved_turns

    old = _write_transcript(tmp_path / "old.json", audio_sha256="1" * 64, text="old")
    new = _write_transcript(tmp_path / "new.json", audio_sha256="1" * 64, text="new")
    diarization = _write_diarization(tmp_path / "diarization.json")
    output = tmp_path / "output.json"

    reassign_saved_turns(old, new, diarization, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["turns"] == json.loads(diarization.read_text())["turns"]
    assert payload["cues"][0]["text"] == "new"
    assert payload["cues"][0]["speaker"] == "SPEAKER_00"
    assert payload["audio_sha256"] == "1" * 64
    assert payload["transcript_sha256"] == hashlib.sha256(new.read_bytes()).hexdigest()


def test_reassign_tree_continues_after_identity_failure_and_writes_manifest(
    tmp_path: Path,
) -> None:
    from arvamusfestivali_transcripts.diarization_reassign import reassign_tree

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    saved_dir = tmp_path / "saved"
    output_dir = tmp_path / "output"
    _write_transcript(old_dir / "100.json", audio_sha256="1" * 64, text="old")
    _write_transcript(new_dir / "100.json", audio_sha256="1" * 64, text="new")
    _write_diarization(saved_dir / "100.json")
    old_200 = json.loads((old_dir / "100.json").read_text())
    new_200 = json.loads((new_dir / "100.json").read_text())
    diar_200 = json.loads((saved_dir / "100.json").read_text())
    for payload in (old_200, new_200):
        payload["episode"]["id"] = "200"
        payload["episode"]["rss_guid"] = "guid-200"
    new_200["episode"]["audio_sha256"] = "2" * 64
    diar_200["episode_id"] = "200"
    (old_dir / "200.json").write_text(json.dumps(old_200))
    (new_dir / "200.json").write_text(json.dumps(new_200))
    (saved_dir / "200.json").write_text(json.dumps(diar_200))

    manifest = reassign_tree(old_dir, new_dir, saved_dir, output_dir)

    assert manifest["counts"] == {"selected": 2, "completed": 1, "failed": 1}
    assert (output_dir / "100.json").is_file()
    assert not (output_dir / "200.json").exists()
    assert manifest["failures"]["200"].startswith("audio SHA-256 changed")
