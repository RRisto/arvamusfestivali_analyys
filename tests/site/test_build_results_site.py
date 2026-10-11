from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_results_site import build_site_data, normalize_search_text


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(root: Path, *, include_second_archive: bool = True) -> None:
    results = root / "data/topic-analysis/results"
    _write_csv(
        results / "jev-semantic-all/high-level-summary.csv",
        [
            {
                "jev_high_level_id": "education",
                "jev_high_level_name": "Haridus, õppimine ja oskused",
                "segments": 2,
                "talks": 1,
                "recording_hours": 0.2,
            },
            {
                "jev_high_level_id": "ai",
                "jev_high_level_name": "Tehisintellekt ja digitehnoloogia",
                "segments": 1,
                "talks": 1,
                "recording_hours": 0.1,
            },
        ],
    )
    segment_rows = [
        {
            "segment_key": "s1",
            "episode_id": "episode-1",
            "title": "Õppimise tulevik",
            "start_seconds": 5,
            "end_seconds": 15,
            "audio_link": "https://audio.invalid/one.mp3#t=5",
            "jev_topic_id": 1,
            "jev_topic_name": "Õpetaja roll",
            "jev_high_level_id": "education",
            "jev_high_level_name": "Haridus, õppimine ja oskused",
            "x": 1.5,
            "y": -2.5,
            "programme_year": 2025,
            "organizer": "Tartu Ülikool",
        },
        {
            "segment_key": "s2",
            "episode_id": "episode-1",
            "title": "Õppimise tulevik",
            "start_seconds": 15,
            "end_seconds": 25,
            "audio_link": "https://audio.invalid/one.mp3#t=15",
            "jev_topic_id": 2,
            "jev_topic_name": "Digioskused",
            "jev_high_level_id": "education",
            "jev_high_level_name": "Haridus, õppimine ja oskused",
            "x": 2.0,
            "y": -1.0,
            "programme_year": 2025,
            "organizer": "Tartu Ülikool",
        },
        {
            "segment_key": "s3",
            "episode_id": "episode-2",
            "title": "Tehisaru tööl",
            "start_seconds": 0,
            "end_seconds": 10,
            "audio_link": "https://audio.invalid/two.mp3#t=0",
            "jev_topic_id": 3,
            "jev_topic_name": "AI assistendid",
            "jev_high_level_id": "ai",
            "jev_high_level_name": "Tehisintellekt ja digitehnoloogia",
            "x": -3.0,
            "y": 4.0,
            "programme_year": 2026,
            "organizer": "Tartu Ülikool",
        },
    ]
    _write_csv(
        results / "jev-semantic-all/segment-maps/segment-coordinates.csv", segment_rows
    )
    _write_csv(
        results / "talk-metadata/talks.csv",
        [
            {
                "episode_id": "episode-1",
                "title": "Õppimise tulevik",
                "organizer": "Tartu Ülikool",
                "programme_year": 2025,
                "programme_moderator": "Mari Maasikas",
                "programme_panelists": "Jüri Juurikas",
                "programme_stage": "Haridusala",
                "page_url": "https://page.invalid/one",
                "audio_url": "https://audio.invalid/one.mp3",
                "duration_seconds": 30,
            },
            {
                "episode_id": "episode-2",
                "title": "Tehisaru tööl",
                "organizer": "Tartu Ülikool",
                "programme_year": 2026,
                "programme_moderator": "",
                "programme_panelists": "",
                "programme_stage": "Digiala",
                "page_url": "https://page.invalid/two",
                "audio_url": "https://audio.invalid/two.mp3",
                "duration_seconds": 20,
            },
        ],
    )
    _write_csv(
        results / "jev-topic-analysis/year-comparison/main-topic-changes.csv",
        [
            {
                "jev_high_level_id": "education",
                "jev_high_level_name": "Haridus, õppimine ja oskused",
                "minutes_2025": 12,
                "minutes_2026": 8,
                "talks_2025": 1,
                "talks_2026": 1,
                "time_share_2025": 60,
                "time_share_2026": 40,
                "talk_coverage_2025": 100,
                "talk_coverage_2026": 50,
                "time_share_delta_pp": -20,
                "talk_coverage_delta_pp": -50,
            },
            {
                "jev_high_level_id": "ai",
                "jev_high_level_name": "Tehisintellekt ja digitehnoloogia",
                "minutes_2025": 0,
                "minutes_2026": 12,
                "talks_2025": 0,
                "talks_2026": 1,
                "time_share_2025": 0,
                "time_share_2026": 60,
                "talk_coverage_2025": 0,
                "talk_coverage_2026": 50,
                "time_share_delta_pp": 60,
                "talk_coverage_delta_pp": 50,
            },
        ],
    )
    style = {"education": {"color": "#22577a"}, "ai": {"color": "#d97706"}}
    style_path = results / "high-level-topics/category-style.json"
    style_path.parent.mkdir(parents=True, exist_ok=True)
    style_path.write_text(json.dumps(style), encoding="utf-8")

    archive_dir = root / "data/transcripts/2026"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "episode-1.json").write_text(
        json.dumps(
            {
                "episode": {"id": "episode-1"},
                "transcription": {
                    "cues": [
                        {"start_seconds": 0, "end_seconds": 6, "text": "Esimene lause."},
                        {"start_seconds": 6, "end_seconds": 16, "text": "Õppimine muutub."},
                        {"start_seconds": 16, "end_seconds": 26, "text": "Digioskused aitavad."},
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if include_second_archive:
        (archive_dir / "episode-2.json").write_text(
            json.dumps(
                {
                    "episode": {"id": "episode-2"},
                    "transcription": {
                        "cues": [
                            {"start_seconds": 0, "end_seconds": 10, "text": "AI aitab tööl."}
                        ]
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def test_normalize_search_text_removes_estonian_diacritics_and_case() -> None:
    assert normalize_search_text("TÖÖõigus ja ÄriÜhing") == "toooigus ja ariuhing"


def test_build_site_data_reconstructs_passages_and_counts_distinct_talks(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    outputs = build_site_data(tmp_path, tmp_path / "site/data")

    overview = json.loads(outputs["overview"].read_text(encoding="utf-8"))
    transcripts = json.loads(outputs["transcripts"].read_text(encoding="utf-8"))
    comparison = json.loads(outputs["comparison"].read_text(encoding="utf-8"))
    assert overview["schema_version"] == 1
    assert overview["metrics"] == {
        "categories": 2,
        "segments": 3,
        "talks": 2,
        "years": [2025, 2026],
    }
    education = next(row for row in overview["categories"] if row["id"] == "education")
    assert education["talks"] == 1
    matrix = next(
        row
        for row in overview["organizer_topics"]
        if row["organizer"] == "Tartu Ülikool" and row["category"] == "education"
    )
    assert matrix["talks"] == 1
    first = next(row for row in transcripts["segments"] if row["id"] == "s1")
    assert first["text"] == "Esimene lause. Õppimine muutub."
    assert normalize_search_text(first["text"]).startswith("esimene lause oppimine muutub")
    assert first["audio"] == "https://audio.invalid/one.mp3#t=5"
    assert comparison["categories"][0]["time_share_2025"] in {0.0, 60.0}


def test_build_site_data_reports_missing_transcript_archive(tmp_path: Path) -> None:
    _write_fixture(tmp_path, include_second_archive=False)

    with pytest.raises(FileNotFoundError, match="episode-2"):
        build_site_data(tmp_path, tmp_path / "site/data")
