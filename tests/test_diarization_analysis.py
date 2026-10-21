import json

import pytest

from arvamusfestivali_transcripts.diarization_analysis import (
    align_topic_chunks_to_speakers,
    load_conversation_summaries,
    participation_metrics,
    primary_topic_summary,
    speaker_duration_rows,
    speaker_transition_rows,
    summarize_conversation,
)


def test_summarize_conversation_calculates_duration_concentration_metrics():
    payload = {
        "episode_id": "example",
        "turns": [
            {"start_seconds": 0.0, "end_seconds": 6.0, "speaker": "A"},
            {"start_seconds": 6.0, "end_seconds": 9.0, "speaker": "B"},
            {"start_seconds": 9.0, "end_seconds": 10.0, "speaker": "C"},
        ],
    }

    summary = summarize_conversation(payload)

    assert summary == {
        "episode_id": "example",
        "segment_count": 3,
        "valid_segment_count": 3,
        "invalid_segment_count": 0,
        "short_segment_count": 0,
        "text_cue_count": 0,
        "discarded_text_cue_count": 0,
        "unique_speakers": 3,
        "total_speech_seconds": 10.0,
        "median_segment_seconds": 3.0,
        "dominant_speaker_share": pytest.approx(0.6),
        "top_two_speaker_share": pytest.approx(0.9),
        "normalized_hhi": pytest.approx(0.19),
        "gini": pytest.approx(1 / 3),
        "effective_speaker_count": pytest.approx(1 / 0.46),
    }


def test_summarize_conversation_reports_invalid_and_short_segments():
    payload = {
        "episode_id": "quality",
        "turns": [
            {"start_seconds": 1.0, "end_seconds": 1.0, "speaker": "A"},
            {"start_seconds": 2.0, "end_seconds": 1.0, "speaker": "B"},
            {"start_seconds": 3.0, "end_seconds": 3.1, "speaker": "A"},
            {"start_seconds": "bad", "end_seconds": 4.0, "speaker": "C"},
        ],
    }

    summary = summarize_conversation(payload, short_segment_threshold=0.25)

    assert summary["segment_count"] == 4
    assert summary["valid_segment_count"] == 1
    assert summary["invalid_segment_count"] == 3
    assert summary["short_segment_count"] == 1
    assert summary["unique_speakers"] == 1
    assert summary["dominant_speaker_share"] == pytest.approx(1.0)
    assert summary["normalized_hhi"] == pytest.approx(1.0)


def test_summarize_conversation_reports_meaningful_and_discarded_text_cues():
    payload = {
        "episode_id": "cue-quality",
        "turns": [
            {"start_seconds": 0.0, "end_seconds": 2.0, "speaker": "A"},
        ],
        "cues": [
            {"start_seconds": 0.0, "end_seconds": 0.5, "text": "..."},
            {"start_seconds": 0.5, "end_seconds": 1.0, "text": "  —?!  "},
            {"start_seconds": 1.0, "end_seconds": 1.5, "text": "Tere!"},
            {"start_seconds": 1.5, "end_seconds": 2.0, "text": "42%"},
        ],
    }

    summary = summarize_conversation(payload)

    assert summary["segment_count"] == 1
    assert summary["text_cue_count"] == 2
    assert summary["discarded_text_cue_count"] == 2


def test_speaker_duration_rows_returns_shares_sorted_largest_first():
    payload = {
        "episode_id": "shares",
        "turns": [
            {"start_seconds": 0.0, "end_seconds": 2.0, "speaker": "B"},
            {"start_seconds": 2.0, "end_seconds": 8.0, "speaker": "A"},
            {"start_seconds": 8.0, "end_seconds": 10.0, "speaker": "B"},
        ],
    }

    rows = speaker_duration_rows(payload)

    assert rows == [
        {
            "episode_id": "shares",
            "speaker": "A",
            "speech_seconds": 6.0,
            "segment_count": 1,
            "speech_share": pytest.approx(0.6),
            "speaker_rank": 1,
        },
        {
            "episode_id": "shares",
            "speaker": "B",
            "speech_seconds": 4.0,
            "segment_count": 2,
            "speech_share": pytest.approx(0.4),
            "speaker_rank": 2,
        },
    ]


def test_load_conversation_summaries_reads_numeric_json_files(tmp_path):
    episode = {
        "episode_id": "123",
        "turns": [{"start_seconds": 0.0, "end_seconds": 1.0, "speaker": "A"}],
    }
    (tmp_path / "123.json").write_text(json.dumps(episode), encoding="utf-8")
    (tmp_path / "batch-status.json").write_text("{}", encoding="utf-8")

    summaries, speaker_rows = load_conversation_summaries(tmp_path)

    assert [row["episode_id"] for row in summaries] == ["123"]
    assert [row["episode_id"] for row in speaker_rows] == ["123"]


def test_primary_topic_summary_weights_topics_by_segment_duration():
    rows = [
        {"category_id": "health", "category": "Health", "duration_seconds": 4.0},
        {"category_id": "education", "category": "Education", "duration_seconds": 3.0},
        {"category_id": "health", "category": "Health", "duration_seconds": 2.0},
    ]

    summary = primary_topic_summary(
        rows,
        topic_id_key="category_id",
        topic_name_key="category",
    )

    assert summary == {
        "primary_topic_id": "health",
        "primary_topic_name": "Health",
        "primary_topic_share": pytest.approx(2 / 3),
        "topic_count": 2,
        "topic_duration_seconds": pytest.approx(9.0),
    }


def test_align_topic_chunks_allocates_exact_overlap_to_each_speaker():
    chunks = [
        {
            "start_seconds": 2.0,
            "end_seconds": 8.0,
            "high_level_id": "health",
            "high_level_name": "Health",
        }
    ]
    turns = [
        {"start_seconds": 0.0, "end_seconds": 5.0, "speaker": "A"},
        {"start_seconds": 4.0, "end_seconds": 10.0, "speaker": "B"},
        {"start_seconds": 10.0, "end_seconds": 12.0, "speaker": "C"},
    ]

    rows = align_topic_chunks_to_speakers(chunks, turns)

    assert rows == [
        {
            "topic_chunk_index": 0,
            "chunk_start_seconds": 2.0,
            "chunk_end_seconds": 8.0,
            "high_level_id": "health",
            "high_level_name": "Health",
            "speaker": "A",
            "overlap_seconds": 3.0,
        },
        {
            "topic_chunk_index": 0,
            "chunk_start_seconds": 2.0,
            "chunk_end_seconds": 8.0,
            "high_level_id": "health",
            "high_level_name": "Health",
            "speaker": "B",
            "overlap_seconds": 4.0,
        },
    ]


def test_participation_metrics_measure_duration_equality():
    metrics = participation_metrics({"A": 6.0, "B": 3.0, "C": 1.0})

    assert metrics == {
        "unique_speakers": 3,
        "total_speech_seconds": 10.0,
        "dominant_speaker_share": pytest.approx(0.6),
        "top_two_speaker_share": pytest.approx(0.9),
        "normalized_hhi": pytest.approx(0.19),
        "gini": pytest.approx(1 / 3),
        "effective_speaker_count": pytest.approx(1 / 0.46),
    }


def test_speaker_transition_rows_collapses_adjacent_same_speaker_turns():
    turns = [
        {"start_seconds": 0.0, "end_seconds": 1.0, "speaker": "A"},
        {"start_seconds": 1.1, "end_seconds": 2.0, "speaker": "A"},
        {"start_seconds": 2.1, "end_seconds": 3.0, "speaker": "B"},
        {"start_seconds": 3.1, "end_seconds": 4.0, "speaker": "B"},
        {"start_seconds": 4.1, "end_seconds": 5.0, "speaker": "C"},
        {"start_seconds": 5.1, "end_seconds": 6.0, "speaker": "A"},
    ]

    rows = speaker_transition_rows(turns)

    assert rows == [
        {"from_speaker": "A", "to_speaker": "B", "transition_count": 1},
        {"from_speaker": "B", "to_speaker": "C", "transition_count": 1},
        {"from_speaker": "C", "to_speaker": "A", "transition_count": 1},
    ]
