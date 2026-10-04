"""Review views preserve native boundaries and safely display original text."""

import pandas as pd
import pytest

from arvamusfestivali_transcripts.topic_analysis.segment_review import (
    document_catalog,
    segment_review_html,
)


@pytest.fixture
def segments():
    rows = []
    for episode, title in [("2", "Zoo"), ("1", "Alpha")]:
        for mode in ["fixed", "semantic"]:
            rows.append(
                dict(
                    episode_id=episode,
                    talk_name=title,
                    segmentation=mode,
                    start_seconds=10.0,
                    end_seconds=100.0,
                    topic_id=0,
                    topic_name="Õpetajad",
                    topic_key=f"{mode}:0",
                    confidence=0.4,
                    text=f"{mode} <script> text",
                    audio_link="https://example.test/audio#t=10",
                    model_key=mode,
                    segment_key=f"{episode}:{mode}:a",
                )
            )
            rows.append(
                dict(
                    episode_id=episode,
                    talk_name=title,
                    segmentation=mode,
                    start_seconds=100.0,
                    end_seconds=200.0,
                    topic_id=-1,
                    topic_name="Unassigned",
                    topic_key=f"{mode}:-1",
                    confidence=None,
                    text="Noise content",
                    audio_link="https://example.test/audio#t=100",
                    model_key=mode,
                    segment_key=f"{episode}:{mode}:b",
                )
            )
    return pd.DataFrame(rows)


def test_index_selects_same_document_for_both_modes(segments):
    assert document_catalog(segments).episode_id.tolist() == ["1", "2"]
    output = segment_review_html(segments, 0)
    assert "Document 0: Alpha" in output
    assert "Zoo" not in output
    assert "Semantic segmentation" in output and "Fixed segmentation" in output
    assert "&lt;script&gt;" in output and "<script>" not in output
    assert "0 · Õpetajad" in output
    assert "10.00–100.00 seconds" in output
    assert "Membership strength: 0.400" in output
    assert "https://example.test/audio#t=10" in output


def test_mode_and_filters_affect_cards_without_mutating_assignments(segments):
    original = segments.copy(deep=True)
    output = segment_review_html(segments, 0, mode="semantic", only_unassigned=True)
    assert "Fixed segmentation" not in output
    assert "Showing 1 of 2 segments" in output
    assert '<div class="text">Noise content</div>' in output
    assert '<div class="text">semantic' not in output
    no_matches = segment_review_html(
        segments, 0, mode="fixed", weak_only=True, weak_threshold=0.2, search="Õpetajad"
    )
    assert "No segments match the filters." in no_matches
    pd.testing.assert_frame_equal(segments, original)


def test_invalid_document_index_is_rejected(segments):
    with pytest.raises(ValueError, match="index"):
        segment_review_html(segments, 2)
