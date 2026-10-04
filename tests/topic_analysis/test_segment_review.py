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


def test_jev_sidecar_displays_separate_scores_without_changing_cluster(segments):
    import json

    segments["jev_topic_id"] = float("nan")
    mask = (segments.episode_id == "1") & (segments.segmentation == "semantic")
    segments.loc[mask, "jev_topic_id"] = 0
    segments.loc[mask, "jev_topic_name"] = "Teacher <support>"
    segments.loc[mask, "jev_probability"] = 0.8
    segments.loc[mask, "jev_confidence"] = 0.6
    segments.loc[mask, "jev_model"] = "jev-1.13.0"
    segments.loc[mask, "jev_probabilities"] = json.dumps({"0": 0.8, "-1": 0.2})
    segments.loc[mask, "jev_high_level_name"] = "Haridus <parent>"
    original = segments.copy(deep=True)
    output = segment_review_html(segments, 0, mode="semantic")
    assert "Teacher &lt;support&gt;" in output
    assert "Choice probability: 80.0%" in output
    assert "Jev confidence: 0.600" in output
    assert "Membership strength: 0.400" in output
    assert "High-level category (dictionary): Haridus &lt;parent&gt;" in output
    assert "Top alternatives:" in output
    pd.testing.assert_frame_equal(segments, original)
    assert "Jev: not sampled" in segment_review_html(segments, 1, mode="semantic")


def test_shorter_comparison_keeps_overlaps_distinct_from_predictions(segments):
    import json

    from arvamusfestivali_transcripts.topic_analysis.segment_review import (
        semantic_length_review_html,
    )

    short = pd.DataFrame(
        [
            dict(
                episode_id="1",
                start_seconds=10.0,
                end_seconds=80.0,
                duration_seconds=70.0,
                segment_key="short:1:a",
                text="Short <script> example",
                cue_count=3,
                audio_link="https://example.test/audio#t=10",
                jev_topic_id=0,
                jev_topic_key="semantic:0",
                jev_topic_name="Õpetajad",
                jev_probability=0.8,
                jev_confidence=0.6,
                jev_model="jev-1.13.0",
                jev_probabilities=json.dumps({"0": 0.8, "-1": 0.2}),
                original_segment_overlaps=json.dumps(
                    [
                        {"segment_key": "1:semantic:a", "overlap_seconds": 70.0},
                    ]
                ),
            )
        ]
    )
    original = short.copy(deep=True)
    output = semantic_length_review_html(segments, short, 0)
    assert "Shorter semantic segments" in output
    assert "Short &lt;script&gt; example" in output
    assert "Original segment 1: Õpetajad (70s overlap)" in output
    assert "Shorter segments have Jev predictions only" in output
    pd.testing.assert_frame_equal(short, original)
    assert "outside the shorter-segment pilot" in semantic_length_review_html(segments, short, 1)
    with pytest.raises(ValueError, match="index"):
        semantic_length_review_html(segments, short, 2)


def test_experiment_pair_uses_same_document_and_time_window(segments):
    from arvamusfestivali_transcripts.topic_analysis.segment_review import (
        segmentation_experiment_review_html,
    )

    variants = {"original": segments, "sentence5": segments}
    output = segmentation_experiment_review_html(
        segments,
        variants,
        0,
        left="original",
        right="sentence5",
        start_seconds=150,
        end_seconds=180,
    )
    assert "Document 0: Alpha" in output
    assert "Sentence cuts · 5 minutes" in output
    assert "Noise content" in output
    assert "semantic &lt;script&gt; text" not in output
    assert "Zoo" not in output
    with pytest.raises(ValueError, match="Time window"):
        segmentation_experiment_review_html(
            segments,
            variants,
            0,
            left="original",
            right="sentence5",
            start_seconds=180,
            end_seconds=150,
        )


def test_refit_labels_are_run_scoped_and_separate_from_jev(segments):
    from arvamusfestivali_transcripts.topic_analysis.segment_review import (
        segmentation_experiment_review_html,
    )

    data = segments.copy()
    data["cluster_model_key"] = "fresh-fit:42"
    data["cluster_topic_id"] = 8
    data["cluster_topic_key"] = "fresh-fit:42:8"
    data["cluster_topic_name"] = "Fresh <topic>"
    data["cluster_membership_strength"] = 0.7
    variants = {"sentence3": data, "sentence5": data}
    rendered = segmentation_experiment_review_html(
        segments,
        variants,
        0,
        left="sentence3",
        right="sentence5",
        assignment_source="topic_model",
    )
    assert "Topic model: 8 · Fresh &lt;topic&gt;" in rendered
    assert "Cluster membership strength: 0.700" in rendered
    assert "Fit: fresh-fit:42" in rendered
    assert "Jev: not sampled" not in rendered
    rendered = segmentation_experiment_review_html(
        segments,
        variants,
        0,
        left="sentence3",
        right="sentence5",
        assignment_source="jev",
    )
    assert "Topic model: 8" not in rendered
