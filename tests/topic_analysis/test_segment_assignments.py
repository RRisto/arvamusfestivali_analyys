"""Topic naming preserves provenance; overlaps retain their original confidence semantics."""

import json

import pandas as pd
import pytest

from arvamusfestivali_transcripts.topic_analysis.segment_assignments import (
    link_overlapping_segments,
    name_segment_topics,
)


def sample_assignments():
    return pd.DataFrame(
        [
            dict(
                passage_id="a",
                episode_id="talk",
                title="Discussion",
                start_seconds=0,
                end_seconds=20,
                topic_id=0,
                membership_strength=0.8,
                text="õpilaste töö",
                audio_link="https://example.org/audio#t=0",
            ),
            dict(
                passage_id="b",
                episode_id="talk",
                title="Discussion",
                start_seconds=20,
                end_seconds=30,
                topic_id=-1,
                membership_strength=None,
                text="other",
                audio_link="https://example.org/audio#t=20",
            ),
        ]
    )


def test_labels_confidence_and_provenance():
    original = sample_assignments()
    named = name_segment_topics(
        original,
        pd.DataFrame({"Topic": [0, -1], "Name": ["0_töö_õpilased", "-1_misc"]}),
        model_key="semantic:bge:leaf-local-6-2:42",
    )
    for column in original.columns:
        pd.testing.assert_series_equal(named[column], original[column])
    assert named.topic_name.tolist() == ["töö, õpilased", "Unassigned"]
    assert named.confidence.iloc[0] == 0.8
    assert pd.isna(named.confidence.iloc[1])
    assert named.talk_name.tolist() == ["Discussion", "Discussion"]


def test_overlap_never_links_other_talks_or_touching_boundaries():
    topics = pd.DataFrame({"Topic": [0, -1], "Name": ["0_work", "-1_misc"]})
    target = name_segment_topics(sample_assignments(), topics, model_key="semantic")
    other = sample_assignments().iloc[[0]].copy()
    other["start_seconds"], other["end_seconds"] = 10, 20
    other["membership_strength"] = 0.4
    wrong_talk = other.copy()
    wrong_talk["episode_id"], wrong_talk["passage_id"] = "another", "c"
    other = name_segment_topics(pd.concat([other, wrong_talk]), topics, model_key="fixed")
    linked = link_overlapping_segments(target, other)
    first = json.loads(linked.other_model_overlaps.iloc[0])
    assert len(first) == 1
    assert first[0]["overlap_seconds"] == 10
    assert first[0]["fraction_of_target_segment"] == 0.5
    assert first[0]["confidence"] == 0.4
    assert linked.confidence.iloc[0] == 0.8
    assert json.loads(linked.other_model_overlaps.iloc[1]) == []


def test_unknown_topics_are_rejected():
    with pytest.raises(ValueError, match="unknown topic"):
        name_segment_topics(
            sample_assignments(),
            pd.DataFrame({"Topic": [-1], "Name": ["-1_misc"]}),
            model_key="semantic",
        )
