"""Sentence snapping preserves cues and avoids arbitrary or overly distant cuts."""

from dataclasses import replace
from pathlib import Path

import pytest

from arvamusfestivali_transcripts.topic_analysis.semantic_segmentation import _passage
from arvamusfestivali_transcripts.topic_analysis.sentence_boundaries import (
    is_sentence_ending,
    snap_sentence_boundaries,
)
from arvamusfestivali_transcripts.topic_analysis.types import CanonicalEpisode, Cue


def episode(texts):
    return CanonicalEpisode(
        episode_id="1",
        duplicate_episode_ids=("1",),
        title="Example",
        published_at="2026-01-01T00:00:00Z",
        audio_sha256="a" * 64,
        duration_seconds=len(texts) * 10,
        audio_url="https://example.test/audio",
        cues=tuple(Cue(i * 10, (i + 1) * 10, text) for i, text in enumerate(texts)),
        source_paths=(Path("1.json"),),
    )


def test_moves_cut_to_complete_sentence_without_losing_cues():
    ep = episode(
        [
            "start",
            "middle",
            "continues",
            "story",
            "more",
            "needs",
            "a card.",
            "New story",
            "continues",
            "done.",
        ]
    )
    proposed = (_passage(ep, range(6)), _passage(ep, range(6, 10)))
    result, diagnostics = snap_sentence_boundaries(
        ep,
        proposed,
        maximum_seconds=90,
        window_seconds=20,
        minimum_seconds=20,
    )
    assert result[0].end_seconds == 70
    assert result[1].start_seconds == 70
    assert diagnostics[0]["sentence_ending"]
    assert " ".join(p.text for p in result) == " ".join(c.text for c in ep.cues)
    assert sum(p.cue_count for p in result) == len(ep.cues)


def test_fallback_retains_cut_when_no_eligible_sentence():
    ep = episode(["still talking"] * 10)
    proposed = (_passage(ep, range(5)), _passage(ep, range(5, 10)))
    result, diagnostics = snap_sentence_boundaries(ep, proposed, maximum_seconds=80)
    assert result == proposed
    assert diagnostics[0]["cut_reason"] == "no_eligible_sentence_ending"


@pytest.mark.parametrize("text", ["Finished.", "Question?", "Done! ”", "Valmis…"])
def test_recognises_sentence_punctuation(text):
    assert is_sentence_ending(text)


@pytest.mark.parametrize("text", ["not finished", "näiteks nt.", "dr.", "jne."])
def test_excludes_common_abbreviations_and_fragments(text):
    assert not is_sentence_ending(text)


def test_rejects_cuts_inside_source_cues():
    ep = episode(["text"] * 10)
    proposed = (_passage(ep, range(5)), replace(_passage(ep, range(5, 10)), start_seconds=51))
    with pytest.raises(ValueError, match="align"):
        snap_sentence_boundaries(ep, proposed, maximum_seconds=80)
