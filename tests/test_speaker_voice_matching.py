from __future__ import annotations

import numpy as np

from arvamusfestivali_transcripts.speaker_voice_matching import (
    rank_named_candidates,
    select_speaker_cues,
)


def test_rank_named_candidates_uses_normalized_centroids() -> None:
    ranked = rank_named_candidates(
        np.array([1.0, 0.0]),
        {
            "Mari Mets": [np.array([1.0, 0.1]), np.array([1.0, -0.1])],
            "Jaan Tamm": [np.array([0.0, 1.0])],
        },
    )

    assert ranked[0]["name"] == "Mari Mets"
    assert ranked[0]["similarity"] > ranked[1]["similarity"]


def test_select_speaker_cues_drops_short_and_overlapped_cues() -> None:
    cues = [
        {"speaker": "SPEAKER_00", "start_seconds": 0, "end_seconds": 1.0, "speaker_share": 1.0},
        {"speaker": "SPEAKER_00", "start_seconds": 5, "end_seconds": 12, "speaker_share": 0.55},
        {"speaker": "SPEAKER_00", "start_seconds": 20, "end_seconds": 27, "speaker_share": 0.98},
        {"speaker": "SPEAKER_00", "start_seconds": 40, "end_seconds": 47, "speaker_share": 1.0},
    ]

    selected = select_speaker_cues(cues, min_seconds=2.5, max_per_speaker=1)

    assert len(selected) == 1
    assert selected[0]["start_seconds"] == 20
