"""Snap semantic boundaries to nearby sentence endings without changing transcript cues."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .semantic_segmentation import _passage
from .types import CanonicalEpisode, Passage

_ENDING = re.compile(r"[.!?…][\s\"\'»”’)]*$")
_ABBREVIATION = re.compile(r"\b(?:nt|jne|jms|jm|s\.t|s\.o|dr|hr|pr)\.$", re.IGNORECASE)


def is_sentence_ending(text: str) -> bool:
    """Punctuation heuristic, not a guarantee of a complete spoken thought."""
    value = text.strip()
    return bool(_ENDING.search(value)) and not bool(_ABBREVIATION.search(value))


def snap_sentence_boundaries(
    episode: CanonicalEpisode,
    passages: Sequence[Passage],
    *,
    maximum_seconds: float,
    window_seconds: float = 20,
    minimum_seconds: float = 45,
) -> tuple[tuple[Passage, ...], list[dict]]:
    """Keep each semantic cut or snap within ±window, respecting soft duration bounds."""
    if maximum_seconds <= 0 or window_seconds < 0 or minimum_seconds <= 0:
        raise ValueError("Duration bounds must be positive and the window nonnegative")
    if not passages or any(p.episode_id != episode.episode_id for p in passages):
        raise ValueError("Passages must belong to the selected episode")
    original_cuts = [p.start_seconds for p in passages[1:]]
    cue_indices = {cue.start_seconds: i for i, cue in enumerate(episode.cues)}
    if any(cut not in cue_indices for cut in original_cuts):
        raise ValueError("Proposed cuts must align to source cues")
    sentence_cuts = [
        cue.start_seconds
        for i, cue in enumerate(episode.cues)
        if i and is_sentence_ending(episode.cues[i - 1].text)
    ]
    final_cuts, diagnostics = [], []
    start, end = episode.cues[0].start_seconds, episode.cues[-1].end_seconds
    for number, cut in enumerate(original_cuts):
        left = final_cuts[-1] if final_cuts else start
        next_cut = original_cuts[number + 1] if number + 1 < len(original_cuts) else end
        eligible = [
            candidate
            for candidate in sentence_cuts
            if abs(candidate - cut) <= window_seconds
            and candidate - left >= minimum_seconds
            and next_cut - candidate >= minimum_seconds
            and candidate - left <= maximum_seconds + window_seconds
            # A later cut can correct the next interval; the final tail cannot be corrected.
            and (
                number + 1 < len(original_cuts)
                or end - candidate <= maximum_seconds + window_seconds
            )
        ]
        chosen = (
            min(eligible, key=lambda candidate: (abs(candidate - cut), candidate))
            if eligible
            else cut
        )
        final_cuts.append(chosen)
        previous_text = episode.cues[cue_indices[chosen] - 1].text
        diagnostics.append(
            {
                "episode_id": episode.episode_id,
                "proposed_seconds": cut,
                "selected_seconds": chosen,
                "shift_seconds": chosen - cut,
                "sentence_ending": is_sentence_ending(previous_text),
                "cut_reason": "sentence_ending" if eligible else "no_eligible_sentence_ending",
                "left_cue_text": previous_text,
                "right_cue_text": episode.cues[cue_indices[chosen]].text,
            }
        )
    indices = [0, *[cue_indices[cut] for cut in final_cuts], len(episode.cues)]
    result = tuple(
        _passage(episode, range(left, right)) for left, right in zip(indices, indices[1:])
    )
    if any(
        p.end_seconds - p.start_seconds > maximum_seconds + window_seconds + 1e-6 for p in result
    ):
        raise ValueError("Snapping exceeded the soft duration bound")
    return result, diagnostics
