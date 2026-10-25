# Multi-reference Speaker Matching Implementation Plan

> **For agentic workers:** Execute this plan in the current workspace. The user asked to plan and implement the multi-reference matching improvement.

**Goal:** Improve unnamed-speaker candidate rankings by comparing each candidate against up to three independent, already named reference recordings and rerunning every eligible discussion.

**Architecture:** Keep the current WavLM embeddings and cosine ranking. For each programme-listed candidate, select up to three longest independent reference recordings, excluding the target discussion and duplicate recordings; average their normalized embeddings as the existing ranking code already does. Exclude prior voice-match assignments from reference profiles to prevent circular self-confirmation.

**Tech Stack:** Python, existing WavLM/Transformers pipeline, CSV/JSON outputs.

**Spec:** `docs/speaker-voice-matching.md`

## Global Constraints

- Keep the existing 10-minute target-speaker floor and up-to-six clean windows per speaker.
- Use only names on the discussion's programme roster as candidates.
- Exclude duplicate recordings and the target recording from reference selection.
- A cosine score ranks candidates; it is not a calibrated probability.
- Keep generated matches as `manual_review_required`; do not automatically assign new names.

## Review Focus

- No independent references after duplicate filtering: omit that candidate and discussion from the eligible set.
- A prior voice-match label could feed back as reference evidence: exclude that method from reference profiles.
- Cached embedding metadata must report selected windows even on cache hits.
- `--all-eligible` must include all eligible discussions without changing the one-discussion default.
- Reference details in the output must identify every recording used for each candidate.

---

### Task 1: Use multiple independent references per candidate

**Files:**
- Modify: `scripts/match_speakers_by_voice.py`
- Modify: `src/arvamusfestivali_transcripts/speaker_voice_matching.py` only if the current centroid aggregation proves insufficient.

**Interfaces:**
- `_candidate_reference_names(...)` becomes `_candidate_references_for_names(..., max_recordings_per_name: int) -> dict[str, list[tuple[str, str, float]]]`.
- `run(..., max_reference_recordings: int = 3, ...)` passes the cap through candidate selection.
- Cached `_speaker_embedding` returns the number of selected windows on cache hits as well as fresh computations.

- [x] Select up to three references per candidate, from distinct duplicate groups and outside the target recording's duplicate group.
- [x] Collect each reference embedding and rank against the existing normalized centroid aggregation.
- [x] Preserve each reference recording and its window count in the candidate CSV; keep `manual_review_required`.
- [x] Exclude `cross_recording_voice_match` rows from the reference-profile index to avoid circular validation.

### Task 2: Run every eligible discussion and document the command

**Files:**
- Modify: `scripts/match_speakers_by_voice.py`
- Modify: `docs/speaker-voice-matching.md`

- [x] Add `--all-eligible` while preserving the current default pilot behavior.
- [x] Add `--max-reference-recordings` with a default of 3.
- [x] Document the full-run command, including all target speaker labels.
- [x] Run against all eligible discussions and save rankings to `data/diarization/speaker-voice-match-candidates.csv`.

### Task 3: Inspect the full-run result

**Files:**
- Output: `data/diarization/speaker-voice-match-candidates.csv`

- [x] Confirm the run completes and includes all eligible discussions and all target labels.
- [x] Summarize discussion count, target-speaker count, reference coverage, and unresolved limits.
- [x] Leave new matches unassigned for review.

## Execution Result

- The updated pipeline completed with `--all-eligible --max-reference-recordings 3`.
- It ranked 168 unnamed speaker labels across 69 eligible discussions, producing 209 candidate rows.
- Fifty target labels used at least two independent reference recordings (up to three).
- All candidates remain marked `manual_review_required`; prior voice-match assignments were excluded from reference profiles.
