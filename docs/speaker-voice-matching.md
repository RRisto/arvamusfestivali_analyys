# Cross-recording speaker voice matching

This experiment ranks unnamed speakers against people listed in the same
discussion's official programme, using speaker embeddings from the
`microsoft/wavlm-base-plus-sv` model. It only considers speakers with at least
10 minutes of speech by default and uses up to six clean diarized speech windows
per speaker. Programme names are candidates only when that person also has a
named speaker profile in a different recording. Each candidate profile uses
up to three independent reference recordings by default; duplicate recordings
and the target discussion are excluded. Their normalized voice embeddings are
combined before ranking. Earlier voice-match suggestions are never reused as
reference profiles, to avoid circular confirmation.

The command downloads the selected MP3s into the ignored `data/audio-cache/`
directory and caches embeddings under the ignored
`data/diarization/speaker-voice-cache/` directory. It writes ranked results to
`data/diarization/speaker-voice-match-candidates.csv`. It does not change speaker
assignments: every suggested match needs manual review. Cosine similarity is a
ranking signal, not a calibrated probability or a universal acceptance threshold.

## Setup and run

Install the optional dependencies and ensure `ffmpeg` is on `PATH`:

```powershell
uv sync --extra speaker-match
```

Run one known discussion as a pilot:

```powershell
uv run python scripts/match_speakers_by_voice.py --episode-id 2277338249 --max-target-speakers 2 --max-reference-names 3
```

To let the pipeline select a discussion with long unnamed speakers and known
programme participants from other recordings, omit `--episode-id`:

```powershell
uv run python scripts/match_speakers_by_voice.py
```

To process every eligible discussion and every eligible unnamed speaker, using
up to three reference recordings per candidate:

```powershell
uv run python scripts/match_speakers_by_voice.py --all-eligible
```

Use `--max-reference-recordings N` to change the reference cap. The output
records each reference episode, speaker label, selected window count, and the
number of reference recordings used for each candidate.

The implementation uses Microsoft's WavLM speaker-verification checkpoint via
Transformers. The checkpoint is trained for speaker verification and its model
card describes cosine similarity as a retrieval signal; its threshold is
dataset-dependent. Festival audio and Estonian speech may differ from the
training domain, so scores must be checked against the transcript and programme
before any name is assigned.
