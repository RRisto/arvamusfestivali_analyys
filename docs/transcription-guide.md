# Arvamusfestival transcript archive

This project builds a durable JSON transcript archive for the 2026 Arvamusfestival
podcast season. It resolves the podcast collection through the [Apple Podcasts
lookup API](https://itunes.apple.com/lookup) (collection ID `1477431807`) and then
uses the authoritative SoundCloud RSS feed returned by Apple. The RSS enclosure is
the audio source; its publication date determines the archive year.

For a reproducible selected-recording transcript repair on RunPod, including explicit episode
scope, isolated staging, checksummed partial promotion, saved-turn diarization reassignment,
and the existing-topic Jev refresh, follow
[RUNPOD_RETRANSCRIPTION.md](../RUNPOD_RETRANSCRIPTION.md).

## Results website

The static results website has three focused pages: a topic overview, a separate
2025–2026 comparison, and a transcript explorer with timestamped audio links.
Build its browser-ready data from the repository root and start a local server:

```powershell
uv run python scripts/build_results_site.py --root . --output site/data
uv run python -m http.server 8765 --directory site
```

Then open `http://localhost:8765/`. Do not open the HTML files directly: browsers
block the JSON requests used by the interactive charts and transcript search. The
site is fully static after the build; it does not call a model or application API.

## Prerequisites

Use Windows PowerShell with [uv](https://docs.astral.sh/uv/) installed and on
`PATH`. Install `ffmpeg` and make it available on `PATH`, because the bundled
Estonian audio transcriber processes MP3 audio. Its script and pinned dependencies
are in `vendor/estonian-audio-transcription/`.

```powershell
uv sync --dev
uv run arvamusfestivali-transcripts run --year 2026 --dry-run --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py
uv run arvamusfestivali-transcripts run --year 2026 --limit 1 --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py
uv run arvamusfestivali-transcripts run --year 2026 --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py
```

The first command is a dry run: it discovers the selected episodes but does not
write state, cache, or archives. The second is the one-episode acceptance run. Use
the third only after inspecting that archive. Pass `--uv-executable` when the
transcription skill must use a uv executable outside `PATH`.

## Operations

`run` downloads an episode, invokes the local transcriber, validates its output, and
writes `data/transcripts/<year>/<soundcloud-id>.json`. It resumes safely: existing
valid archives are skipped unless `--force` is supplied. Process several episodes
concurrently with `--parallelism`; each worker uses the transcriber's default four
CPU threads, so `--parallelism 4` is appropriate for a 16-vCPU machine:

```powershell
uv run arvamusfestivali-transcripts run --year 2026 --parallelism 4 --catalog-snapshot data/catalog/<timestamp>-2026.json --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py
```

The default remains `--parallelism 1`. Each worker uses an independent HTTP client,
state-database connection, and engine-output directory. Retry an episode after fixing
its cause with the same command plus `--episode-id`:

```powershell
uv run arvamusfestivali-transcripts run --year 2026 --episode-id 2400217815 --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py
```

`fetch` only builds the verified audio cache; `transcribe` only consumes verified
cached audio and a saved catalog. All commands support `--root`, `--limit`,
`--episode-id`, `--parallelism`, `--force`, and `--dry-run` as applicable. Year values
are accepted from 2019 through the current UTC year.

For a reproducible run, pass `--catalog-snapshot data/catalog/<timestamp>-2026.json`.
For `run`, this frozen snapshot must match `--year`; it prevents Apple and RSS
requests. The snapshot’s sibling XML is the raw source record. To add a later year,
run the same command with that year after its festival feed has published episodes;
the first regular run writes its matching catalog snapshot automatically.

### Stereo phase cancellation and missing speech

Some SoundCloud recordings have two nearly identical stereo channels with opposite
polarity. A normal stereo-to-mono mix then cancels much of the speech: in the affected
2026 recordings the mix was about 25 dB quieter than either individual channel. The
recognizer can consequently emit long cues containing only a few words or omit whole
sentences even though the original audio sounds normal when played in stereo.

The bundled TalTech adapter protects against this in `select_channel()`. It measures
the complete recording and keeps the mono mix normally; when the mix is at least
12 dB weaker than the stronger source channel, it transcribes that channel alone.
Do not remove this selection or unconditionally downmix stereo audio. The chosen value
is recorded as `transcription.runtime.audio_channel` (`mix`, `left`, or `right`) in
every archive.

After changing the transcriber or repairing old archives:

1. Run one affected episode with `--force` and inspect the resulting JSON before a
   batch run. A phase-cancelled recording should report `left` or `right`, not `mix`.
2. Compare word count and suspicious cues with the previous archive. A useful review
   threshold is a cue lasting at least 8 seconds with fewer than 20 words per minute;
   listen to every remaining match because pauses and applause can be legitimate.
3. Only then rerun all selected recordings and downstream diarization/topic analysis.

```powershell
uv run arvamusfestivali-transcripts run --year 2026 --episode-id 2397264564 --force --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py

$archive = Get-Content -Raw data/transcripts/2026/2397264564.json | ConvertFrom-Json
$archive.transcription.runtime.audio_channel
$archive.transcription.cues | Where-Object {
    $duration = [double]$_.end_seconds - [double]$_.start_seconds
    $words = ([regex]::Matches([string]$_.text, '\p{L}+')).Count
    $duration -ge 8 -and ($words / $duration * 60) -lt 20
} | Select-Object start_seconds, end_seconds, text
```

This check detects the systematic cancellation failure, not every ASR omission. The
TalTech large streaming model can still miss isolated words or sentences, so sparse
matches and material word-count decreases require listening rather than automatic
acceptance or rejection.

## Archive format and links

Each transcript is UTF-8 JSON with `schema_version: 1`, `episode`, and
`transcription` objects. Episode data retains the SoundCloud ID, RSS GUID, title,
publication timestamp, Apple collection ID, page/audio URLs, audio byte count and
SHA-256. Transcription data retains engine `model` and `runtime` metadata, complete
text, and timestamped cues (`start_seconds`, `end_seconds`, `text`). These canonical
transcript archives do not contain speaker assignments; speaker diarization is kept
separately under `data/diarization/<year>/`.

Construct a link to a cue with the W3C media fragment form
`<episode.audio_url>#t=<cue.start_seconds>`; for example,
`https://example.invalid/episode.mp3#t=754.2`.

Audio cache files are deleted only after a validated transcript archive is written.
On download, transcription, or archive failure, the verified cache remains for a
safe retry. Temporary partial downloads use `.part` files.

## One-episode speaker diarization pilot

The pilot uses pyannote's local `speaker-diarization-community-1` model to assign
anonymous speaker labels to the existing timestamped cues. Accept the model's terms
on Hugging Face, create an access token, and expose it only for the current shell:

```powershell
$env:HF_TOKEN = "<your-hugging-face-token>"
uv run scripts/diarize_episode.py data/transcripts/2026/2258666735.json
```

This shortest archived episode is about 6.4 minutes. The script downloads and verifies
its source audio, automatically uses CUDA when available, and writes
`data/diarization/2026/2258666735.json`. Supply an already downloaded file with
`--audio <path>`. If the participant count is known, use `--num-speakers`; otherwise
`--min-speakers` and `--max-speakers` constrain automatic speaker counting. The
canonical transcript is never modified.

Each output contains the exclusive speaker turns plus a `speaker` and `speaker_share`
for every transcript cue. Low `speaker_share` values identify cues that cross a
speaker boundary and need review. Speaker labels are anonymous and episode-local.

Build a standalone review page after diarization:

```powershell
uv run python scripts/build_diarization_review.py data/transcripts/2026/2258666735.json
uv run python -m http.server 8765 --directory site
```

Open `http://localhost:8765/diarization/2258666735.html`. Consecutive cues from the
same detected speaker are merged into cards; selecting a card seeks the fixed audio
player to that segment and starts playback. Amber cards contain at least one cue with
low speaker overlap and should be checked manually.

For the complete archive, use the resumable GPU batch runner. A five-episode
benchmark is intentionally the first step:

```bash
python scripts/diarize_all.py --limit 5 --dry-run
python scripts/diarize_all.py --limit 5
python scripts/diarize_all.py
```

The model is loaded once and reused. Valid outputs are skipped, temporary audio is
removed after each episode, failures do not stop the batch, and progress is saved in
`data/diarization/batch-status.json`. Review pages and their index are written under
`site/diarization/`. The batch requires CUDA by default; see [RUNPOD.md](RUNPOD.md)
for the Pod configuration, secure token handling, benchmark, resume, and download
steps.

### Speaker-name matching review

The 2026 name-matching review covers all 264 diarized recordings. It associates
episode-local diarization labels with names found in transcript introductions or
supported by nearby conversational context. Each suggestion includes its evidence,
confidence (`High`, `Medium`, or `Low`), and matching method. These are review
candidates, not verified identities; listen to the linked audio before treating a
match as confirmed.

The saved results are:

- [`data/diarization/speaker-name-detections-2026.csv`](../data/diarization/speaker-name-detections-2026.csv) — 464 speaker/episode entries, including 403 name suggestions and 61 entries without a clear match.
- [`site/diarization/name-matching-review.html`](../site/diarization/name-matching-review.html) — full transcript viewer for all 264 recordings, with audio playback, time-coded segments, diarized speaker labels, matched names where available, and the recording title as the topic label.

The viewer groups adjacent transcript cues from the same speaker into one row. It
aligns the grouped text with the original transcript so spaces and punctuation are
preserved, including words split across cue boundaries. The topic label is the
recording title; it is not a separate per-segment topic classification.

## Git policy

Commit the catalog snapshots in `data/catalog/` and final transcript JSON in
`data/transcripts/`. Do not commit `data/audio-cache/`, `data/engine-output/`, or
`data/state/`; these resumability and engine artifacts are intentionally ignored,
along with `*.part` download files.

## Development

```powershell
uv run pytest -q
uv run ruff check .
uv build
```
