# Arvamusfestival transcript archive

This project builds a durable JSON transcript archive for the 2026 Arvamusfestival
podcast season. It resolves the podcast collection through the [Apple Podcasts
lookup API](https://itunes.apple.com/lookup) (collection ID `1477431807`) and then
uses the authoritative SoundCloud RSS feed returned by Apple. The RSS enclosure is
the audio source; its publication date determines the archive year.

## Prerequisites

Use Windows PowerShell with [uv](https://docs.astral.sh/uv/) installed and on
`PATH`. Install `ffmpeg` and make it available on `PATH`, because the external
Estonian audio-transcription skill processes MP3 audio. Configure the local skill
script path exactly as shown below, or replace it with the matching path on your
machine.

```powershell
uv sync --dev
uv run arvamusfestivali-transcripts run --year 2026 --dry-run --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
uv run arvamusfestivali-transcripts run --year 2026 --limit 1 --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
uv run arvamusfestivali-transcripts run --year 2026 --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
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
uv run arvamusfestivali-transcripts run --year 2026 --parallelism 4 --catalog-snapshot data/catalog/<timestamp>-2026.json --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
```

The default remains `--parallelism 1`. Each worker uses an independent HTTP client,
state-database connection, and engine-output directory. Retry an episode after fixing
its cause with the same command plus `--episode-id`:

```powershell
uv run arvamusfestivali-transcripts run --year 2026 --episode-id 2400217815 --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py
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

## Archive format and links

Each transcript is UTF-8 JSON with `schema_version: 1`, `episode`, and
`transcription` objects. Episode data retains the SoundCloud ID, RSS GUID, title,
publication timestamp, Apple collection ID, page/audio URLs, audio byte count and
SHA-256. Transcription data retains engine `model` and `runtime` metadata, complete
text, and timestamped cues (`start_seconds`, `end_seconds`, `text`). There is no
speaker diarization: cue timestamps identify time, not a speaker.

Construct a link to a cue with the W3C media fragment form
`<episode.audio_url>#t=<cue.start_seconds>`; for example,
`https://example.invalid/episode.mp3#t=754.2`.

Audio cache files are deleted only after a validated transcript archive is written.
On download, transcription, or archive failure, the verified cache remains for a
safe retry. Temporary partial downloads use `.part` files.

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

## Compare transcript embedding models

The [comparison notebook](notebooks/compare_embedding_models.ipynb) compares
Qwen3-Embedding-0.6B and BGE-M3 embeddings as inputs to BERTopic. Gemini is optional.
It uses validated archives in `data/transcripts/2026/`, deduplicates recordings by
audio SHA-256, and deterministically chooses six recordings with varied titles. Each
model receives the same timestamped passages and BERTopic settings.

Install the analysis dependencies and open the notebook from the repository root:

```powershell
uv sync --group dev --group topic-analysis
uv run --group topic-analysis jupyter lab notebooks/compare_embedding_models.ipynb
```

To compare the original fixed passages with local semantic boundaries, open the
[segmentation notebook](notebooks/compare_segmentation_modes.ipynb):

```powershell
uv run --group topic-analysis jupyter lab notebooks/compare_segmentation_modes.ipynb
```

If this checkout uses the repository-visible Windows environment created under
`data/state`, launch it with:

```powershell
$uv = "$PWD\data\state\uv-tool\bin\uv.exe"
$env:UV_PROJECT_ENVIRONMENT = "$PWD\data\state\af-topic-env"
& $uv run --group topic-analysis jupyter lab notebooks/compare_segmentation_modes.ipynb
```

The segmentation notebook retains both modes. BGE-M3 embeds non-overlapping,
cue-aligned atomic blocks and detects contextual similarity drops; the resulting
semantic passages and the fixed passages are each compared with Qwen and BGE under
controlled BERTopic settings. Atomic BGE vectors are cached separately, so boundary
threshold experiments reuse them. Segmentation is entirely local and never invokes
Gemini. Inspect the boundary-score chart on a small sample before increasing
`EPISODE_COUNT` or switching `DEVICE` to `"cuda"`.

Run the cells in order. The configuration cell exposes `YEAR`, `EPISODE_COUNT`,
`EXPLICIT_EPISODE_IDS`, `MODEL_KEYS`, `DEVICE`, `BATCH_SIZES`, `CHUNKING`, and
`TOPIC_MODEL`. To choose recordings yourself, set canonical IDs such as
`EXPLICIT_EPISODE_IDS = ("2400207762", "2400207759")`; this replaces automatic
selection. The IDs must exist in the chosen year after deduplication. You can also
omit a model from `MODEL_KEYS` to run fewer adapters. The defaults run local models on
the CPU with batch size four; expect the first run to download model weights and take
substantial time and several gigabytes of disk and memory. Use a smaller batch size
if memory is tight. A compatible GPU can be selected with `DEVICE = "cuda"`.

Gemini is skipped cleanly when `GEMINI_API_KEY` is absent. To enable it for the
current PowerShell session, enter the key without printing it:

```powershell
$env:GEMINI_API_KEY = Read-Host -MaskInput "Gemini API key"
```

Enabling Gemini sends the selected passage text to Google's API and may incur API
usage charges. Check that the transcripts are appropriate to send before setting the
key. A failed model reports a remediation hint while successful model results remain
available. Per-passage embedding files are saved under
`data/topic-analysis/cache/`; repeat runs reuse matching cached vectors. Each run
exports a manifest, passage and assignment tables, metrics, cross-model comparison,
and a manual-review CSV under `data/topic-analysis/results/<experiment-id>/`.
In the notebook's manual-review cell, add `ManualTopicReview` records to
`MANUAL_REVIEWS` with a `coherent`, `mixed`, `duplicate`, or `unclear` verdict and a
note. Rerun that cell and the export cell to preserve completed annotations.
Probability exports include the explicit topic-column mapping; HDBSCAN memberships
are not calibrated semantic confidence. Outliers have no assigned-topic probability.
Both generated directories are ignored by Git. Keep the exported manifest with any
results you share: it records model identities, configurations, audio hashes, and the
Git revision.

Local model defaults pin immutable Hugging Face snapshot commits (Qwen
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, BGE
`5617a9f61b028005a4858fdac845db406aefb181`). The manifest saves full cache configuration,
including revisions, instruction, dimensions, and adapter schema. Schema 2 invalidates
older vectors because both cache keys and inference now use NFC text with collapsed
whitespace; original transcript text remains available for inspection.

The metric table is descriptive. Silhouette uses Euclidean distance on the actual
UMAP coordinates supplied to HDBSCAN, separately from the two-dimensional display
projection. It excludes outliers and may be unavailable
with fewer than two non-outlier topics; topic count and outlier fraction can change
with clustering settings. Separate UMAP maps are **not geometrically aligned**, so
compare assignments and source passages rather than plot coordinates. Use the
timestamped audio links in the representative, disputed, and low-membership boundary
tables. Disputes compare which passages are clustered together, independent of topic
numbering; two noise points are not a shared cluster. Then fill
the notebook's manual review annotations before judging coherence.
