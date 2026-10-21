# Selected 2026 transcript repair on RunPod

This is a targeted repair, not a 2026 retranscription. The checked-in selection contains
22 unique recordings whose current transcript has at least one cue longer than 30 seconds.
Nine recordings also have duplicate RSS archive IDs, so validation and promotion cover 31
JSON archives while RunPod transcribes only the 22 unique recordings. The remaining 233
archives are never staged or replaced.

Run the complete audio for all 22 selected unique recordings. Do not restrict the rerun to
episode `2397264564`, but do not expand it to the full 2026 archive either. The existing
selection already captures the recordings with long recognition gaps.

No command in this document runs transcription locally. Run the transcription section on
RunPod yourself.

## Fixed scope and model

- Catalog: `data/catalog/20260920T112540Z-2026.json`
- Selection: `data/retranscription/2026-affected-episodes.json`
- Lock: `data/retranscription/2026-run-lock.json`
- Transcriber: `vendor/estonian-audio-transcription/scripts/transcribe.py`. It streams
  the complete audio through the pinned model, checks stereo phase cancellation
  across the recording, and writes the selected channel to runtime metadata.
- Model: `TalTechNLP/streaming-zipformer-large.et-en`, revision
  `2812262bd4fd5d4327dc09ca0e6c19e8a32a03d6`, int8
- Runtime: sherpa-onnx 1.13.7, modified beam search, four threads
- Audio preparation: automatic stereo handling. Use the normal mono mix unless destructive
  phase cancellation makes it at least 12 dB weaker than an individual channel; then use the
  stronger channel. The selected mode is recorded as `runtime.audio_channel`.

The selection file is mandatory. The pipeline refuses an unscoped lock, unknown or repeated
IDs, and selecting two IDs for the same audio recording. Lock creation automatically adds all
duplicate archive IDs sharing the 22 selected audio hashes.

Recreate the lock only when the selection, catalog, baseline transcripts, or transcription
script intentionally changes:

```powershell
.\.venv\Scripts\python.exe scripts\manage_retranscription.py create-lock `
  --catalog data\catalog\20260920T112540Z-2026.json `
  --baseline data\transcripts\2026 `
  --episodes data\retranscription\2026-affected-episodes.json `
  --transcriber-script vendor\estonian-audio-transcription\scripts\transcribe.py `
  --output data\retranscription\2026-run-lock.json
```

Expected output: `31 episodes`. The lock separately records the 22
`transcribe_episode_ids` and the 31 archives that will be validated and promoted.

## Package for RunPod

Build an allow-listed archive; do not include secrets, current topic outputs, or the complete
transcript corpus.

```powershell
$handoff = Join-Path $PWD '.tmp\runpod-transcript-repair'
New-Item -ItemType Directory -Force "$handoff\project\scripts", "$handoff\project\data\catalog", "$handoff\project\data\retranscription", "$handoff\project\baseline\data\transcripts\2026", "$handoff\project\vendor" | Out-Null
Copy-Item pyproject.toml,uv.lock -Destination "$handoff\project"
Copy-Item src -Destination "$handoff\project\src" -Recurse
Copy-Item scripts\manage_retranscription.py -Destination "$handoff\project\scripts"
Copy-Item data\catalog\20260920T112540Z-2026.json -Destination "$handoff\project\data\catalog"
Copy-Item data\retranscription\2026-affected-episodes.json,data\retranscription\2026-run-lock.json -Destination "$handoff\project\data\retranscription"
$lock = Get-Content data\retranscription\2026-run-lock.json -Raw | ConvertFrom-Json
$lock.episodes.PSObject.Properties.Name | ForEach-Object { Copy-Item "data\transcripts\2026\$_.json" -Destination "$handoff\project\baseline\data\transcripts\2026" }
New-Item -ItemType Directory -Force "$handoff\project\vendor\estonian-audio-transcription\scripts" | Out-Null
Copy-Item vendor\estonian-audio-transcription\pyproject.toml,vendor\estonian-audio-transcription\uv.lock -Destination "$handoff\project\vendor\estonian-audio-transcription"
Copy-Item vendor\estonian-audio-transcription\scripts\transcribe.py -Destination "$handoff\project\vendor\estonian-audio-transcription\scripts"
Get-ChildItem $handoff -Recurse -Force | Where-Object { $_.Name -match '^\.env|token|credential|secret' }
tar -C $handoff -czf .tmp\runpod-transcript-repair.tgz project
Get-FileHash .tmp\runpod-transcript-repair.tgz -Algorithm SHA256 | Format-List
```

The secret-name scan must print nothing. Upload through an authenticated channel.

## RunPod transcription

The bundled ONNX model uses CPU. A 32-vCPU instance with 32–64 GiB RAM and persistent
storage is appropriate; an expensive GPU does not improve this model.

```bash
cd /workspace
sha256sum runpod-transcript-repair.tgz
tar -xzf runpod-transcript-repair.tgz
cd /workspace/project
sudo apt-get update && sudo apt-get install -y ffmpeg tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --no-dev
uv run vendor/estonian-audio-transcription/scripts/transcribe.py \
  --download-only --model large --precision int8
```

Run only the 22 IDs in the selection file. `--episode-list` is handled inside the normal
pipeline, so its existing parallelism, resume, cache, and state behavior remain in use:

```bash
tmux new -s transcript-repair
cd /workspace/project
set -o pipefail
uv run arvamusfestivali-transcripts run \
  --year 2026 --root /workspace/project/stage-channel-safe \
  --catalog-snapshot data/catalog/20260920T112540Z-2026.json \
  --episode-list data/retranscription/2026-affected-episodes.json \
  --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py \
  --parallelism 8 |& tee -a /workspace/project/repair-run.log
```

This uses a new stage root intentionally. Do not reuse the pre-channel-fix `stage` directory,
because valid-looking old archives would be skipped. Rerun the same command to resume this
new run; do not add `--force`.
After the 22 transcriptions finish, create the nine duplicate-ID archives from the newly
transcribed matching audio. This copies only transcription content while retaining each alias's
own frozen episode metadata:

```bash
uv run python scripts/manage_retranscription.py materialize-aliases \
  --baseline /workspace/project/baseline/data/transcripts/2026 \
  --stage stage-channel-safe/data/transcripts/2026 \
  --lock data/retranscription/2026-run-lock.json
```

The allow-listed package includes the 31 locked baseline JSON files at that path. It does not
include the other 233 transcripts.

Validate exactly the locked 31 archives and package them:

```bash
uv run python scripts/manage_retranscription.py validate \
  --catalog data/catalog/20260920T112540Z-2026.json \
  --stage stage-channel-safe/data/transcripts/2026 \
  --lock data/retranscription/2026-run-lock.json \
  --manifest stage-channel-safe/2026-repair-manifest.json --year 2026
tar -C stage-channel-safe -czf /workspace/af-2026-transcript-repair.tgz \
  data/transcripts/2026 2026-repair-manifest.json data/state/pipeline.sqlite3
sha256sum /workspace/af-2026-transcript-repair.tgz > /workspace/af-2026-transcript-repair.sha256
```

Validation must report `expected=31`, `valid=31`, and zero missing, failed, or unexpected.

## Local import and promotion

Extract into a new incoming directory and validate again. This does not transcribe anything.

```powershell
$run = 'data\retranscription\incoming-2026-runpod'
New-Item -ItemType Directory -Force $run | Out-Null
tar -C $run -xzf .tmp\af-2026-transcript-repair.tgz
.\.venv\Scripts\python.exe scripts\manage_retranscription.py validate `
  --catalog data\catalog\20260920T112540Z-2026.json `
  --stage "$run\data\transcripts\2026" `
  --lock data\retranscription\2026-run-lock.json `
  --manifest "$run\local-validation-manifest.json" --year 2026
```

Reuse existing pyannote turns only for the 31 changed archive IDs:

```powershell
.\.venv\Scripts\python.exe scripts\reassign_diarization.py `
  --old-transcripts data\transcripts\2026 `
  --new-transcripts "$run\data\transcripts\2026" `
  --saved-diarization data\diarization\2026 `
  --output "$run\data\diarization\2026" `
  --review-output "$run\site\diarization" `
  --manifest "$run\diarization-reassignment-manifest.json"
```

Require `selected=31`, `completed=31`, `failed=0`. Promote the transcript files only after
that succeeds:

```powershell
.\.venv\Scripts\python.exe scripts\manage_retranscription.py promote `
  --stage "$run\data\transcripts\2026" `
  --target data\transcripts\2026 `
  --backup data\retranscription\backups\2026-before-runpod `
  --manifest "$run\local-validation-manifest.json"
```

Promotion rechecks the staged checksums and each selected file's original locked checksum. It
moves only the 31 selected files into the backup and canonical directory. The other 233 archive
files remain byte-for-byte untouched. Promote reassigned diarization and review files with the
same selected-file discipline; do not replace their whole directories.

## Existing-topic Jev refresh

Transcript text changes invalidate semantic boundaries, embeddings, and Jev assignments. They
do **not** require topic modelling. Keep the existing 236-topic vocabulary and run:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_jev_topics.py `
  --root . --year 2026 --device auto --workers 4 --allow-paid-jev
```

The refresh fingerprints the merged canonical corpus, rebuilds semantic segments, reuses
content-addressed BGE vectors for unchanged text, computes new vectors only where text changed,
and assigns the existing Jev vocabulary. It does not fit BERTopic and does not call OpenAI topic
naming. Use `--dry-run` first to inspect the plan without Jev calls.

## Required checks

- Selection remains 22 unique recordings / 31 archive IDs.
- RunPod receives `--episode-list`; no year-wide command is used.
- All 22 selected recordings are decoded from their complete audio with automatic channel
  selection; `2397264564` must report `runtime.audio_channel` as `left` or `right`, not `mix`.
- Stage validation is 31/31 with the pinned model and unchanged audio hashes.
- Promotion backs up/replaces only those 31 JSON files.
- Saved speaker turns are reassigned only to corrected cues.
- Jev refresh uses existing topic vocabulary; no topic fit or topic renaming runs.
- Tests pass before committing and pushing.
