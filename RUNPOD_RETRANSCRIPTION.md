# 2026 full retranscription on RunPod

This runbook retranscribes the frozen 264-episode, 385.981-hour 2026 catalog into an isolated staging root. Nothing writes to 'data/transcripts/2026' until the operator runs the validated promotion command. The prior archive is retained as a directory backup.

## Pinned inputs and hardware

- Catalog: 'data/catalog/20260920T112540Z-2026.json'. Its checksum and all 264 baseline audio SHA-256 values are pinned in 'data/retranscription/2026-run-lock.json'.
- Integration: package the exact 'estonian-audio-transcription' skill used to create the lock. The lock records the transcription script checksum.
- Model: 'TalTechNLP/streaming-zipformer-large.et-en', revision '2812262bd4fd5d4327dc09ca0e6c19e8a32a03d6', int8.
- Runtime: sherpa-onnx 1.13.7, modified beam search, four threads per episode.

The bundled script explicitly selects the ONNX CPU provider. A CUDA GPU does not accelerate it. Prefer a CPU Pod/CPU instance with **32 vCPU, 32–64 GiB RAM, and at least 100 GiB persistent disk**. Start at parallelism eight. Use 64 GiB on shared/burstable-memory providers and reduce parallelism if the benchmark swaps. If a GPU-backed Pod is mandatory, pick the cheapest GPU; it will remain unused. An A100/H100 brings no benefit.

Use Ubuntu 22.04/24.04, Python 3.12, ffmpeg, tar, sqlite3, tmux, and uv. Persistent '/workspace' storage is required.

## 1. Work done locally before upload

Run focused tests and recreate the lock only when an input intentionally changes:

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retranscription.py tests/test_diarization_reassign.py -q --basetemp .tmp\pytest-retranscription
.\.venv\Scripts\python.exe scripts\manage_retranscription.py create-lock `
  --catalog data\catalog\20260920T112540Z-2026.json `
  --baseline data\transcripts\2026 `
  --transcriber-script C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription\scripts\transcribe.py `
  --output data\retranscription\2026-run-lock.json
~~~

Build an allow-listed package. This excludes '.env', API tokens, old transcripts, topic outputs, audio, logs, Git metadata, and the dirty site tree.

~~~powershell
$handoff = Join-Path $PWD '.tmp\runpod-retranscription'
New-Item -ItemType Directory -Force "$handoff\project\scripts", "$handoff\project\data\catalog", "$handoff\project\data\retranscription", "$handoff\project\vendor" | Out-Null
Copy-Item pyproject.toml,uv.lock -Destination "$handoff\project"
Copy-Item src -Destination "$handoff\project\src" -Recurse
Copy-Item scripts\manage_retranscription.py -Destination "$handoff\project\scripts"
Copy-Item data\catalog\20260920T112540Z-2026.json -Destination "$handoff\project\data\catalog"
Copy-Item data\retranscription\2026-run-lock.json -Destination "$handoff\project\data\retranscription"
Copy-Item C:\Users\risto\projects\skils_plugins\plugins\estonian-audio-transcription\skills\estonian-audio-transcription -Destination "$handoff\project\vendor\estonian-audio-transcription" -Recurse
Get-ChildItem $handoff -Recurse -Force | Where-Object { $_.Name -match '^\.env|token|credential|secret' }
tar -C $handoff -czf .tmp\runpod-retranscription.tgz project
tar -tzf .tmp\runpod-retranscription.tgz
Get-FileHash .tmp\runpod-retranscription.tgz -Algorithm SHA256 | Format-List
~~~

The secret-name scan must print nothing. Upload through RunPod's authenticated file/volume interface or scp, never a public bucket. Record the local checksum.

## 2. Commands run on RunPod

~~~bash
cd /workspace
sha256sum runpod-retranscription.tgz  # compare with local Get-FileHash
tar -xzf runpod-retranscription.tgz
cd /workspace/project
sudo apt-get update
sudo apt-get install -y ffmpeg tmux sqlite3
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --no-dev
uv run vendor/estonian-audio-transcription/scripts/transcribe.py \
  --download-only --model large --precision int8
~~~

Do not set OPENAI_API_KEY, GEMINI_API_KEY, HF_TOKEN, or another cloud-model credential. This run uses public audio and the pinned local ONNX model only.

### One-episode acceptance

Episode 2258666735 is the 383-second shortest recording:

~~~bash
mkdir -p /workspace/project/stage
uv run arvamusfestivali-transcripts run \
  --year 2026 --root /workspace/project/stage \
  --catalog-snapshot /workspace/project/data/catalog/20260920T112540Z-2026.json \
  --transcriber-script /workspace/project/vendor/estonian-audio-transcription/scripts/transcribe.py \
  --episode-id 2258666735 --parallelism 1 |& tee /workspace/project/acceptance.log
uv run python scripts/manage_retranscription.py status \
  --catalog data/catalog/20260920T112540Z-2026.json \
  --stage stage/data/transcripts/2026 \
  --lock data/retranscription/2026-run-lock.json --year 2026 || true
~~~

The pipeline must report one completion and zero failures. Status intentionally exits 1 until complete; now it should show one valid and 263 missing, with no validation error for 2258666735. Inspect its nonempty text, ordered cues, audio hash, and model revision.

### Small throughput benchmark

These three 48–60 minute episodes are realistic and remain in the resumable stage:

~~~bash
for id in 2310594914 2302899440 2400164916; do
  /usr/bin/time -v uv run arvamusfestivali-transcripts run \
    --year 2026 --root /workspace/project/stage \
    --catalog-snapshot /workspace/project/data/catalog/20260920T112540Z-2026.json \
    --transcriber-script /workspace/project/vendor/estonian-audio-transcription/scripts/transcribe.py \
    --episode-id "$id" --parallelism 1
done |& tee /workspace/project/benchmark.log
~~~

Their catalog duration totals 9,897 seconds. Divide by summed wall seconds for single-worker real-time speed. Estimate corpus hours as '385.981 / (speed * parallelism)', plus 15% for downloads and variance. Confirm peak RSS leaves headroom before parallelism eight.

### Resumable full corpus

Use tmux. Rerunning this exact command skips valid stage files. Never add '--force'.

~~~bash
tmux new -s retranscription
cd /workspace/project
set -o pipefail
uv run arvamusfestivali-transcripts run \
  --year 2026 --root /workspace/project/stage \
  --catalog-snapshot /workspace/project/data/catalog/20260920T112540Z-2026.json \
  --transcriber-script /workspace/project/vendor/estonian-audio-transcription/scripts/transcribe.py \
  --parallelism 8 |& tee -a /workspace/project/full-run.log
~~~

Detach with Ctrl-b d. Read-only progress checks are safe during the run:

~~~bash
cd /workspace/project
uv run python scripts/manage_retranscription.py status \
  --catalog data/catalog/20260920T112540Z-2026.json \
  --stage stage/data/transcripts/2026 \
  --lock data/retranscription/2026-run-lock.json --year 2026 || true
sqlite3 stage/data/state/pipeline.sqlite3 \
  "select status,count(*) from episodes where year=2026 group by status order by status;"
sqlite3 stage/data/state/pipeline.sqlite3 \
  "select episode_id,failure_stage,failure_message from episodes where status='failed' order by episode_id;"
~~~

Retry a failure with the frozen command plus '--episode-id ID'. Failed episodes retain verified cached audio; successful episodes delete it.

### Final validation and download

Only an exact complete corpus exits zero:

~~~bash
uv run python scripts/manage_retranscription.py validate \
  --catalog data/catalog/20260920T112540Z-2026.json \
  --stage stage/data/transcripts/2026 \
  --lock data/retranscription/2026-run-lock.json \
  --manifest stage/2026-retranscription-manifest.json --year 2026
tar -C stage -czf /workspace/af-2026-retranscription.tgz \
  data/transcripts/2026 2026-retranscription-manifest.json data/state/pipeline.sqlite3
cp acceptance.log benchmark.log full-run.log /workspace/
sha256sum /workspace/af-2026-retranscription.tgz /workspace/acceptance.log \
  /workspace/benchmark.log /workspace/full-run.log > /workspace/af-2026-downloads.sha256
~~~

Download the tarball, checksum file, and logs over an authenticated channel. Compare local Get-FileHash results before extraction.

## 3. Local import, reassignment, and explicit promotion

Extract into a new ignored incoming directory, never the canonical archive:

~~~powershell
$run = 'data\retranscription\incoming-2026-runpod'
New-Item -ItemType Directory -Force $run | Out-Null
tar -C $run -xzf .tmp\af-2026-retranscription.tgz
.\.venv\Scripts\python.exe scripts\manage_retranscription.py validate `
  --catalog data\catalog\20260920T112540Z-2026.json `
  --stage "$run\data\transcripts\2026" `
  --lock data\retranscription\2026-run-lock.json `
  --manifest "$run\local-validation-manifest.json" --year 2026
~~~

Reuse saved pyannote turns before promotion. This proves old/new audio SHA equality for every episode, writes cue assignments atomically, and regenerates filtered review HTML in staging. It does not run pyannote or change existing results.

~~~powershell
.\.venv\Scripts\python.exe scripts\reassign_diarization.py `
  --old-transcripts data\transcripts\2026 `
  --new-transcripts "$run\data\transcripts\2026" `
  --saved-diarization data\diarization\2026 `
  --output "$run\data\diarization\2026" `
  --review-output "$run\site\diarization" `
  --manifest "$run\diarization-reassignment-manifest.json"
~~~

Require selected=264, completed=264, failed=0. A changed audio hash blocks reuse; rerun pyannote only for that episode. Do not promote while 'git status --short' contains unpreserved transcript, diarization, review, or site work. This checkout is already dirty in adjacent analysis/site files, so preserve that work first.

Transcript promotion rechecks every checksum, renames the current archive to a backup, then renames staging into place. It restores the original target if the second rename fails:

~~~powershell
.\.venv\Scripts\python.exe scripts\manage_retranscription.py promote `
  --stage "$run\data\transcripts\2026" `
  --target data\transcripts\2026 `
  --backup data\retranscription\backups\2026-before-runpod `
  --manifest "$run\local-validation-manifest.json"
~~~

Keep the backup through downstream acceptance. Promote staged diarization and review HTML only after their zero-failure manifest and after preserving dirty work. Use backup-first whole-directory renames with rollback; never copy over existing directories, which would silently mix generations.

## 4. Ordered downstream rebuild

All text-derived outputs become **stale at transcript promotion**. Unchanged audio preserves deduplication and speaker turns, not cue boundaries, embeddings, BERTopic fits, Jev assignments, topic names, exports, site data, or notebook outputs.

Work under 'data/retranscription/rebuild/' and direct supported '--output' options there. Do not overwrite 'data/topic-analysis/results', 'site/data', or notebooks until the staged rebuild passes hashes and counts.

1. Validate 264 canonical archives and confirm 219 unique audio hashes. Duplicate membership must match the run lock.
2. Promote reassigned diarization/review outputs as whole directories and regenerate the review index. Rerun 'notebooks/analyze_diarization.ipynb'. Turn metrics should be stable; cue/text metrics may change.
3. Rerun 'notebooks/compare_segmentation_modes.ipynb' to rebuild semantic and fixed segments. Embedding cache keys include normalized text, audio SHA, model revision, and chunking: unchanged passages may hit; changed text must miss. Never reuse old assignments merely because audio is unchanged.
4. Refit both selected BGE models into staging and export the new dataset:

~~~powershell
uv run --group topic-analysis python scripts\export_segment_dataset.py `
  --input-root data\retranscription\rebuild\fine-topics `
  --output-root data\retranscription\rebuild\segment-dataset
~~~

5. Topic IDs and vocabulary can change. Run 'name_topics_openai.py' and 'assign_topics_jev.py' only with separate explicit authorization for paid/cloud APIs; this handoff does not call them. Reuse responses only for exact request/source hashes recorded in manifests.
6. With fresh names and Jev assignments, regenerate in order: 'apply_topic_names.py', 'export_high_level_topics.py', 'export_jev_full_review.py', 'export_jev_treemaps.py', 'export_jev_bubble_maps.py', 'export_jev_segment_maps.py', 'export_jev_year_results.py', 'compare_jev_year_topics.py', 'export_named_topic_charts.py', then 'export_topic_maps.py'. Every manifest must reference the new dataset hash.
7. Build site data into staging:

~~~powershell
uv run python scripts\build_results_site.py --root . `
  --output data\retranscription\rebuild\site-data
~~~

8. Restart kernels and rerun: 'compare_embedding_models.ipynb', 'compare_segmentation_modes.ipynb', 'review_segment_topics.ipynb', 'analyze_jev_topics.ipynb', 'compare_jev_topics_2025_2026.ipynb', 'analyze_diarization.ipynb', and 'analyze_speaker_topic_participation.ipynb'.
9. Promote topic results, site data/pages, and executed notebooks only as whole staged directories with timestamped backups after tests and source-hash checks.

## Operator checklist

### Local before upload

- [ ] Preserve the dirty tree; run focused tests and recreate the run lock.
- [ ] Build the allow-listed package; verify no secrets or old outputs.
- [ ] Record SHA-256 and upload through an authenticated channel.

### On RunPod

- [ ] Verify checksum, install frozen dependencies, and prefetch the pinned model.
- [ ] Pass the 383-second acceptance; benchmark three realistic recordings.
- [ ] Run in tmux; monitor status/SQLite and retry failed IDs.
- [ ] Validate exactly 264 files; write the manifest and download checksums.

### Downloaded back

- [ ] Verify checksums before extracting into 'data/retranscription/incoming-*'.
- [ ] Revalidate with zero missing, failed, unexpected, model, or audio mismatch.
- [ ] Reassign saved turns with 264/264 success; stage filtered review HTML.
- [ ] Explicitly promote transcripts with a retained backup; never merge directories.

### After import

- [ ] Validate 264 archives and 219 canonical audio hashes.
- [ ] Promote cue reassignment/review only after preserving current dirty work.
- [ ] Rebuild segments, changed embeddings, topic fits, datasets, authorized API-derived names/assignments, exports, site pages, and notebooks in order.
- [ ] Verify source hashes and tests, then promote whole staged downstream directories. Keep old results marked stale until that gate passes.
