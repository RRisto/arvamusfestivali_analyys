# Run the diarization batch on Runpod

The archive currently contains 264 transcripts and about 386 hours / 35 GB of
source audio. The batch streams and verifies one episode at a time, keeps only
the small JSON/HTML results, and resumes by skipping valid outputs.

## 1. Create the Pod

Create a regular GPU Pod with:

- GPU: RTX 3090 (24 GB)
- Template: a current Runpod PyTorch template
- Container disk: 50 GB
- One GPU
- Automatic termination: 24 hours as a safety limit

Add `HF_TOKEN` as a Pod environment variable. Do not upload the local `.env`
file. The Hugging Face account behind this token must have accepted the terms
for `pyannote/speaker-diarization-community-1`.

## 2. Transfer this working tree

The batch needs the code and transcripts. Include the completed pilot so the
remote run skips it. From PowerShell in the repository root, make an upload
archive without credentials while preserving the directory layout:

```powershell
tar.exe -a -c -f runpod-diarization.zip `
  src scripts data/transcripts data/diarization site/diarization `
  pyproject.toml docs/RUNPOD.md
```

Upload `runpod-diarization.zip` into `/workspace` using the Pod's file browser,
then open its terminal:

```bash
cd /workspace
unzip runpod-diarization.zip -d arvamusfestivali_salvestused
cd arvamusfestivali_salvestused
```

If resuming from an earlier Pod, also copy its `data/diarization/` and
`site/diarization/` directories into this working tree before starting.

## 3. Install and verify the runtime

Keep the CUDA-enabled PyTorch from the Runpod image and install pyannote around
it in a virtual environment:

```bash
apt-get update
apt-get install -y ffmpeg unzip
python -m venv --system-site-packages .venv-runpod
source .venv-runpod/bin/activate
python -m pip install --upgrade pip
python -m pip install "httpx>=0.28,<0.29" "pyannote.audio>=4,<5"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ffmpeg -version | head -n 1
```

The Python check must print `True` and an NVIDIA GPU name. The batch refuses to
run without CUDA unless `--allow-cpu` is explicitly supplied.

## 4. Benchmark five episodes

Preview the selection without loading the model:

```bash
python scripts/diarize_all.py --limit 5 --dry-run
```

Then run the benchmark:

```bash
time python scripts/diarize_all.py --limit 5 2>&1 | tee benchmark.log
```

Download or inspect `site/diarization/index.html` and a few linked episode pages.
The runner writes `data/diarization/batch-status.json` after every episode.

## 5. Continue the full archive

```bash
time python scripts/diarize_all.py 2>&1 | tee full-batch.log
```

Re-running the same command is safe: complete JSON outputs are validated and
skipped, while missing, corrupt, and previously failed episodes are retried.
One episode failure is logged and does not stop later episodes. A nonzero final
exit status means one or more selected episodes failed.

## 6. Retrieve results and terminate

Download these paths before terminating the Pod:

```text
data/diarization/
site/diarization/
benchmark.log
full-batch.log
```

After copying them into the same paths in the local repository, serve the review
index with:

```powershell
uv run python -m http.server 8765 --directory site
```

Open `http://localhost:8765/diarization/`. Terminate the Pod after confirming the
results have been downloaded; stopping it can continue to incur storage charges.
