# Arvamusfestival transcript archive

This project archives Arvamusfestival podcast transcripts as timestamped JSON,
adds speaker diarization, and publishes a static site for exploring transcripts
and topic analysis. It discovers episodes from the Apple Podcasts collection
and the linked SoundCloud RSS feed.

## Quick start

Use Python 3.12+, [uv](https://docs.astral.sh/uv/), and `ffmpeg` on `PATH`.
From the repository root in PowerShell:

```powershell
uv sync --dev
uv run arvamusfestivali-transcripts run --year 2026 --dry-run --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py
uv run arvamusfestivali-transcripts run --year 2026 --limit 1 --transcriber-script vendor/estonian-audio-transcription/scripts/transcribe.py
```

Inspect the one-episode result in `data/transcripts/2026/` before running the
full year. The command skips valid existing archives when resumed. For the full
run, remove `--limit 1`.

## View the results

```powershell
uv run python scripts/build_results_site.py --root . --output site/data
uv run python -m http.server 8765 --directory site
```

Open `http://localhost:8765/`. Serve the site over HTTP so its JSON data loads.

## Project data

- `data/transcripts/<year>/<soundcloud-id>.json`: canonical transcript archives.
- `data/diarization/<year>/`: separate speaker assignments.
- `site/`: static results website and review pages.
- `notebooks/`: topic analysis and comparison notebooks.

## Guides

- [Transcription, archive format, and diarization](docs/transcription-guide.md)
- [RunPod diarization batch](docs/RUNPOD.md)
- [RunPod transcript repair](RUNPOD_RETRANSCRIPTION.md)
- [Topic analysis and experiments](docs/topic-analysis-guide.md)
- [Topic model review](docs/REVIEW.md)

For development, run `uv run pytest -q` and `uv run ruff check .`.
