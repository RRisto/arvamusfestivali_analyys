# Arvamusfestival transcript archive

This project turns Arvamusfestival podcast audio into searchable transcripts and
topic analysis. It finds recordings through Apple Podcasts and the linked
SoundCloud feed, downloads each MP3, transcribes the speech into timestamped
Estonian text, and separately identifies when different speakers talk. After a
transcript is validated, its local audio cache is deleted; the JSON keeps a link
to the original audio. The site lets you read transcripts and jump to a point in
the recording.

**What is included:** `data/transcripts/2026/` contains 264 transcript files,
all published in the podcast feed in 2026. After duplicate audio is removed,
the topic analysis covers 219 distinct festival talks: 114 from the **2025
programme** and 105 from the **2026 programme**. The folder year is the podcast
publication year; the programme year describes when the festival talk took place.

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

See [.env.example](.env.example) for optional API keys used by diarization and
topic analysis. Set the keys you need in your shell; the project does not load
`.env` automatically.

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
