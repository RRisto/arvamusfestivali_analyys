# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.28,<0.29",
#   "pyannote.audio>=4,<5",
# ]
# ///
# ruff: noqa: E402, I001
"""Run resumable batch diarization from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arvamusfestivali_transcripts.diarization_batch import main


if __name__ == "__main__":
    raise SystemExit(main())
