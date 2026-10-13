# ruff: noqa: E402, I001
"""Build a standalone speaker-segment review page from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arvamusfestivali_transcripts.diarization_review import main


if __name__ == "__main__":
    raise SystemExit(main())
