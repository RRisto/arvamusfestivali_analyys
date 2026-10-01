"""Executable contract for the fixed-versus-semantic segmentation notebook."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from arvamusfestivali_transcripts.archive import write_archive

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "compare_segmentation_modes.ipynb"


def test_segmentation_notebook_is_thin_local_and_clean() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    source = "\n".join(cell.source for cell in code_cells)

    for public_name in (
        "build_atomic_blocks_many",
        "segment_episodes_semantically",
        "plot_semantic_boundaries",
        "chunk_episodes",
        "embed_passages",
        "fit_topic_model",
    ):
        assert public_name in source
    assert 'SEGMENTATION_MODES = ("fixed", "semantic")' in source
    assert 'MODEL_KEYS = ("qwen", "bge")' in source
    assert "GeminiEmbedder" not in source
    assert "GEMINI_API_KEY" not in source
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)


def test_segmentation_notebook_executes_offline_for_both_modes(
    tmp_path: Path, six_topic_archives: Path, monkeypatch
) -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    project_root = tmp_path / "segmentation-notebook-project"
    transcript_root = project_root / "data" / "transcripts" / "2026"
    transcript_root.mkdir(parents=True)
    for archive in (six_topic_archives / "2026").glob("*.json"):
        copied = transcript_root / archive.name
        shutil.copy2(archive, copied)
        payload = json.loads(copied.read_text(encoding="utf-8"))
        title = payload["episode"]["title"]
        payload["transcription"]["cues"][0]["text"] = (
            f"{title} opening context and public priorities."
        )
        payload["transcription"]["cues"][1]["text"] = (
            f"{title} proposed solutions and implementation details."
        )
        payload["transcription"]["text"] = " ".join(
            cue["text"] for cue in payload["transcription"]["cues"]
        )
        write_archive(copied, payload, force=True)

    monkeypatch.setenv("TOPIC_ANALYSIS_PROJECT_ROOT", str(project_root))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    config = next(
        cell
        for cell in notebook.cells
        if "DRY_RUN_WITH_FAKE_EMBEDDINGS = False" in cell.source
    )
    config.source = config.source.replace(
        "DRY_RUN_WITH_FAKE_EMBEDDINGS = False",
        "DRY_RUN_WITH_FAKE_EMBEDDINGS = True",
    )

    executed = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    ).execute()

    assert all(
        cell.execution_count is not None
        for cell in executed.cells
        if cell.cell_type == "code"
    )
    output = "".join(
        item.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for item in cell.outputs
    )
    assert "SEGMENTATION_NOTEBOOK_OK" in output
    assert "modes=fixed,semantic" in output
    assert "runs=4" in output
    assert "boundary_figures=1" in output
