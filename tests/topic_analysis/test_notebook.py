"""Executable contract for the embedding-comparison notebook."""

import ast
import json
import re
import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from arvamusfestivali_transcripts.archive import write_archive
from arvamusfestivali_transcripts.topic_analysis import load_corpus, select_diverse_episodes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "compare_embedding_models.ipynb"


def test_comparison_notebook_is_thin_and_ordered() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    for public_name in (
        "load_corpus",
        "select_diverse_episodes",
        "chunk_episodes",
        "embed_passages",
        "fit_topic_model",
        "evaluate_run",
        "export_experiment",
    ):
        assert public_name in source
    assert "GEMINI_API_KEY" in source
    assert "class QwenEmbedder" not in source
    assert "def chunk_episode" not in source


def test_notebook_has_no_saved_outputs_or_execution_counts() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)


def test_readme_explicit_ids_select_distinct_archived_recordings() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    example = re.search(r"EXPLICIT_EPISODE_IDS = (\([^\n]+\))", readme)
    assert example is not None
    episode_ids = ast.literal_eval(example.group(1))
    assert len(episode_ids) == 2

    corpus = load_corpus(PROJECT_ROOT / "data" / "transcripts", 2026)
    selected = select_diverse_episodes(corpus, explicit_ids=episode_ids)
    assert len({episode.audio_sha256 for episode in selected}) == 2


def test_notebook_executes_offline_and_exports_manifest(
    tmp_path: Path, six_topic_archives: Path, monkeypatch
) -> None:
    project_root = tmp_path / "notebook-project"
    transcript_root = project_root / "data" / "transcripts" / "2026"
    transcript_root.mkdir(parents=True)
    for archive in (six_topic_archives / "2026").glob("*.json"):
        copied = transcript_root / archive.name
        shutil.copy2(archive, copied)
        payload = json.loads(copied.read_text(encoding="utf-8"))
        # The compact shared fixture repeats one text; BERTopic needs distinct text to
        # resolve representative documents to a passage ID.
        title = payload["episode"]["title"]
        payload["transcription"]["cues"][0]["text"] = f"{title} public discussion begins."
        payload["transcription"]["cues"][1]["text"] = f"{title} speakers consider solutions."
        payload["transcription"]["text"] = " ".join(
            cue["text"] for cue in payload["transcription"]["cues"]
        )
        write_archive(copied, payload, force=True)

    monkeypatch.setenv("TOPIC_ANALYSIS_PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    config = next(
        cell for cell in notebook.cells
        if "DRY_RUN_WITH_FAKE_EMBEDDINGS = False" in cell.source
    )
    config.source = config.source.replace(
        "DRY_RUN_WITH_FAKE_EMBEDDINGS = False", "DRY_RUN_WITH_FAKE_EMBEDDINGS = True"
    )

    executed = NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    ).execute()

    assert all(
        cell.execution_count is not None
        for cell in executed.cells if cell.cell_type == "code"
    )
    manifests = list((project_root / "data" / "topic-analysis" / "results").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["selected_audio_hashes"]) == 6
    assert set(manifest["model_metadata"]) == {"qwen", "bge"}
    assert "Gemini skipped: GEMINI_API_KEY is not set" in "".join(
        output.get("text", "")
        for cell in executed.cells if cell.cell_type == "code"
        for output in cell.outputs
    )
