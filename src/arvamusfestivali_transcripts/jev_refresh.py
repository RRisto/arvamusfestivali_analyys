"""Refresh Jev assignments from current transcript text without refitting topics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

JEV_VOCABULARY_MODEL = "semantic:bge:leaf-local-6-2:42"
JEV_REFRESH_STAGES = (
    "semantic_source",
    "jev_assignment",
    "jev_review",
    "jev_bubble_maps",
    "jev_treemaps",
    "jev_segment_maps",
    "jev_year_results",
    "jev_year_comparison",
    "site_data",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            os.replace(temporary, path)
        except PermissionError:
            if os.name != "nt":
                raise
            path.unlink(missing_ok=True)
            os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def transcript_corpus_fingerprint(transcript_root: Path, years: Sequence[int]) -> str:
    """Hash ordered archive names and bytes so any transcript correction invalidates outputs."""
    requested = tuple(sorted(set(years)))
    if not requested:
        raise ValueError("years must be nonempty")
    digest = hashlib.sha256()
    count = 0
    for year in requested:
        files = sorted((Path(transcript_root) / str(year)).glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no transcript archives found for {year}")
        for path in files:
            relative = f"{year}/{path.name}".encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(_sha256(path)))
            count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def build_semantic_source(
    transcript_root: Path,
    *,
    years: Sequence[int],
    output: Path,
    cache_root: Path,
    adapter: Any,
    batch_size: int,
    programme_metadata: Path | None = None,
) -> dict[str, Any]:
    """Rebuild native semantic segments and only the BGE vectors their current text needs."""
    import pandas as pd

    from .topic_analysis.cache import EmbeddingCache
    from .topic_analysis.corpus import load_corpus
    from .topic_analysis.embedders import embed_passages
    from .topic_analysis.semantic_segmentation import (
        build_atomic_blocks_many,
        segment_episodes_semantically,
    )
    from .topic_analysis.types import ChunkingConfig, SemanticSegmentationConfig

    requested = tuple(sorted(set(years)))
    corpus_sha256 = transcript_corpus_fingerprint(transcript_root, requested)
    episodes = tuple(
        episode
        for year in requested
        for episode in load_corpus(Path(transcript_root), year)
    )
    if not episodes:
        raise ValueError("current transcript corpus is empty")

    config = SemanticSegmentationConfig()
    cache = EmbeddingCache(cache_root)
    atomic = build_atomic_blocks_many(episodes, config)
    boundary_embeddings = embed_passages(
        adapter,
        atomic,
        cache,
        batch_size,
        chunking=config.atomic_chunking,
    )
    semantic = segment_episodes_semantically(
        episodes,
        atomic,
        boundary_embeddings,
        config,
    )
    segment_chunking = ChunkingConfig(
        min_seconds=90,
        target_seconds=300,
        max_seconds=600,
        min_words=1,
        max_words=1_000_000,
        overlap_seconds=0,
        merge_tail_seconds=0,
        max_merged_seconds=600,
    )
    segment_embeddings = embed_passages(
        adapter,
        semantic.passages,
        cache,
        batch_size,
        chunking=segment_chunking,
    )
    programme_year = {episode.episode_id: int(episode.published_at[:4]) for episode in episodes}
    if programme_metadata is not None:
        metadata = pd.read_parquet(programme_metadata)
        if metadata.episode_id.astype(str).duplicated().any():
            raise ValueError("programme metadata episode IDs must be unique")
        mapped = dict(
            zip(
                metadata.episode_id.astype(str),
                metadata.programme_year.astype(int),
                strict=True,
            )
        )
        missing = sorted(set(programme_year) - set(mapped))
        if missing:
            raise ValueError(f"programme metadata is missing {len(missing)} transcript episodes")
        programme_year = {episode_id: mapped[episode_id] for episode_id in programme_year}
    records = []
    for passage in semantic.passages:
        records.append(
            {
                "model_key": JEV_VOCABULARY_MODEL,
                "passage_id": passage.passage_id,
                "episode_id": passage.episode_id,
                "title": passage.title,
                "duplicate_episode_ids": json.dumps(passage.duplicate_episode_ids),
                "audio_sha256": passage.audio_sha256,
                "start_seconds": passage.start_seconds,
                "end_seconds": passage.end_seconds,
                "topic_id": -1,
                "membership_strength": None,
                "text": passage.text,
                "audio_link": passage.timestamped_audio_url,
                "topic_key": f"{JEV_VOCABULARY_MODEL}:-1",
                "topic_label": "-1_unassigned",
                "topic_name": "Unassigned",
                "confidence": None,
                "confidence_type": "not_run_jev_only_refresh",
                "talk_name": passage.title,
                "duration_seconds": passage.end_seconds - passage.start_seconds,
                "segmentation": "semantic",
                "candidate": "existing-jev-vocabulary",
                "seed": None,
                "segment_key": f"{JEV_VOCABULARY_MODEL}:{passage.passage_id}",
                "other_model_overlaps": "[]",
                "topic_name_keywords": "",
                "topic_name_source": "existing-jev-vocabulary",
                "topic_naming_confidence": None,
                "topic_coherence": None,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty or frame.segment_key.duplicated().any():
        raise ValueError("semantic source must contain unique segments")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest = {
        "schema_version": 1,
        "purpose": "Jev-only refresh; existing vocabulary, no topic refit or renaming",
        "assignment_baseline": "none",
        "corpus_sha256": corpus_sha256,
        "years": list(requested),
        "episodes": len(episodes),
        "segments": len(frame),
        "programme_year_counts": {
            str(year): sum(value == year for value in programme_year.values())
            for year in sorted(set(programme_year.values()))
        },
        "vocabulary_model": JEV_VOCABULARY_MODEL,
        "embedding_model_id": adapter.model_id,
        "embedding_revision": adapter.model_revision,
        "embedding_dimension": adapter.dimension,
        "semantic_segmentation": asdict(config),
        "segment_chunking": asdict(segment_chunking),
        "boundary_embedding_cache_hits": boundary_embeddings.cache_hits,
        "segment_embedding_cache_hits": segment_embeddings.cache_hits,
        "embedding_cache_hits": (
            boundary_embeddings.cache_hits + segment_embeddings.cache_hits
        ),
        "models": {
            "semantic": {
                "embedding_model_id": adapter.model_id,
                "embedding_revision": adapter.model_revision,
                "embedding_dimension": adapter.dimension,
            }
        },
        "source_sha256": _sha256(output),
    }
    _atomic_json(output.parent / "manifest.json", manifest)
    return manifest


def run_refresh_pipeline(
    *,
    corpus_sha256: str,
    state_path: Path,
    actions: Mapping[str, Callable[[], object]],
    allow_paid_jev: bool,
) -> dict[str, Any]:
    """Run and checkpoint the Jev-only refresh, resetting all stages for new transcript text."""
    if not allow_paid_jev:
        raise ValueError("pass --allow-paid-jev to authorize Jev API calls")
    if set(actions) != set(JEV_REFRESH_STAGES):
        raise ValueError("actions must define every Jev refresh stage exactly once")
    state: dict[str, Any] = {}
    if state_path.exists():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            state = loaded
    if state.get("corpus_sha256") != corpus_sha256:
        state = {"corpus_sha256": corpus_sha256, "completed_stages": []}
        _atomic_json(state_path, state)
    completed = list(state.get("completed_stages", []))
    if any(stage not in JEV_REFRESH_STAGES for stage in completed):
        raise ValueError("refresh state contains an unknown stage")
    for stage in JEV_REFRESH_STAGES:
        if stage in completed:
            continue
        actions[stage]()
        completed.append(stage)
        state = {"corpus_sha256": corpus_sha256, "completed_stages": completed}
        _atomic_json(state_path, state)
    return state


def _run_command(root: Path, arguments: Sequence[str]) -> None:
    subprocess.run(list(arguments), cwd=root, check=True)


def _copy_tree_contents(source: Path, target: Path) -> None:
    """Copy files without metadata operations unsupported by some mounted filesystems."""
    target.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing symlink in staged refresh output: {path}")
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        else:
            raise ValueError(f"unsupported staged refresh entry: {path}")


def _replace_from_stage(source: Path, target: Path, backup: Path, root: Path) -> None:
    root = root.resolve()
    for path in (source, target, backup):
        resolved = path.resolve()
        if resolved == root or root not in resolved.parents:
            raise ValueError(f"refusing directory replacement outside project root: {path}")
    if not source.is_dir():
        raise FileNotFoundError(f"staged refresh output is missing: {source}")
    if target.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        _copy_tree_contents(target, backup)
    if target.exists():
        shutil.rmtree(target)
    _copy_tree_contents(source, target)


def _default_actions(
    *,
    root: Path,
    years: tuple[int, ...],
    corpus_sha256: str,
    device: str,
    batch_size: int,
    workers: int,
) -> dict[str, Callable[[], object]]:
    results = root / "data" / "topic-analysis" / "results"
    cache = root / "data" / "topic-analysis" / "cache"
    stage = root / "data" / "retranscription" / "rebuild" / f"jev-{corpus_sha256[:12]}"
    staged_dataset = stage / "segment-dataset"
    staged_jev = stage / "jev-semantic-all"
    canonical_dataset = results / "segment-dataset"
    canonical_jev = results / "jev-semantic-all"
    backup = root / "data" / "retranscription" / "backups" / f"jev-before-{corpus_sha256[:12]}"
    metadata = results / "talk-metadata" / "talks.parquet"

    def semantic_source() -> None:
        from .topic_analysis.embedders import BgeM3Embedder

        selected_device = device
        if selected_device == "auto":
            import torch

            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        build_semantic_source(
            root / "data" / "transcripts",
            years=years,
            output=staged_dataset / "all-segments.parquet",
            cache_root=cache,
            adapter=BgeM3Embedder(device=selected_device),
            batch_size=batch_size,
            programme_metadata=metadata if metadata.exists() else None,
        )

    def jev_assignment() -> None:
        command = [
            sys.executable,
            str(root / "scripts" / "assign_topics_jev.py"),
            "--root",
            str(root),
            "--all-episodes",
            "--workers",
            str(workers),
            "--dataset",
            str(staged_dataset / "all-segments.parquet"),
            "--output",
            str(staged_jev),
            "--vocabulary-model",
            JEV_VOCABULARY_MODEL,
        ]
        previous = canonical_jev / "responses.jsonl"
        if previous.exists():
            command.extend(["--reuse-from", str(previous)])
        _run_command(root, command)

    def jev_review() -> None:
        _replace_from_stage(
            staged_dataset,
            canonical_dataset,
            backup / "segment-dataset",
            root,
        )
        _replace_from_stage(staged_jev, canonical_jev, backup / "jev-semantic-all", root)
        _run_command(root, [sys.executable, str(root / "scripts" / "export_jev_full_review.py")])

    script_for_stage = {
        "jev_bubble_maps": "export_jev_bubble_maps.py",
        "jev_treemaps": "export_jev_treemaps.py",
        "jev_segment_maps": "export_jev_segment_maps.py",
        "jev_year_results": "export_jev_year_results.py",
        "jev_year_comparison": "compare_jev_year_topics.py",
    }
    actions: dict[str, Callable[[], object]] = {
        "semantic_source": semantic_source,
        "jev_assignment": jev_assignment,
        "jev_review": jev_review,
    }
    for stage_name, script_name in script_for_stage.items():
        actions[stage_name] = lambda script_name=script_name: _run_command(
            root, [sys.executable, str(root / "scripts" / script_name)]
        )
    actions["site_data"] = lambda: _run_command(
        root,
        [
            sys.executable,
            str(root / "scripts" / "build_results_site.py"),
            "--root",
            str(root),
            "--output",
            "site/data",
        ],
    )
    return actions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh semantic segments and Jev assignments without refitting topics."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--year", action="append", type=int, dest="years", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allow-paid-jev", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers <= 0:
        parser.error("batch size and workers must be positive")
    root = args.root.resolve()
    years = tuple(sorted(set(args.years)))
    corpus_sha256 = transcript_corpus_fingerprint(root / "data" / "transcripts", years)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "corpus_sha256": corpus_sha256,
                    "years": list(years),
                    "stages": list(JEV_REFRESH_STAGES),
                    "topic_refit": False,
                    "topic_renaming": False,
                    "paid_jev_authorized": args.allow_paid_jev,
                },
                indent=2,
            )
        )
        return 0
    actions = _default_actions(
        root=root,
        years=years,
        corpus_sha256=corpus_sha256,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    state = run_refresh_pipeline(
        corpus_sha256=corpus_sha256,
        state_path=root / "data" / "retranscription" / "jev-refresh-state.json",
        actions=actions,
        allow_paid_jev=args.allow_paid_jev,
    )
    print(json.dumps(state, indent=2))
    return 0
