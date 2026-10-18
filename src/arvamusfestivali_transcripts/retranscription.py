"""Reproducible staging, validation, and promotion for selected retranscriptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .archive import validate_archive
from .schema import CatalogSnapshot

PINNED_MODEL = {
    "name": "large",
    "repository": "TalTechNLP/streaming-zipformer-large.et-en",
    "revision": "2812262bd4fd5d4327dc09ca0e6c19e8a32a03d6",
    "precision": "int8",
}


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def create_run_lock(
    catalog_path: Path,
    baseline_dir: Path,
    transcriber_script: Path,
    *,
    episode_ids: Sequence[str],
    expected_model: Mapping[str, str],
) -> dict[str, Any]:
    """Pin selected episodes, integration script, model, and baseline identities."""
    catalog = CatalogSnapshot.from_json(catalog_path.read_text(encoding="utf-8"))
    required_model_fields = {"name", "repository", "revision", "precision"}
    if set(expected_model) != required_model_fields or not all(expected_model.values()):
        raise ValueError("expected model must pin name, repository, revision, and precision")
    if not transcriber_script.is_file():
        raise FileNotFoundError(transcriber_script)

    selected_ids = [str(episode_id) for episode_id in episode_ids]
    if not selected_ids:
        raise ValueError("episode selection must contain at least one ID")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("episode selection must not repeat IDs")
    catalog_by_id = {episode.id: episode for episode in catalog.episodes}
    unknown = sorted(set(selected_ids) - set(catalog_by_id))
    if unknown:
        raise ValueError(f"episode selection contains IDs not in catalog: {', '.join(unknown)}")

    baseline_archives: dict[str, tuple[dict[str, Any], str]] = {}
    for episode in catalog.episodes:
        archive_path = baseline_dir / f"{episode.id}.json"
        archive = _read_object(archive_path)
        validate_archive(archive, catalog.year)
        metadata = archive["episode"]
        if metadata["id"] != episode.id or metadata["audio_url"] != episode.audio_url:
            raise ValueError(f"baseline identity does not match catalog for episode {episode.id}")
        baseline_archives[episode.id] = (archive, sha256_file(archive_path))

    source_by_audio: dict[str, str] = {}
    for episode_id in selected_ids:
        archive, _ = baseline_archives[episode_id]
        audio_sha256 = str(archive["episode"]["audio_sha256"])
        if audio_sha256 in source_by_audio:
            raise ValueError("episode selection must not repeat the same recording")
        source_by_audio[audio_sha256] = episode_id

    episodes: dict[str, dict[str, str]] = {}
    for episode in catalog.episodes:
        archive, baseline_sha256 = baseline_archives[episode.id]
        metadata = archive["episode"]
        audio_sha256 = str(metadata["audio_sha256"])
        source_episode_id = source_by_audio.get(audio_sha256)
        if source_episode_id is None:
            continue
        episodes[episode.id] = {
            "audio_sha256": audio_sha256,
            "audio_url": episode.audio_url,
            "baseline_sha256": baseline_sha256,
            "source_episode_id": source_episode_id,
        }

    return {
        "schema_version": 2,
        "scope": "selected_episodes",
        "year": catalog.year,
        "catalog_filename": catalog_path.name,
        "catalog_sha256": sha256_file(catalog_path),
        "transcriber_script_sha256": sha256_file(transcriber_script),
        "expected_model": dict(expected_model),
        "transcribe_episode_ids": selected_ids,
        "episodes": episodes,
    }


def write_run_lock(path: Path, lock: Mapping[str, Any]) -> Path:
    """Write a run lock atomically."""
    return _atomic_json(path, lock)


def read_episode_selection(path: Path) -> list[str]:
    """Read an explicit JSON selection of affected episode IDs."""
    payload = _read_object(path)
    episode_ids = payload.get("episode_ids")
    if not isinstance(episode_ids, list) or not all(
        isinstance(episode_id, str) for episode_id in episode_ids
    ):
        raise ValueError("episode selection must contain a string episode_ids list")
    return episode_ids


def materialize_aliases(
    baseline_dir: Path,
    stage_dir: Path,
    lock: Mapping[str, Any],
) -> int:
    """Create duplicate-audio archive aliases from each newly transcribed source."""
    episodes = lock.get("episodes")
    sources = lock.get("transcribe_episode_ids")
    if not isinstance(episodes, dict) or not isinstance(sources, list):
        raise ValueError("invalid run lock")
    source_ids = set(sources)
    created = 0
    for episode_id, metadata in sorted(episodes.items()):
        if episode_id in source_ids:
            continue
        if not isinstance(metadata, dict):
            raise ValueError("invalid run lock episode metadata")
        source_id = metadata.get("source_episode_id")
        if not isinstance(source_id, str) or source_id not in source_ids:
            raise ValueError(f"invalid source episode for alias {episode_id}")
        source = _read_object(stage_dir / f"{source_id}.json")
        alias = _read_object(baseline_dir / f"{episode_id}.json")
        if source["episode"]["audio_sha256"] != alias["episode"]["audio_sha256"]:
            raise ValueError(f"audio identity differs for alias {episode_id}")
        alias["transcription"] = source["transcription"]
        _atomic_json(stage_dir / f"{episode_id}.json", alias)
        created += 1
    return created


def validate_stage(
    catalog_path: Path,
    transcript_dir: Path,
    lock: Mapping[str, Any],
    *,
    year: int,
) -> dict[str, Any]:
    """Validate a staged archive against its immutable inputs and return a manifest."""
    if lock.get("year") != year:
        raise ValueError("run lock year does not match requested year")
    if lock.get("catalog_sha256") != sha256_file(catalog_path):
        raise ValueError("catalog changed after run lock creation")
    catalog = CatalogSnapshot.from_json(catalog_path.read_text(encoding="utf-8"))
    if catalog.year != year:
        raise ValueError("catalog year does not match requested year")

    locked_episodes = lock.get("episodes")
    expected_model = lock.get("expected_model")
    if not isinstance(locked_episodes, dict) or not isinstance(expected_model, dict):
        raise ValueError("invalid run lock")
    if lock.get("scope") != "selected_episodes":
        raise ValueError("run lock is not scoped to selected episodes")
    catalog_by_id = {episode.id: episode for episode in catalog.episodes}
    expected_ids = set(locked_episodes)
    if not expected_ids or not expected_ids.issubset(catalog_by_id):
        raise ValueError("run lock episode set is empty or not contained in catalog")

    failures: list[dict[str, str]] = []
    files: dict[str, dict[str, Any]] = {}
    missing = 0
    for episode_id in sorted(expected_ids):
        episode = catalog_by_id[episode_id]
        path = transcript_dir / f"{episode.id}.json"
        if not path.is_file():
            missing += 1
            failures.append({"episode_id": episode.id, "reason": "missing transcript"})
            continue
        try:
            archive = _read_object(path)
            validate_archive(archive, year)
            episode_data = archive["episode"]
            if episode_data["id"] != episode.id:
                raise ValueError("episode ID does not match filename/catalog")
            if episode_data["audio_url"] != episode.audio_url:
                raise ValueError("audio URL does not match frozen catalog")
            locked = locked_episodes[episode.id]
            if episode_data["audio_sha256"] != locked["audio_sha256"]:
                raise ValueError("audio SHA-256 differs from baseline")
            if archive["transcription"]["model"] != expected_model:
                raise ValueError("transcription model metadata differs from run lock")
            files[path.name] = {
                "sha256": sha256_file(path),
                "baseline_sha256": locked["baseline_sha256"],
                "audio_sha256": episode_data["audio_sha256"],
                "cue_count": len(archive["transcription"]["cues"]),
            }
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append({"episode_id": episode.id, "reason": str(error)})

    present_json = {path.name for path in transcript_dir.glob("*.json") if path.is_file()}
    expected_names = {f"{episode_id}.json" for episode_id in expected_ids}
    unexpected = sorted(present_json - expected_names)
    valid = len(files)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "year": year,
        "catalog_filename": catalog_path.name,
        "catalog_sha256": lock["catalog_sha256"],
        "transcriber_script_sha256": lock["transcriber_script_sha256"],
        "model": dict(expected_model),
        "counts": {
            "expected": len(expected_ids),
            "valid": valid,
            "missing": missing,
            "failed": len(failures),
            "unexpected": len(unexpected),
        },
        "failures": failures,
        "unexpected_files": unexpected,
        "files": dict(sorted(files.items())),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Write a stage-validation manifest atomically."""
    return _atomic_json(path, manifest)


def manifest_is_promotable(manifest: Mapping[str, Any]) -> bool:
    """Return whether validation proved an exact, complete corpus."""
    counts = manifest.get("counts")
    return bool(
        isinstance(counts, dict)
        and counts.get("valid") == counts.get("expected")
        and counts.get("missing") == 0
        and counts.get("failed") == 0
        and counts.get("unexpected") == 0
        and not manifest.get("failures")
        and not manifest.get("unexpected_files")
    )


def promote_stage(
    stage_dir: Path,
    target_dir: Path,
    backup_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Promote verified selected files with rollback and byte-preserved bystanders."""
    if not manifest_is_promotable(manifest):
        raise ValueError("manifest is not promotable")
    if not stage_dir.is_dir():
        raise FileNotFoundError(stage_dir)
    if not target_dir.is_dir():
        raise FileNotFoundError(target_dir)
    if backup_dir.exists() or backup_dir.is_symlink():
        raise FileExistsError(f"backup already exists: {backup_dir}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest has no file checksums")
    actual_names = {path.name for path in stage_dir.glob("*.json") if path.is_file()}
    if actual_names != set(files):
        raise ValueError("staged file set changed after validation")
    for name, metadata in files.items():
        current_digest = sha256_file(stage_dir / name)
        if not isinstance(metadata, dict) or metadata.get("sha256") != current_digest:
            raise ValueError(f"staged file changed after validation: {name}")
        target_path = target_dir / name
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        if metadata.get("baseline_sha256") != sha256_file(target_path):
            raise ValueError(f"canonical file changed after run lock creation: {name}")

    backup_dir.mkdir(parents=True)
    backed_up: list[str] = []
    promoted: list[str] = []
    try:
        for name in sorted(files):
            os.replace(target_dir / name, backup_dir / name)
            backed_up.append(name)
        for name in sorted(files):
            os.replace(stage_dir / name, target_dir / name)
            promoted.append(name)
    except Exception:
        for name in reversed(promoted):
            os.replace(target_dir / name, stage_dir / name)
        for name in reversed(backed_up):
            os.replace(backup_dir / name, target_dir / name)
        backup_dir.rmdir()
        raise
    try:
        stage_dir.rmdir()
    except OSError:
        pass


def _add_validation_arguments(parser: argparse.ArgumentParser, *, manifest: bool) -> None:
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    if manifest:
        parser.add_argument("--manifest", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-lock", help="pin inputs before packaging")
    create.add_argument("--catalog", type=Path, required=True)
    create.add_argument("--baseline", type=Path, required=True)
    create.add_argument("--transcriber-script", type=Path, required=True)
    create.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="JSON file containing the explicit affected episode_ids list",
    )
    create.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate stage and write its manifest")
    _add_validation_arguments(validate, manifest=True)
    status = commands.add_parser("status", help="print current stage validation counts")
    _add_validation_arguments(status, manifest=False)
    promote = commands.add_parser("promote", help="replace canonical archive after validation")
    promote.add_argument("--stage", type=Path, required=True)
    promote.add_argument("--target", type=Path, required=True)
    promote.add_argument("--backup", type=Path, required=True)
    promote.add_argument("--manifest", type=Path, required=True)
    aliases = commands.add_parser(
        "materialize-aliases",
        help="create duplicate-audio aliases from selected staged transcriptions",
    )
    aliases.add_argument("--baseline", type=Path, required=True)
    aliases.add_argument("--stage", type=Path, required=True)
    aliases.add_argument("--lock", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create-lock":
            lock = create_run_lock(
                args.catalog,
                args.baseline,
                args.transcriber_script,
                episode_ids=read_episode_selection(args.episodes),
                expected_model=PINNED_MODEL,
            )
            write_run_lock(args.output, lock)
            print(f"wrote {args.output} ({len(lock['episodes'])} episodes)")
            return 0
        if args.command == "materialize-aliases":
            created = materialize_aliases(args.baseline, args.stage, _read_object(args.lock))
            print(f"materialized {created} duplicate-audio aliases")
            return 0
        if args.command in {"validate", "status"}:
            lock = _read_object(args.lock)
            manifest = validate_stage(args.catalog, args.stage, lock, year=args.year)
            if args.command == "validate":
                write_manifest(args.manifest, manifest)
            print(json.dumps(manifest["counts"], sort_keys=True))
            return 0 if manifest_is_promotable(manifest) else 1
        manifest = _read_object(args.manifest)
        promote_stage(args.stage, args.target, args.backup, manifest)
        print(f"promoted {args.stage} to {args.target}; backup retained at {args.backup}")
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
