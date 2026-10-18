from __future__ import annotations

import json

import pandas as pd

from arvamusfestivali_transcripts.topic_analysis.embedders import DeterministicHashEmbedder


def test_transcript_fingerprint_changes_with_text_but_not_file_order(
    tmp_path, write_topic_archive
):
    from arvamusfestivali_transcripts.jev_refresh import transcript_corpus_fingerprint

    root = tmp_path / "transcripts"
    first = write_topic_archive(root, episode_id="2", sha256="2" * 64, title="Second")
    write_topic_archive(root, episode_id="1", sha256="1" * 64, title="First")

    before = transcript_corpus_fingerprint(root, (2026,))
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["transcription"]["cues"][0]["text"] = "Corrected transcript text."
    payload["transcription"]["text"] = " ".join(
        cue["text"] for cue in payload["transcription"]["cues"]
    )
    first.write_text(json.dumps(payload), encoding="utf-8")

    assert transcript_corpus_fingerprint(root, (2026,)) != before


def test_semantic_source_uses_current_text_and_content_addressed_embeddings(
    tmp_path, write_topic_archive
):
    from arvamusfestivali_transcripts.jev_refresh import build_semantic_source

    transcripts = tmp_path / "transcripts"
    cues = [
        {"start_seconds": 0.0, "end_seconds": 100.0, "text": "Corrected opening."},
        {"start_seconds": 100.0, "end_seconds": 200.0, "text": "Corrected ending."},
    ]
    write_topic_archive(transcripts, cues=cues)
    output = tmp_path / "source" / "all-segments.parquet"
    cache = tmp_path / "cache"
    metadata = tmp_path / "talks.parquet"
    pd.DataFrame([{"episode_id": "1", "programme_year": 2025}]).to_parquet(
        metadata, index=False
    )
    adapter = DeterministicHashEmbedder(dimension=16)
    adapter.key = "bge"
    adapter.model_id = "test-bge"

    first = build_semantic_source(
        transcripts,
        years=(2026,),
        output=output,
        cache_root=cache,
        adapter=adapter,
        batch_size=4,
        programme_metadata=metadata,
    )
    second = build_semantic_source(
        transcripts,
        years=(2026,),
        output=output,
        cache_root=cache,
        adapter=adapter,
        batch_size=4,
        programme_metadata=metadata,
    )

    data = pd.read_parquet(output)
    assert set(data.segmentation) == {"semantic"}
    assert " ".join(data.text) == "Corrected opening. Corrected ending."
    assert "programme_year" not in data
    assert data.segment_key.is_unique
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert second["embedding_cache_hits"] > 0
    assert second["models"]["semantic"]["embedding_model_id"] == "test-bge"
    assert second["programme_year_counts"] == {"2025": 1}


def test_refresh_pipeline_runs_only_jev_flow_and_resets_when_transcripts_change(tmp_path):
    from arvamusfestivali_transcripts.jev_refresh import JEV_REFRESH_STAGES, run_refresh_pipeline

    assert JEV_REFRESH_STAGES == (
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
    state = tmp_path / "state.json"
    calls: list[str] = []
    actions = {stage: (lambda stage=stage: calls.append(stage)) for stage in JEV_REFRESH_STAGES}

    run_refresh_pipeline(
        corpus_sha256="a" * 64,
        state_path=state,
        actions=actions,
        allow_paid_jev=True,
    )
    run_refresh_pipeline(
        corpus_sha256="a" * 64,
        state_path=state,
        actions=actions,
        allow_paid_jev=True,
    )
    assert calls == list(JEV_REFRESH_STAGES)

    run_refresh_pipeline(
        corpus_sha256="b" * 64,
        state_path=state,
        actions=actions,
        allow_paid_jev=True,
    )
    assert calls == [*JEV_REFRESH_STAGES, *JEV_REFRESH_STAGES]


def test_refresh_pipeline_requires_explicit_paid_jev_authorization(tmp_path):
    import pytest

    from arvamusfestivali_transcripts.jev_refresh import JEV_REFRESH_STAGES, run_refresh_pipeline

    with pytest.raises(ValueError, match="allow-paid-jev"):
        run_refresh_pipeline(
            corpus_sha256="a" * 64,
            state_path=tmp_path / "state.json",
            actions={stage: lambda: None for stage in JEV_REFRESH_STAGES},
            allow_paid_jev=False,
        )


def test_cli_dry_run_reports_jev_only_stages_without_running_paid_work(
    tmp_path, write_topic_archive, capsys
):
    from arvamusfestivali_transcripts.jev_refresh import main

    write_topic_archive(tmp_path / "data" / "transcripts")

    assert main(["--root", str(tmp_path), "--year", "2026", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["stages"] == [
        "semantic_source",
        "jev_assignment",
        "jev_review",
        "jev_bubble_maps",
        "jev_treemaps",
        "jev_segment_maps",
        "jev_year_results",
        "jev_year_comparison",
        "site_data",
    ]
    assert plan["topic_refit"] is False
    assert plan["topic_renaming"] is False
