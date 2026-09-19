from datetime import datetime

import pytest

from arvamusfestivali_transcripts.state import EpisodeStatus, PipelineState


def test_state_tracks_failure_and_retry(tmp_path) -> None:
    with PipelineState(tmp_path / "pipeline.sqlite3") as state:
        state.record_discovered("2400217815", 2026)
        state.mark_started("2400217815", EpisodeStatus.DOWNLOADING)
        state.mark_failed("2400217815", "download", "connection reset")
        record = state.status_for("2400217815")

        assert record.status is EpisodeStatus.FAILED
        assert record.failure_stage == "download"
        assert record.failure_message == "connection reset"
        assert state.retryable_ids(2026) == ("2400217815",)


def test_state_persists_records_across_connections(tmp_path) -> None:
    database = tmp_path / "pipeline.sqlite3"
    with PipelineState(database) as state:
        state.record_discovered("episode-1", 2025)
        state.mark_started("episode-1", EpisodeStatus.DOWNLOADING)

    with PipelineState(database) as state:
        record = state.status_for("episode-1")

    assert record.status is EpisodeStatus.DOWNLOADING
    assert record.year == 2025
    timestamp = datetime.fromisoformat(record.updated_at)
    assert timestamp.tzinfo is not None


def test_complete_clears_failure_data(tmp_path) -> None:
    with PipelineState(tmp_path / "pipeline.sqlite3") as state:
        state.record_discovered("episode-1", 2026)
        state.mark_failed("episode-1", "transcribe", "service unavailable")
        state.mark_complete("episode-1")
        record = state.status_for("episode-1")
        retryable = state.retryable_ids(2026)

    assert record.status is EpisodeStatus.COMPLETE
    assert record.failure_stage is None
    assert record.failure_message is None
    assert retryable == ()


def test_mutations_reject_unknown_ids_and_empty_failure_data(tmp_path) -> None:
    with PipelineState(tmp_path / "pipeline.sqlite3") as state:
        with pytest.raises(KeyError):
            state.mark_started("missing", EpisodeStatus.DOWNLOADING)
        with pytest.raises(KeyError):
            state.mark_complete("missing")
        with pytest.raises(KeyError):
            state.mark_failed("missing", "download", "broken")

        state.record_discovered("episode-1", 2026)
        with pytest.raises(ValueError):
            state.mark_failed("episode-1", "", "broken")
        with pytest.raises(ValueError):
            state.mark_failed("episode-1", "download", "")
