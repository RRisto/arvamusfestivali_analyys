"""Persistent state for resumable episode processing."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class EpisodeStatus(StrEnum):
    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    TRANSCRIBING = "transcribing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    year: int
    status: EpisodeStatus
    failure_stage: str | None
    failure_message: str | None
    updated_at: str


class PipelineState:
    """SQLite-backed state store for episode pipeline progress."""

    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(database)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                year INTEGER NOT NULL,
                status TEXT NOT NULL,
                failure_stage TEXT,
                failure_message TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def __enter__(self) -> PipelineState:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._connection.close()

    def record_discovered(self, episode_id: str, year: int) -> None:
        """Record an episode without regressing an already known state."""
        self._connection.execute(
            """
            INSERT OR IGNORE INTO episodes
                (episode_id, year, status, failure_stage, failure_message, updated_at)
            VALUES (?, ?, ?, NULL, NULL, ?)
            """,
            (episode_id, year, EpisodeStatus.DISCOVERED.value, _utc_now()),
        )
        self._connection.commit()

    def mark_started(self, episode_id: str, status: EpisodeStatus) -> None:
        """Set an episode to an active pipeline status."""
        if status not in {
            EpisodeStatus.DOWNLOADING,
            EpisodeStatus.DOWNLOADED,
            EpisodeStatus.TRANSCRIBING,
        }:
            raise ValueError(f"invalid started status: {status!r}")
        self._update_existing(
            episode_id,
            "status = ?, failure_stage = NULL, failure_message = NULL, updated_at = ?",
            (status.value, _utc_now()),
        )

    def mark_complete(self, episode_id: str) -> None:
        self._update_existing(
            episode_id,
            "status = ?, failure_stage = NULL, failure_message = NULL, updated_at = ?",
            (EpisodeStatus.COMPLETE.value, _utc_now()),
        )

    def mark_failed(self, episode_id: str, stage: str, message: str) -> None:
        if not stage or not message:
            raise ValueError("failure stage and message must be nonempty")
        self._update_existing(
            episode_id,
            "status = ?, failure_stage = ?, failure_message = ?, updated_at = ?",
            (EpisodeStatus.FAILED.value, stage, message, _utc_now()),
        )

    def status_for(self, episode_id: str) -> EpisodeRecord:
        row = self._connection.execute(
            "SELECT episode_id, year, status, failure_stage, failure_message, updated_at "
            "FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return EpisodeRecord(
            episode_id=row["episode_id"],
            year=row["year"],
            status=EpisodeStatus(row["status"]),
            failure_stage=row["failure_stage"],
            failure_message=row["failure_message"],
            updated_at=row["updated_at"],
        )

    def retryable_ids(self, year: int) -> tuple[str, ...]:
        rows = self._connection.execute(
            "SELECT episode_id FROM episodes WHERE year = ? AND status = ? ORDER BY episode_id",
            (year, EpisodeStatus.FAILED.value),
        ).fetchall()
        return tuple(row["episode_id"] for row in rows)

    def _update_existing(
        self, episode_id: str, assignments: str, values: tuple[object, ...]
    ) -> None:
        cursor = self._connection.execute(
            f"UPDATE episodes SET {assignments} WHERE episode_id = ?", values + (episode_id,)
        )
        if cursor.rowcount == 0:
            self._connection.rollback()
            raise KeyError(episode_id)
        self._connection.commit()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
