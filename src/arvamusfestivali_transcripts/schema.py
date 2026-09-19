from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Episode:
    id: str
    rss_guid: str
    title: str
    published_at: datetime
    published_raw: str
    page_url: str
    audio_url: str
    audio_bytes: int | None
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CatalogInfo:
    collection_id: int
    collection_name: str
    apple_page_url: str
    feed_url: str


@dataclass(frozen=True, slots=True)
class DownloadedAudio:
    path: Path
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    apple_collection_id: int
    apple_page_url: str
    feed_url: str
    resolved_at: datetime
    year: int
    episodes: tuple[Episode, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["resolved_at"] = self.resolved_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        payload["episodes"] = [episode_to_dict(item) for item in self.episodes]
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> CatalogSnapshot:
        payload = json.loads(text)
        payload["resolved_at"] = datetime.fromisoformat(
            payload["resolved_at"].replace("Z", "+00:00")
        )
        payload["episodes"] = tuple(episode_from_dict(item) for item in payload["episodes"])
        return cls(**payload)


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    payload = asdict(episode)
    payload["published_at"] = (
        episode.published_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    )
    return payload


def episode_from_dict(payload: dict[str, Any]) -> Episode:
    values = dict(payload)
    values["published_at"] = datetime.fromisoformat(values["published_at"].replace("Z", "+00:00"))
    return Episode(**values)
