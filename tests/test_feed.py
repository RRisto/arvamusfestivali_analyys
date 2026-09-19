from datetime import UTC, datetime
from pathlib import Path

import httpx

from arvamusfestivali_transcripts.feed import fetch_feed, parse_duration, parse_feed, write_snapshot
from arvamusfestivali_transcripts.schema import CatalogSnapshot

FIXTURES = Path(__file__).parent / "fixtures"


def test_fetch_feed_returns_raw_response_bytes() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"feed bytes"))
    )

    assert fetch_feed(client, "https://example.com/feed.xml") == b"feed bytes"


def test_parse_feed_filters_year_and_deduplicates() -> None:
    episodes = parse_feed((FIXTURES / "feed.xml").read_bytes(), 2026)

    assert [episode.id for episode in episodes] == ["2400217815", "2400217000"]
    assert all(episode.published_at.year == 2026 for episode in episodes)
    assert episodes[0].title == "Esimene 2026 osa"
    assert episodes[0].published_at == datetime(2026, 9, 14, 13, 40, 21, tzinfo=UTC)
    assert episodes[0].duration_seconds == 5400.0
    assert episodes[1].duration_seconds == 2535.0


def test_parse_duration_accepts_supported_forms() -> None:
    assert parse_duration("90") == 90.0
    assert parse_duration("01:30") == 90.0
    assert parse_duration("1:01:30") == 3690.0


def test_write_snapshot_atomically_writes_xml_and_json(tmp_path: Path) -> None:
    snapshot = CatalogSnapshot(
        apple_collection_id=1477431807,
        apple_page_url="https://podcasts.apple.com/us/podcast/arvamusfestival/id1477431807",
        feed_url="https://feeds.soundcloud.com/users/soundcloud:users:96052562/sounds.rss",
        resolved_at=datetime(2026, 9, 19, 10, 0, tzinfo=UTC),
        year=2026,
        episodes=(),
    )

    xml_path, json_path = write_snapshot(tmp_path, snapshot, b"<rss />")

    assert xml_path == tmp_path / "data" / "catalog" / "20260919T100000Z-2026.xml"
    assert json_path == tmp_path / "data" / "catalog" / "20260919T100000Z-2026.json"
    assert xml_path.read_bytes() == b"<rss />"
    assert CatalogSnapshot.from_json(json_path.read_text(encoding="utf-8")) == snapshot
    assert not list(xml_path.parent.glob("*.tmp"))
