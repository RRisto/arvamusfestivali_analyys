from datetime import UTC, datetime, timedelta, timezone

from arvamusfestivali_transcripts.schema import (
    CatalogSnapshot,
    Episode,
    episode_from_dict,
    episode_to_dict,
)


def sample_episode() -> Episode:
    return Episode(
        id="2400217815",
        rss_guid="tag:soundcloud,2010:tracks/2400217815",
        title="Kelle vastutus on ennetus tervishoius_",
        published_at=datetime(2026, 9, 14, 13, 40, 21, tzinfo=UTC),
        published_raw="Mon, 14 Sep 2026 13:40:21 +0000",
        page_url="https://soundcloud.com/arvamusfestival/example-episode",
        audio_url="https://feeds.soundcloud.com/stream/2400217815-example.mp3",
        audio_bytes=86222137,
        duration_seconds=5400.0,
    )


def test_episode_round_trip() -> None:
    episode = sample_episode()
    assert episode_from_dict(episode_to_dict(episode)) == episode


def test_snapshot_json_is_stable() -> None:
    snapshot = CatalogSnapshot(
        apple_collection_id=1477431807,
        apple_page_url="https://podcasts.apple.com/us/podcast/arvamusfestival/id1477431807",
        feed_url="https://feeds.soundcloud.com/users/soundcloud:users:96052562/sounds.rss",
        resolved_at=datetime(2026, 9, 19, 10, 0, tzinfo=UTC),
        year=2026,
        episodes=(sample_episode(),),
    )
    rendered = snapshot.to_json()
    assert '"year": 2026' in rendered
    assert rendered.endswith("\n")
    assert CatalogSnapshot.from_json(rendered) == snapshot


def test_serialized_dates_are_normalized_to_utc() -> None:
    offset = timezone(timedelta(hours=3))
    episode = Episode(
        id="offset-episode",
        rss_guid="offset-guid",
        title="Offset episode",
        published_at=datetime(2026, 9, 14, 16, 40, 21, tzinfo=offset),
        published_raw="Mon, 14 Sep 2026 16:40:21 +0300",
        page_url="https://example.com/episode",
        audio_url="https://example.com/episode.mp3",
        audio_bytes=None,
        duration_seconds=60.0,
    )
    snapshot = CatalogSnapshot(
        apple_collection_id=1,
        apple_page_url="https://example.com/catalog",
        feed_url="https://example.com/feed.xml",
        resolved_at=datetime(2026, 9, 19, 13, 0, tzinfo=offset),
        year=2026,
        episodes=(episode,),
    )

    assert episode_to_dict(episode)["published_at"] == "2026-09-14T13:40:21Z"
    assert '"resolved_at": "2026-09-19T10:00:00Z"' in snapshot.to_json()
