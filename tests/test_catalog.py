from pathlib import Path

import httpx
import pytest

from arvamusfestivali_transcripts.catalog import resolve_catalog

FIXTURES = Path(__file__).parent / "fixtures"


def test_resolve_catalog_extracts_feed_url() -> None:
    payload = (FIXTURES / "apple_lookup.json").read_bytes()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    )

    info = resolve_catalog(client, 1477431807)

    assert info.collection_id == 1477431807
    assert info.feed_url.endswith("sounds.rss")


def test_resolve_catalog_rejects_missing_or_ambiguous_feed() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"results": []}))
    )

    with pytest.raises(ValueError, match="did not return one feed"):
        resolve_catalog(client, 1477431807)
