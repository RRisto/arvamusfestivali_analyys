import httpx

from arvamusfestivali_transcripts.schema import CatalogInfo

APPLE_LOOKUP_URL = "https://itunes.apple.com/lookup"


def resolve_catalog(client: httpx.Client, collection_id: int) -> CatalogInfo:
    """Resolve one Apple podcast collection to its authoritative RSS feed."""
    response = client.get(
        APPLE_LOOKUP_URL,
        params={"id": collection_id, "media": "podcast", "entity": "podcast"},
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    matches = [item for item in results if item.get("collectionId") == collection_id]
    if len(matches) != 1 or not matches[0].get("feedUrl"):
        raise ValueError(f"Apple lookup did not return one feed for {collection_id}")
    item = matches[0]
    return CatalogInfo(
        collection_id=collection_id,
        collection_name=str(item["collectionName"]),
        apple_page_url=str(item["collectionViewUrl"]),
        feed_url=str(item["feedUrl"]),
    )
