import re
from datetime import UTC
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

import httpx

from arvamusfestivali_transcripts.schema import CatalogSnapshot, Episode

SOUNDCLOUD_ID = re.compile(r"tracks/(\d+)$")
ITUNES_DURATION = "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration"


def fetch_feed(client: httpx.Client, feed_url: str) -> bytes:
    """Fetch an RSS feed using the caller's configured HTTP client."""
    response = client.get(feed_url)
    response.raise_for_status()
    return response.content


def required_text(item: ElementTree.Element, path: str) -> str:
    """Return nonempty element text or explain which RSS value is missing."""
    element = item.find(path)
    if element is None or element.text is None or not element.text.strip():
        raise ValueError(f"missing required RSS element: {path}")
    return element.text.strip()


def parse_duration(value: str) -> float:
    """Convert SoundCloud's seconds, MM:SS, or HH:MM:SS duration into seconds."""
    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"invalid duration: {value}")


def parse_feed(xml_bytes: bytes, year: int) -> tuple[Episode, ...]:
    """Parse one RSS feed, retaining only the requested year and first ID occurrence."""
    root = ElementTree.fromstring(xml_bytes)
    by_id: dict[str, Episode] = {}
    for item in root.findall("./channel/item"):
        guid = required_text(item, "guid")
        match = SOUNDCLOUD_ID.search(guid)
        enclosure = item.find("enclosure")
        if match is None or enclosure is None or not enclosure.get("url"):
            continue
        published_raw = required_text(item, "pubDate")
        published_at = parsedate_to_datetime(published_raw).astimezone(UTC)
        if published_at.year != year:
            continue
        episode = Episode(
            id=match.group(1),
            rss_guid=guid,
            title=required_text(item, "title"),
            published_at=published_at,
            published_raw=published_raw,
            page_url=required_text(item, "link"),
            audio_url=enclosure.get("url", ""),
            audio_bytes=int(enclosure.get("length")) if enclosure.get("length") else None,
            duration_seconds=parse_duration(required_text(item, ITUNES_DURATION)),
        )
        by_id.setdefault(episode.id, episode)
    return tuple(sorted(by_id.values(), key=lambda item: (item.published_at, item.id)))


def write_snapshot(
    root: Path, snapshot: CatalogSnapshot, xml_bytes: bytes
) -> tuple[Path, Path]:
    """Atomically persist matching raw RSS and parsed catalog snapshot files."""
    catalog_dir = root / "data" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    timestamp = snapshot.resolved_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}-{snapshot.year}"
    xml_path = catalog_dir / f"{stem}.xml"
    json_path = catalog_dir / f"{stem}.json"

    xml_tmp = xml_path.with_suffix(".xml.tmp")
    json_tmp = json_path.with_suffix(".json.tmp")
    xml_tmp.write_bytes(xml_bytes)
    json_tmp.write_text(snapshot.to_json(), encoding="utf-8")
    xml_tmp.replace(xml_path)
    json_tmp.replace(json_path)
    return xml_path, json_path
