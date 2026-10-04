"""Match official festival programmes and export enriched canonical talk metadata."""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/topic-analysis/results/jev-topic-analysis"
METADATA = ROOT / "data/topic-analysis/results/talk-metadata"
PROGRAMMES = {2025: "https://arvamusfestival.ee/2025-arhiiv/",
              2026: "https://arvamusfestival.ee/kava2026/"}
# Reviewed title variants in the 2025 archive. Preserve explicit evidence and
# episode scope, rather than accepting fuzzy similarities across the corpus.
REVIEWED = {
    "2247188696": (8845, "versus vs abbreviation"),
    "2247188699": (8844, "same learning-at-work title with reordered words"),
    "2254333412": (8746, "audio-project .alp suffix"),
    "2277338243": (8728, "quiz title adds mõtteharjutused koos; same Ivo Linna"),
    "2277338246": (8727, "parallelmaailmad spelling variant"),
    "2277338249": (8729, "same title with reordered liikuma panna"),
    "2280202457": (8804, "changed title; programme Sotsiodraama format; transcript "
                           "introduces Miina Pruuli and Grete Pastak, both listed panelists"),
    "2293106231": (8818, "Copy (2) upload suffix"),
    "2310594920": (8849, "audio-project .alp suffix"),
    "2310594926": (8850, "audio-project .alp suffix"),
    "2323554785": (8894, "recording includes Demokraatiatrenn; programme format agrees"),
    "2323554797": (8895, "programme adds Majandusarengu aruteluploki finaal subtitle"),
    "2323554800": (8886, "programme includes Demokraatiatrenn suffix"),
}


def normalize_title(title: str) -> str:
    title = html.unescape(unicodedata.normalize("NFC", title)).casefold()
    title = re.sub(r"\s*\(\d+\)\s*$", "", title)
    return "".join(c for c in title if c.isalnum())


def plain(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    events, snapshots = [], []
    for year, url in PROGRAMMES.items():
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        marker = re.search(r"\bevents\s*:\s*\[", response.text)
        if marker is None:
            raise ValueError(f"Programme events JSON not found: {url}")
        records, _ = json.JSONDecoder().raw_decode(response.text[marker.end() - 1:])
        (OUT / f"programme-{year}.html").write_text(response.text)
        (OUT / f"programme-{year}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        for event in records:
            events.append({**event, "programme_year": year, "programme_source": url})
        snapshots.append(dict(year=year, source=url, events=len(records),
                              page_sha256=hashlib.sha256(response.content).hexdigest()))
    dataset = ROOT / "data/topic-analysis/results/segment-dataset/all-segments.parquet"
    source_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    segments = pd.read_parquet(dataset)
    talks = segments[["episode_id", "title", "audio_sha256", "duplicate_episode_ids"]
                     ].drop_duplicates("episode_id")
    catalog_path = sorted((ROOT / "data/catalog").glob("*.json"))[-1]
    catalog = json.loads(catalog_path.read_text())
    catalog_lookup = {e["id"]: e for e in catalog["episodes"]}
    lookup = defaultdict(list)
    for event in events:
        lookup[normalize_title(event["title"])].append(event)
    matches, audit, metadata = [], [], []
    for talk in talks.itertuples():
        episode = catalog_lookup[talk.episode_id]
        candidates = lookup.get(normalize_title(talk.title), [])
        method, note = "normalized_exact", ""
        if not candidates and re.search(r"\s+[2-9]$", talk.title):
            candidates = lookup.get(normalize_title(re.sub(r"\s+[2-9]$", "", talk.title)), [])
            method = "normalized_exact_without_part_suffix"
        if talk.episode_id in REVIEWED:
            event_id, note = REVIEWED[talk.episode_id]
            candidates = [e for e in events if e["programme_year"] == 2025
                          and e["id"] == event_id]
            method = "reviewed_title_variant"
        candidates = [e for e in candidates if episode["published_at"][:10]
                      >= e["startDate"][:10]]
        if candidates:
            latest_year = max(e["programme_year"] for e in candidates)
            candidates = [e for e in candidates if e["programme_year"] == latest_year]
        # Repeated workshop sessions can share a title and organizer. Keep all
        # possible event IDs and only merge fields on which they agree.
        orgs = {plain(e.get("organizer", "")) for e in candidates}
        status = ("unmatched" if not candidates else "ambiguous_organizer"
                  if len(orgs) > 1 else "matched" if next(iter(orgs))
                  else "matched_no_organizer")
        chosen = candidates if status.startswith("matched") else []
        organizer = next(iter(orgs)) if chosen else ""
        evidence = json.dumps([dict(id=e["id"], title=e["title"],
            organizer=e["organizer"], date=e["startDate"], source=e["programme_source"])
            for e in chosen], ensure_ascii=False)
        organizer_source = chosen[0]["programme_source"] if chosen else ""
        record = dict(episode_id=talk.episode_id, title=talk.title, status=status,
                      match_method=method, match_note=note,
                      programme_event_ids=json.dumps([e["id"] for e in candidates]))
        audit.append(record)
        if organizer:
            matches.append(dict(episode_id=talk.episode_id, organizer=organizer,
                                source=organizer_source, evidence=evidence))
        enriched = {**episode, "episode_id": talk.episode_id,
                    "audio_sha256": talk.audio_sha256,
                    "duplicate_episode_ids": talk.duplicate_episode_ids,
                    "organizer": organizer, "organizer_source": organizer_source,
                    "organizer_evidence": evidence, "organizer_match_status": status,
                    "organizer_match_method": method, "organizer_match_note": note,
                    "programme_event_ids": record["programme_event_ids"],
                    "programme_year": chosen[0]["programme_year"] if chosen else None}
        fields = {"programme_title": "title", "programme_start": "startDate",
                  "programme_end": "endDate", "programme_moderator": "host",
                  "programme_panelists": "panelists", "programme_support_text": "supportText",
                  "programme_language": "language", "programme_recording_link": "link"}
        for dest, key in fields.items():
            values = {e.get(key, "") for e in chosen}
            enriched[dest] = next(iter(values)) if len(values) == 1 else ""
        stages = {e.get("stage", {}).get("name", "") for e in chosen}
        enriched["programme_stage"] = next(iter(stages)) if len(stages) == 1 else ""
        enriched["programme_curator"] = (enriched["programme_stage"].split("kureerib", 1)[1]
                                          .strip() if "kureerib" in enriched["programme_stage"]
                                          else "")
        metadata.append(enriched)
    pd.DataFrame(matches).to_csv(OUT / "programme-organizers.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "programme-match-audit.csv", index=False)
    table = pd.DataFrame(metadata)
    table.to_parquet(METADATA / "talks.parquet", index=False)
    table.to_csv(METADATA / "talks.csv", index=False)
    columns = [c for c in table if c.startswith("organizer") or c.startswith("programme")]
    enriched_segments = segments.merge(table[["episode_id", *columns]], on="episode_id",
                                       validate="many_to_one", sort=False)
    pd.testing.assert_frame_equal(enriched_segments[segments.columns], segments)
    enriched_segments.to_parquet(METADATA / "segments-with-metadata.parquet", index=False)
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == source_hash
    manifest = dict(retrieved_at=datetime.now(UTC).isoformat(), sources=snapshots,
        dataset_sha256=source_hash,
        catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        matched_talks=len(matches), canonical_talks=len(talks),
        programme_matched_talks=int(table.organizer_match_status.str.startswith("matched").sum()),
        status_counts=table.organizer_match_status.value_counts().to_dict(),
        matched_by_year=table.loc[table.organizer.ne("")].programme_year.value_counts().to_dict(),
        policy="Normalized titles and reviewed scoped variants; publication >= event; "
               "latest eligible programme; joint labels intact; moderators, curators and "
               "sponsors separate; missing organizer fields retained as missing.")
    for dest in [OUT / "programme-manifest.json", METADATA / "manifest.json"]:
        dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    (METADATA / "README.md").write_text(
        "# Canonical talk metadata enriched with festival programmes\n\n"
        f"{len(matches)}/{len(talks)} talks have explicit organizer labels. "
        "See manifest.json and ../jev-topic-analysis/programme-match-audit.csv for coverage.\n\n"
        "talks.parquet / talks.csv join canonical catalog metadata, audio deduplication "
        "provenance and programme fields. segments-with-metadata.parquet attaches these "
        "fields to all native fixed/semantic rows. The original dataset remains unchanged.\n\n"
        "organizer is the official label, including joint organizations. Moderator, "
        "stage curator and sponsor/support text are separate fields. Empty organizer "
        "fields are not inferred from these roles. Event IDs can list repeated sessions "
        "when titles and organizers agree but session timing is unresolved.\n")
    print(f"Organizers: {len(matches)}/{len(talks)}; programme matches: "
          f"{manifest['programme_matched_talks']}; {manifest['status_counts']}")


if __name__ == "__main__":
    main()
