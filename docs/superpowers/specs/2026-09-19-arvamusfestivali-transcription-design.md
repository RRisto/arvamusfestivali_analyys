# Arvamusfestival Transcription Archive Design

## Purpose

Build a reproducible local pipeline that discovers Arvamusfestival podcast
episodes, downloads recordings for a selected publication year, transcribes
Estonian speech with timestamps, and commits compact canonical transcript JSON
files to Git for later analysis.

The first archive target is the 2026 publication year. Later years must use the
same commands and storage format without changes to the architecture.

## Confirmed Scope

- Resolve Apple Podcasts collection `1477431807` to its underlying RSS feed.
- Use the SoundCloud RSS feed as the episode and audio source:
  `https://feeds.soundcloud.com/users/soundcloud:users:96052562/sounds.rss`.
- Select episodes by RSS publication year, initially `2026`.
- Download audio as a resumable local cache rather than a permanent archive.
- Transcribe with the existing `estonian-audio-transcription` skill and retain
  timestamps.
- Store one canonical format per episode: JSON.
- Commit canonical transcript JSON files to Git.
- Preserve enough source metadata to generate a link to the original audio at
  any transcript cue's start time.
- Do not provide speaker diarization, translation, or manual transcript editing
  in the initial version.

As observed on 2026-09-19, the RSS feed exposes 500 episodes. The 2026 subset
contains 264 episodes, approximately 386 hours and 35 GiB of compressed audio.
The feed is a moving source, so every run must save a feed snapshot and derive
the selected set from that snapshot.

## Approach

Implement a small packaged Python command-line application. A feed-driven
pipeline gives deterministic metadata, year filtering, stable identifiers,
checksums, retries, and direct integration with the transcription skill. Generic
podcast downloaders provide less control over the archive contract, while
scraping Apple or SoundCloud player pages is unnecessarily fragile.

The command surface will support:

```text
arvamusfestivali-transcripts fetch --year 2026
arvamusfestivali-transcripts transcribe --year 2026
arvamusfestivali-transcripts run --year 2026
```

All commands are idempotent. Existing valid outputs are skipped unless the user
explicitly requests replacement.

## Components

### Catalog resolver

Given the Apple Podcasts collection ID, query Apple's public catalog endpoint
and extract the current RSS URL. Preserve the collection ID, Apple page URL,
resolved feed URL, and resolution time in the feed snapshot metadata.

### Feed reader

Download the RSS document, save the raw snapshot, parse episode metadata, and
select items whose publication timestamp falls in the requested calendar year.
Normalize dates to UTC while retaining the original RSS value. Identify an
episode primarily by its SoundCloud track ID, with RSS GUID as a secondary
identity field.

### Download cache

Download one enclosure at a time into `data/audio-cache/<year>/`. Write to a
`.part` path and atomically rename only after the expected byte count is met.
Compute SHA-256 while finalizing the download. Support safe retry and HTTP range
resume when the source server supports it.

### Transcription adapter

Invoke the existing local `estonian-audio-transcription` skill with
`--formats json`. Its TalTech Zipformer output supplies full text, cue-level
timestamps, token timestamps, model identity, and runtime settings.

The adapter converts the engine output into the archive schema. It keeps the
full text and cue-level timestamps but removes redundant raw token arrays to
limit Git growth. Model and runtime provenance remain in the canonical file.

### Archive writer

Enrich the transcript with podcast metadata, source URLs, duration, audio size,
and checksum. Validate the completed document and write it atomically. Delete
the cached MP3 only after validation succeeds.

### Run manifest

Maintain local resumable state for each episode: discovered, downloading,
downloaded, transcribing, complete, or failed. Record failure type and message
so targeted retries do not repeat successful work. Runtime state is generated
and excluded from Git; source snapshots and canonical transcripts are durable.

## Repository Layout

```text
src/arvamusfestivali_transcripts/
    __init__.py
    __main__.py
    catalog.py
    feed.py
    download.py
    transcription.py
    archive.py
    schema.py
data/
    catalog/
        <snapshot timestamp>.xml
        <snapshot timestamp>.json
    audio-cache/
        2026/
    transcripts/
        2026/
            <soundcloud-track-id>.json
    state/
        pipeline.sqlite3
tests/
    fixtures/
```

Git tracks source code, tests, feed snapshots, compact catalog metadata, and
canonical transcript JSON. Git ignores audio, partial downloads, transcription
engine intermediates, model files, the local state database, virtual
environments, and build output.

## Canonical Transcript Schema

Each episode has exactly one canonical JSON document:

```json
{
  "schema_version": 1,
  "episode": {
    "id": "2400217815",
    "rss_guid": "tag:soundcloud,2010:tracks/2400217815",
    "title": "Kelle vastutus on ennetus tervishoius_",
    "published_at": "2026-09-14T13:40:21Z",
    "apple_collection_id": 1477431807,
    "page_url": "https://soundcloud.com/arvamusfestival/example-episode",
    "audio_url": "https://feeds.soundcloud.com/stream/2400217815-arvamusfestival-example-episode.mp3",
    "audio_bytes": 86222137,
    "audio_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "duration_seconds": 5400.0
  },
  "transcription": {
    "model": {
      "name": "large",
      "repository": "TalTechNLP/streaming-zipformer-large.et-en",
      "revision": "pinned-model-revision-from-manifest",
      "precision": "int8"
    },
    "runtime": {
      "sherpa_onnx": "installed-runtime-version",
      "decoding_method": "modified_beam_search",
      "threads": 4
    },
    "text": "Complete transcript text.",
    "cues": [
      {
        "start_seconds": 12.4,
        "end_seconds": 20.7,
        "text": "First timestamped transcript cue."
      }
    ]
  }
}
```

The audio URL is stored once at episode level. A consumer generates a listening
link for a cue as `<audio_url>#t=<start_seconds>`, following the W3C temporal
media-fragment convention. Consumers should also expose `page_url` as a fallback
because browser handling of direct MP3 time fragments can vary.

## Validation Rules

A transcript is complete only when all of the following hold:

- Required episode identifiers and source URLs are present.
- Publication time belongs to the requested year.
- Downloaded byte count matches RSS metadata when that metadata is available.
- SHA-256 is present and correctly formatted.
- Audio duration is positive and finite.
- Transcript text and cue list are nonempty.
- Every cue has nonempty text, finite times, `0 <= start < end`, and an end not
  beyond audio duration except for a small documented rounding tolerance.
- Cues are ordered and do not move backward in time.
- Model identity, model revision, precision, and runtime version are present.
- The final JSON parses and validates against the repository's schema.

Failure leaves the source audio in the cache for retry and never creates or
replaces a canonical transcript.

## Error Handling and Recovery

- Network and server errors use bounded retries with backoff.
- Interrupted downloads retain a `.part` file for safe resume.
- A changed enclosure URL is accepted only after the catalog is refreshed; the
  new URL and checksum are recorded in the newly generated transcript.
- Duplicate feed entries with the same stable ID are reported and resolved
  deterministically rather than downloaded twice.
- Existing valid transcripts are immutable by default. Regeneration requires an
  explicit force option and should be committed separately for review.
- Failed episodes remain addressable by ID and can be retried independently.
- The pipeline emits a final summary containing selected, skipped, completed,
  failed, downloaded-byte, and deleted-cache counts.

## Testing Strategy

Unit tests use local fixtures and do not depend on live external services. They
cover:

- Apple catalog response parsing and RSS URL extraction.
- RSS parsing, UTC normalization, year filtering, and stable ID extraction.
- Duplicate and malformed episode handling.
- Safe filenames and deterministic transcript paths.
- Partial download resume and atomic finalization.
- Content-length and checksum mismatch failures.
- Conversion from transcription-engine JSON to canonical archive JSON.
- Removal of raw token arrays while retaining cue timestamps and provenance.
- Transcript schema and timestamp validation.
- Audio timestamp-link generation.
- Idempotent skips and targeted retries.

One public 2026 episode is the end-to-end acceptance case. The acceptance run
must fetch the episode, produce a valid canonical JSON transcript, confirm a
working source URL and timestamp-link value, and delete the cached audio. Bulk
processing begins only after that acceptance case succeeds.

## Operational Plan

1. Implement and test catalog resolution and feed snapshotting.
2. Implement year selection and manifest creation; dry-run the 2026 selection.
3. Implement resumable enclosure downloading and checksum capture.
4. Integrate the Estonian transcription skill and canonical JSON conversion.
5. Implement validation, atomic archive writes, and post-success cache cleanup.
6. Run the single-episode acceptance test.
7. Process 2026 episodes in resumable batches and commit transcript batches.
8. Add later years by invoking the same pipeline with another `--year` value.

## Dependencies and Constraints

- Python 3.12 and `uv` manage the application environment.
- FFmpeg is required to decode downloaded MP3 files.
- The first transcription downloads checksum-verified model assets from Hugging
  Face into the user's cache.
- The transcription model provides recognition timestamps, not forced alignment
  and not speaker diarization.
- The RSS feed currently exposes at most 500 episodes, so years older than the
  feed window may require a separate historical discovery source. That work is
  outside the initial 2026 phase but does not change the transcript schema.
