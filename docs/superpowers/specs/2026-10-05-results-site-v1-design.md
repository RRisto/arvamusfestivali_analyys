# Arvamusfestival Results Site V1 Design

## Goal

Build a local, static, shareable website that presents the main topic-analysis results, compares the 2025 and 2026 festival recordings, and lets visitors search and open timestamped transcript passages.

## Audience and primary tasks

The site serves festival organizers, researchers, journalists, and interested visitors. A visitor must be able to:

1. understand how topics are distributed across talks and recording time;
2. explore the two-dimensional topic landscape and organizer/topic relationships;
3. compare topic distributions between 2025 and 2026 without loading those charts on the overview page; and
4. search transcript text, choose a talk, read its passages in context, and open the original audio at the matching timestamp.

## Information architecture

The site has three separate pages with persistent navigation:

- `index.html` — Overview: headline corpus metrics, one treemap with a talk-count/talk-time toggle, a 2D semantic-segment scatter, and a topic-by-organizer heatmap.
- `compare.html` — 2025 ↔ 2026: normalized topic shares for each year and diverging difference charts for recording-time share and talk coverage.
- `transcripts.html` — Transcript explorer: text search, year/category filters, talk selection, passage display, and timestamped audio links.

This split keeps each page focused and avoids loading all interactive charts together. Overview charts are rendered only when their container becomes visible. The transcript dataset is fetched only on the transcript page.

## Data and interpretation

The build reads only existing tracked results:

- `jev-semantic-all/high-level-summary.csv` for combined topic totals;
- `jev-semantic-all/segment-maps/segment-coordinates.csv` for 2D positions and segment metadata;
- `jev-topic-analysis/year-comparison/main-topic-changes.csv` for year differences;
- `talk-metadata/talks.csv` for talk metadata; and
- `data/transcripts/2026/*.json` for passage text reconstructed from timestamped cues.

The build writes compact JSON under `site/data/`. Topic/organizer co-occurrence counts distinct talks once per topic-category/organizer pair. Year comparisons use within-year percentages because the two cohorts contain different numbers of talks. The UI must state that results describe available transcribed recordings rather than the complete festival programme.

## Interaction requirements

- Treemap metric toggle changes between distinct-talk count and recording hours.
- Scatter hover shows talk, year, detailed topic, high-level topic, organizer, and timestamp; clicking a point opens the transcript page for that segment.
- Organizer heatmap hover shows distinct-talk count; clicking filters the transcript explorer by category and organizer.
- Comparison metric toggle changes between recording-time share and talk coverage; difference bars use a centered zero line.
- Transcript search is case-insensitive and diacritic-tolerant, searches passage text plus talk/topic/organizer metadata, and updates from a debounced input.
- Search results and transcript passages expose accessible links to the source audio using W3C media fragments.
- Query parameters preserve `q`, `year`, `category`, `organizer`, `episode`, and `segment` selections for shareable URLs.

## Visual direction

Use a high-contrast editorial dashboard rather than a generic admin interface: deep forest navigation, white chart surfaces, a restrained amber accent, serif display headings, and category colors only inside visualizations. The first viewport begins with useful metrics and controls, not a marketing hero. Body text is at least 16px, controls at least 14px, layouts work on mobile, focus states are visible, and charts have textual summaries or tables where practical.

## Technical approach

Use plain HTML, CSS, and JavaScript under `site/`, with Plotly loaded from its pinned CDN build for interactive treemap, scatter, and heatmap rendering. A Python standard-library generator creates deterministic JSON assets; it requires no analysis dependencies or API keys. Pytest covers generator behavior, source-data integrity, output contracts, and required page landmarks. The site is served locally with `python -m http.server` because browsers restrict `fetch()` from `file://` URLs.

## V1 boundaries

V1 does not add authentication, editing, annotations, backend search, new topic modelling, statistical-significance claims, or publishing/deployment. It does not embed all 73 MB of raw transcript JSON; it presents the complete sequence of saved semantic passages for each selected talk.

