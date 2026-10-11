# Arvamusfestival Results Site V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested three-page static results website backed by the repository's published topic-analysis and transcript data.

**Architecture:** A standard-library Python build script converts tracked CSV/JSON sources into compact page-specific JSON. Three static HTML pages share one visual system and small JavaScript modules; Plotly renders interactive charts, while transcript search and routing remain dependency-free browser code.

**Tech Stack:** Python 3.12 standard library, pytest, HTML5, CSS, ES modules, Plotly.js 3.1.0 CDN.

**Spec:** `docs/superpowers/specs/2026-10-05-results-site-v1-design.md`

## Global Constraints

- Read existing tracked outputs only; do not invoke model APIs or refit analysis models.
- Keep overview, comparison, and transcript exploration on separate pages.
- Use within-year shares for cross-year comparison and label corpus limitations.
- Reconstruct semantic-passage text from source transcript cues without adding PyArrow/Pandas dependencies.
- Preserve `.superpowers/`, `check_progress.py`, and `check_state.py` unchanged.
- Serve locally over HTTP; do not claim `file://` compatibility.

## Review Focus

- Estonian diacritics and case variations must match transcript search after normalization.
- A segment crossing several cues must contain every overlapping cue exactly once and in timestamp order.
- Missing transcript archives must fail the build with the episode ID instead of silently emitting blank text.
- Duplicate segments or talk/category rows must not inflate organizer co-occurrence counts.
- Empty searches and unmatched filters must show a useful result count/empty state without JavaScript errors.

---

### Task 1: Deterministic site-data builder

**Files:**
- Create: `scripts/build_results_site.py`
- Create: `tests/site/test_build_results_site.py`
- Create: `tests/site/__init__.py`
- Generate: `site/data/overview.json`
- Generate: `site/data/comparison.json`
- Generate: `site/data/transcripts.json`

**Interfaces:**
- Consumes: tracked CSVs plus transcript archive JSON described in the spec.
- Produces: `build_site_data(root: Path, output_dir: Path) -> dict[str, Path]`, `normalize_search_text(value: str) -> str`, and three versioned JSON payloads.

- [ ] **Step 1: Write failing generator tests**

Create fixture-sized CSV and transcript JSON inputs that assert cue overlap, diacritic normalization, distinct-talk co-occurrence, year deltas, source links, missing-archive errors, and versioned output keys.

- [ ] **Step 2: Verify the tests fail for the missing module**

Run: `data/state/af-topic-env/Scripts/python.exe -m pytest tests/site/test_build_results_site.py -q`

Expected: collection failure because `scripts.build_results_site` does not exist.

- [ ] **Step 3: Implement the minimal standard-library builder**

Parse CSV with `csv.DictReader`, transcript JSON with `json`, reconstruct text from overlapping cues, aggregate category/organizer distinct-talk counts with sets, serialize compact UTF-8 JSON with deterministic ordering, and expose a CLI with `--root` and `--output`.

- [ ] **Step 4: Verify generator tests and build real assets**

Run:

```powershell
data/state/af-topic-env/Scripts/python.exe -m pytest tests/site/test_build_results_site.py -q
data/state/af-topic-env/Scripts/python.exe scripts/build_results_site.py --root . --output site/data
```

Expected: tests pass and all three JSON files are created with nonzero records.

### Task 2: Shared shell and overview page

**Files:**
- Create: `site/index.html`
- Create: `site/assets/styles.css`
- Create: `site/assets/common.js`
- Create: `site/assets/overview.js`
- Create: `tests/site/test_site_pages.py`

**Interfaces:**
- Consumes: `site/data/overview.json` and query links to `transcripts.html`.
- Produces: responsive navigation, metric cards, treemap metric toggle, lazy scatter, and topic/organizer heatmap.

- [ ] **Step 1: Write failing page-contract tests**

Assert page title/description, navigation to all three routes, accessible chart-region labels, treemap toggle controls, Plotly version pin, favicon, corpus caveat, and transcript deep-link hooks.

- [ ] **Step 2: Verify contract tests fail**

Run: `data/state/af-topic-env/Scripts/python.exe -m pytest tests/site/test_site_pages.py -q`

Expected: failure because the site files are missing.

- [ ] **Step 3: Implement the overview slice**

Create semantic HTML and the shared visual system, load `overview.json`, render real corpus metrics, and implement Plotly treemap/scatter/heatmap views with `IntersectionObserver`, keyboard-accessible controls, error messages, and transcript deep links.

- [ ] **Step 4: Verify overview contracts**

Run: `data/state/af-topic-env/Scripts/python.exe -m pytest tests/site/test_site_pages.py -q`

Expected: overview tests pass.

### Task 3: Separate year-comparison page

**Files:**
- Create: `site/compare.html`
- Create: `site/assets/compare.js`
- Modify: `tests/site/test_site_pages.py`

**Interfaces:**
- Consumes: `site/data/comparison.json`.
- Produces: 2025/2026 normalized distribution chart, metric toggle, centered difference chart, and largest-increase/decrease summaries.

- [ ] **Step 1: Add failing comparison-page tests**

Assert a separate route, both year labels, comparison metric controls, corpus-comparability caveat, and difference-chart region.

- [ ] **Step 2: Verify the new tests fail**

Run: `data/state/af-topic-env/Scripts/python.exe -m pytest tests/site/test_site_pages.py -q`

Expected: failures for missing comparison markup/script.

- [ ] **Step 3: Implement the comparison page**

Render grouped within-year shares and a sorted diverging delta chart; update narrative highlights when the selected metric changes and link categories into transcript filters.

- [ ] **Step 4: Verify comparison contracts**

Run: `data/state/af-topic-env/Scripts/python.exe -m pytest tests/site/test_site_pages.py -q`

Expected: comparison tests pass.

### Task 4: Transcript search and detail page

**Files:**
- Create: `site/transcripts.html`
- Create: `site/assets/transcripts.js`
- Create: `tests/site/test_transcript_search.js`
- Modify: `tests/site/test_site_pages.py`

**Interfaces:**
- Consumes: `site/data/transcripts.json` and URL query parameters.
- Produces: diacritic-tolerant segment search, year/category/organizer filters, selected-talk passage reader, source-audio links, result counts, and shareable URL state.

- [ ] **Step 1: Add failing transcript page and JavaScript tests**

Assert required controls/landmarks in pytest. Use Node's built-in test runner against exported pure functions for normalization, filtering, empty-state behavior, and query serialization.

- [ ] **Step 2: Verify transcript tests fail**

Run:

```powershell
data/state/af-topic-env/Scripts/python.exe -m pytest tests/site/test_site_pages.py -q
node --test tests/site/test_transcript_search.js
```

Expected: failures because transcript markup and module exports are missing.

- [ ] **Step 3: Implement transcript exploration**

Build accessible form controls and result cards, debounce search, render the selected talk's passages in time order, mark the selected segment, update URL parameters without reload, and keep external audio links explicit.

- [ ] **Step 4: Verify transcript tests**

Run both commands from Step 2; expected: all pass.

### Task 5: End-to-end verification and handoff

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: complete generated site.
- Produces: documented build/serve workflow and verified local preview.

- [ ] **Step 1: Add README usage instructions**

Document the data build command and `python -m http.server 8765 --directory site`, including the local URL and why HTTP serving is required.

- [ ] **Step 2: Run focused verification**

Run:

```powershell
data/state/af-topic-env/Scripts/python.exe -m pytest tests/site -q
node --test tests/site/test_transcript_search.js
data/state/af-topic-env/Scripts/python.exe scripts/build_results_site.py --root . --output site/data
```

Expected: all site tests pass and assets rebuild deterministically.

- [ ] **Step 3: Run repository regression checks**

Run: `data/state/af-topic-env/Scripts/python.exe -m pytest -q`

Expected baseline caveat: current environment may still fail collection at `tests/topic_analysis/test_topic_naming.py` because `openai` is not installed; report this separately. Run `ruff check scripts/build_results_site.py tests/site` and require zero new lint errors.

- [ ] **Step 4: Serve and inspect all routes**

Start the local server, open Overview, 2025 ↔ 2026, and Transcripts, verify chart rendering, responsive layout, search, filters, transcript selection, and timestamped links, then keep the preview available for user review.

