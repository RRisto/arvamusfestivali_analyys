from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: set[str] = set()
        self.scripts: set[str] = set()
        self.buttons: list[dict[str, str | None]] = []
        self.labels: set[str] = set()
        self.title = ""
        self.text: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
        if label := attributes.get("aria-label"):
            self.labels.add(label)
        if tag == "a" and attributes.get("href"):
            self.links.add(str(attributes["href"]))
        if tag == "script" and attributes.get("src"):
            self.scripts.add(str(attributes["src"]))
        if tag == "button":
            self.buttons.append(attributes)
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.text.append(text)
            if self._in_title:
                self.title += text


def parse_page(name: str) -> PageParser:
    parser = PageParser()
    parser.feed((SITE / name).read_text(encoding="utf-8"))
    return parser


def test_overview_exposes_navigation_metrics_and_accessible_charts() -> None:
    page = parse_page("index.html")

    assert "Arvamusfestivali teemavaade" in page.title
    assert {"index.html", "compare.html", "transcripts.html"} <= page.links
    assert {"metric-talks", "metric-segments", "metric-categories"} <= page.ids
    assert {"treemap-chart", "treemap-breadcrumb", "scatter-chart", "organizer-chart"} <= page.ids
    assert {
        "Teemade jaotus",
        "Teemade 2D kaart",
        "Teemade ja korraldajate seosed",
    } <= page.labels
    assert {button.get("data-metric") for button in page.buttons} >= {"talks", "hours"}
    assert {button.get("data-scatter-view") for button in page.buttons} >= {
        "topics",
        "segments",
    }
    assert "https://cdn.plot.ly/plotly-3.1.0.min.js" in page.scripts
    assert "assets/overview.js?v=topic-bubbles3" in page.scripts
    assert any("transkribeeritud salvestisi" in text for text in page.text)


def test_overview_has_metadata_favicon_and_error_surface() -> None:
    source = (SITE / "index.html").read_text(encoding="utf-8")

    assert '<meta name="description"' in source
    assert '<link rel="icon" type="image/svg+xml"' in source
    assert 'id="page-error"' in source
    assert 'aria-live="polite"' in source


def test_comparison_is_a_separate_accessible_page() -> None:
    page = parse_page("compare.html")

    assert "2025 ja 2026" in page.title
    assert {"index.html", "compare.html", "transcripts.html"} <= page.links
    assert {"comparison-summary", "distribution-chart", "difference-chart"} <= page.ids
    assert {
        "2025. ja 2026. aasta teemade jaotus",
        "Teemade muutus protsendipunktides",
    } <= page.labels
    assert {button.get("data-compare-metric") for button in page.buttons} >= {
        "time_share",
        "talk_coverage",
    }
    assert "https://cdn.plot.ly/plotly-3.1.0.min.js" in page.scripts
    assert "assets/compare.js" in page.scripts
    assert any("transkribeeritud salvestiste" in text for text in page.text)


def test_transcript_explorer_exposes_filters_results_and_reader() -> None:
    page = parse_page("transcripts.html")

    assert "Transkriptide otsing" in page.title
    assert {"index.html", "compare.html", "transcripts.html"} <= page.links
    assert {
        "search-query",
        "year-filter",
        "category-filter",
        "organizer-filter",
        "search-results",
        "transcript-reader",
        "result-count",
    } <= page.ids
    assert "assets/transcripts.js" in page.scripts
    assert any("Ajastatud helilink" in text for text in page.text)


def test_local_page_assets_and_generated_data_are_present() -> None:
    for name in ("index.html", "compare.html", "transcripts.html"):
        page = parse_page(name)
        for script in page.scripts:
            if not script.startswith("http"):
                target = script.split("?", 1)[0]
                assert (SITE / target).is_file(), f"{name} references missing {target}"
        for link in page.links:
            target = link.split("?", 1)[0]
            if target.endswith(".html"):
                assert (SITE / target).is_file(), f"{name} links to missing {target}"

    overview = json.loads((SITE / "data" / "overview.json").read_text(encoding="utf-8"))
    comparison = json.loads((SITE / "data" / "comparison.json").read_text(encoding="utf-8"))
    transcripts = json.loads((SITE / "data" / "transcripts.json").read_text(encoding="utf-8"))
    assert {key: overview["metrics"][key] for key in ("talks", "segments", "categories")} == {
        "talks": 219,
        "segments": 4132,
        "categories": 21,
    }
    assert comparison["cohorts"]["2025"]["talks"] == 114
    assert comparison["cohorts"]["2026"]["talks"] == 105
    assert len(transcripts["talks"]) == 219
    assert len(transcripts["segments"]) == 4132
