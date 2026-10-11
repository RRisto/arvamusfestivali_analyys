import assert from "node:assert/strict";
import test from "node:test";

import { filterSegments, normalizeText, stateToQuery } from "../../site/assets/search-core.mjs";

const segments = [
  { id: "a", text: "Õpetaja töö ja rõõm", title: "Haridus", year: 2026, category_id: "education", organizer: "Kool" },
  { id: "b", text: "Mets ja kliima", title: "Loodus", year: 2025, category_id: "nature", organizer: "Roheline" },
];

test("normalization ignores Estonian accents, case, and punctuation", () => {
  assert.equal(normalizeText("  ÕPETAJA, töö! "), "opetaja too");
});

test("segments are filtered by query and facets together", () => {
  assert.deepEqual(filterSegments(segments, { query: "opetaja", year: "2026", category: "education", organizer: "Kool" }).map((item) => item.id), ["a"]);
  assert.deepEqual(filterSegments(segments, { query: "mets", year: "2026", category: "", organizer: "" }), []);
});

test("URL state preserves a selected passage", () => {
  assert.equal(stateToQuery({ query: "õpetaja", year: "2026", category: "education", organizer: "", segment: "a" }), "?q=%C3%B5petaja&year=2026&category=education&segment=a");
});
