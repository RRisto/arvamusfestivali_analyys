import { filterSegments, normalizeText, stateToQuery } from "./search-core.mjs";

const MAX_RESULTS = 80;
let data;
let results = [];
let selectedSegmentId = "";
let timer;

const elements = {};

function stateFromControls() {
  return {
    query: elements.query.value.trim(),
    year: elements.year.value,
    category: elements.category.value,
    organizer: elements.organizer.value,
    episode: elements.talk.value,
    segment: selectedSegmentId,
  };
}

function setOptions(select, entries, currentValue) {
  const placeholder = select.options[0].cloneNode(true);
  select.replaceChildren(placeholder);
  entries.forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.append(option);
  });
  select.value = currentValue || "";
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function addHighlightedText(target, text, query) {
  const normalizedQuery = normalizeText(query);
  if (!normalizedQuery) {
    target.textContent = text;
    return;
  }
  const normalized = normalizeText(text);
  const index = normalized.indexOf(normalizedQuery);
  if (index < 0) {
    target.textContent = text;
    return;
  }
  // Normalization can change character positions; a safe approximate highlight is preferable to HTML injection.
  const start = Math.min(index, text.length);
  const end = Math.min(start + query.length, text.length);
  target.append(document.createTextNode(text.slice(0, start)));
  const mark = document.createElement("mark");
  mark.textContent = text.slice(start, end);
  target.append(mark, document.createTextNode(text.slice(end)));
}

function selectSegment(segment, updateUrl = true) {
  selectedSegmentId = segment.id;
  const talk = data.talks.find((item) => item.episode === segment.episode);
  const passages = data.segments
    .filter((item) => item.episode === segment.episode)
    .sort((a, b) => a.start - b.start);
  elements.reader.replaceChildren();
  const article = document.createElement("article");
  article.className = "reader-content";
  article.innerHTML = `<p class="eyebrow">${segment.year} · ${passages.length} teemalõiku</p><h2 id="reader-title"></h2><p class="reader-meta"></p><div class="reader-actions"></div><div class="passage-list"></div>`;
  article.querySelector("h2").textContent = talk?.title || segment.title;
  article.querySelector(".reader-meta").textContent = [segment.organizer, talk?.stage, talk?.moderator && `Moderaator: ${talk.moderator}`].filter(Boolean).join(" · ");
  const actions = article.querySelector(".reader-actions");
  if (talk?.page) {
    const source = document.createElement("a");
    source.className = "primary-button";
    source.href = talk.page;
    source.target = "_blank";
    source.rel = "noreferrer";
    source.textContent = "Ava salvestuse leht ↗";
    actions.append(source);
  }
  const list = article.querySelector(".passage-list");
  passages.forEach((passage) => {
    const section = document.createElement("section");
    section.className = `reader-passage${passage.id === segment.id ? " selected" : ""}`;
    section.dataset.readerSegment = passage.id;
    const passageHead = document.createElement("div");
    passageHead.className = "passage-head";
    const topic = document.createElement("strong");
    topic.textContent = passage.topic || passage.category;
    const audio = document.createElement("a");
    audio.href = passage.audio;
    audio.target = "_blank";
    audio.rel = "noreferrer";
    audio.textContent = `▶ ${formatTime(passage.start)}`;
    audio.setAttribute("aria-label", `Ajastatud helilink, ${formatTime(passage.start)}`);
    passageHead.append(topic, audio);
    const text = document.createElement("p");
    addHighlightedText(text, passage.text, elements.query.value);
    section.append(passageHead, text);
    list.append(section);
  });
  elements.reader.append(article);
  document.querySelectorAll(".result-card").forEach((card) => card.classList.toggle("selected", card.dataset.segmentId === segment.id));
  if (updateUrl) history.replaceState(null, "", stateToQuery(stateFromControls()));
  requestAnimationFrame(() => {
    const selectedPassage = elements.reader.querySelector(".reader-passage.selected");
    selectedPassage?.scrollIntoView({ behavior: "smooth", block: window.innerWidth < 960 ? "start" : "center" });
  });
}

function renderResults() {
  const state = stateFromControls();
  results = filterSegments(data.segments, state);
  elements.count.textContent = siteCommon.formatNumber(results.length);
  elements.limit.textContent = results.length > MAX_RESULTS ? `Näitan esimest ${MAX_RESULTS}` : "";
  elements.results.replaceChildren();
  if (!results.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Sobivaid lõike ei leitud. Proovi lühemat otsingut või eemalda mõni filter.";
    elements.results.append(empty);
  }
  results.slice(0, MAX_RESULTS).forEach((segment) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "result-card";
    button.dataset.segmentId = segment.id;
    const head = document.createElement("span");
    head.className = "result-head";
    head.textContent = `${segment.year} · ${formatTime(segment.start)} · ${segment.category}`;
    const title = document.createElement("strong");
    title.textContent = segment.title;
    const excerpt = document.createElement("span");
    excerpt.className = "result-excerpt";
    addHighlightedText(excerpt, segment.text.slice(0, 260), state.query);
    button.append(head, title, excerpt);
    button.addEventListener("click", () => selectSegment(segment));
    elements.results.append(button);
  });
  if (selectedSegmentId) {
    const selected = data.segments.find((segment) => segment.id === selectedSegmentId);
    if (selected) selectSegment(selected, false);
  }
  history.replaceState(null, "", stateToQuery(state));
}

function scheduleRender() {
  clearTimeout(timer);
  selectedSegmentId = "";
  timer = setTimeout(renderResults, 180);
}

function initializeControls(params) {
  const categoryNames = new Map(data.segments.map((item) => [item.category_id, item.category]));
  setOptions(elements.category, [...categoryNames].sort((a, b) => a[1].localeCompare(b[1], "et")), params.get("category"));
  const organizers = [...new Set(data.segments.map((item) => item.organizer).filter(Boolean))].sort((a, b) => a.localeCompare(b, "et"));
  setOptions(elements.organizer, organizers.map((value) => [value, value]), params.get("organizer"));
  const talks = [...data.talks].sort((a, b) => a.title.localeCompare(b.title, "et"));
  setOptions(elements.talk, talks.map((talk) => [talk.episode, `${talk.year} · ${talk.title}`]), params.get("episode"));
  elements.query.value = params.get("q") || "";
  elements.year.value = params.get("year") || "";
  selectedSegmentId = params.get("segment") || "";
  [elements.year, elements.category, elements.organizer, elements.talk].forEach((control) => control.addEventListener("change", scheduleRender));
  elements.query.addEventListener("input", scheduleRender);
  document.getElementById("clear-filters").addEventListener("click", () => {
    [elements.query, elements.year, elements.category, elements.organizer, elements.talk].forEach((control) => { control.value = ""; });
    selectedSegmentId = "";
    renderResults();
  });
}

async function init() {
  Object.assign(elements, {
    query: document.getElementById("search-query"),
    year: document.getElementById("year-filter"),
    category: document.getElementById("category-filter"),
    organizer: document.getElementById("organizer-filter"),
    talk: document.getElementById("talk-filter"),
    results: document.getElementById("search-results"),
    reader: document.getElementById("transcript-reader"),
    count: document.getElementById("result-count"),
    limit: document.getElementById("result-limit-note"),
  });
  try {
    data = await siteCommon.fetchJson("data/transcripts.json");
    data.segments.forEach((segment) => { segment._search = normalizeText(`${segment.title} ${segment.topic} ${segment.text} ${segment.organizer}`); });
    initializeControls(new URLSearchParams(window.location.search));
    renderResults();
  } catch (error) {
    siteCommon.showError(error);
    elements.results.innerHTML = '<p class="empty-state">Andmeid ei õnnestunud laadida.</p>';
  }
}

init();
