import { filterSegments, normalizeText, stateToQuery } from "./search-core.mjs";

const MAX_RESULTS = 80;
const ORPHAN_FRAGMENT_TEXT = new Set(["d.", "st.", "ad.", "s.", "gi.", "eid.", "des.", "ks."]);
let data;
let results = [];
let selectedSegmentId = "";
let timer;

const elements = {};

function displayTopicName(topic) {
  return topic === "Unassigned" ? "Määramata" : topic;
}

function isOrphanFragment(text) {
  const value = text.trim().toLocaleLowerCase("et");
  return !/\p{L}/u.test(value) || ORPHAN_FRAGMENT_TEXT.has(value);
}

function preparePassagesForDisplay(passages) {
  const prepared = passages.map((passage) => ({ ...passage }));
  const visible = [];
  prepared.forEach((passage) => {
    const text = passage.text || "";
    const leadingPunctuation = text.match(/^\s*([.!?…]+)\s*/u);
    if (leadingPunctuation && visible.length) {
      const previous = visible[visible.length - 1];
      if (!/[.!?…][\])}"'»”]*\s*$/u.test(previous.text.trimEnd())) {
        previous.text = `${previous.text.trimEnd()}${leadingPunctuation[1]}`;
      }
      passage.text = text.slice(leadingPunctuation[0].length);
    }
    if (!isOrphanFragment(passage.text || "")) visible.push(passage);
  });
  return visible;
}

function renderTalkCatalog() {
  const query = normalizeText(elements.query.value.trim());
  const talks = data.talks
    .filter((talk) => (!elements.year.value || String(talk.year) === elements.year.value)
      && (!elements.category.value || (talk.categories || []).includes(elements.category.value))
      && (!elements.organizer.value || talk.organizer === elements.organizer.value)
      && (!elements.talk.value || talk.episode === elements.talk.value)
      && normalizeText(`${talk.title} ${talk.organizer || ""}`).includes(query))
    .sort((a, b) => b.year - a.year || a.title.localeCompare(b.title, "et"));
  elements.talkCount.textContent = siteCommon.formatNumber(talks.length);
  elements.talkList.replaceChildren();
  if (!talks.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Sobivaid arutelusid ei leitud.";
    elements.talkList.append(empty);
    return;
  }
  talks.forEach((talk) => {
    const card = document.createElement("div");
    card.className = `talk-card-wrap${elements.talk.value === talk.episode ? " selected" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `talk-card${elements.talk.value === talk.episode ? " selected" : ""}`;
    button.setAttribute("aria-pressed", String(elements.talk.value === talk.episode));
    const heading = document.createElement("strong");
    heading.textContent = talk.title;
    const meta = document.createElement("span");
    meta.textContent = [talk.year, talk.organizer].filter(Boolean).join(" · ");
    button.append(heading, meta);
    button.addEventListener("click", () => openTalk(talk));
    const statsLink = document.createElement("a");
    statsLink.className = "talk-stats-link";
    statsLink.href = `diarization/${encodeURIComponent(talk.episode)}-stats.html`;
    statsLink.target = "_blank";
    statsLink.rel = "noreferrer";
    statsLink.textContent = "Kõnelejate statistika ↗";
    card.append(button, statsLink);
    elements.talkList.append(card);
  });
}

function setSearchMode() {
  const inTalk = Boolean(elements.talk.value);
  elements.talkCatalog.hidden = inTalk;
  elements.explorer.hidden = !inTalk;
  elements.searchLabel.textContent = inTalk ? "Otsi transkriptist" : "Otsi arutelu";
  elements.query.placeholder = inTalk ? "Otsi sõna või fraasi transkriptist…" : "Arutelu pealkiri või korraldaja…";
  elements.searchHelp.textContent = inTalk
    ? "Otsib valitud arutelu transkripti lõikudest; vali tulemus, et liikuda vastava ajastatud kohani."
    : "";
  elements.filterGrid.hidden = false;
  document.getElementById("clear-filters").hidden = false;
  document.getElementById("back-to-talks").hidden = !inTalk;
}

function openTalk(talk) {
  const firstPassage = data.segments
    .filter((segment) => segment.episode === talk.episode)
    .sort((a, b) => a.start - b.start)[0];
  if (!firstPassage) return;
  elements.talk.value = talk.episode;
  elements.year.value = String(talk.year);
  elements.category.value = "";
  elements.organizer.value = "";
  elements.query.value = "";
  selectedSegmentId = firstPassage.id;
  setSearchMode();
  renderTalkCatalog();
  renderResults();
  requestAnimationFrame(() => elements.reader.scrollIntoView({ behavior: "smooth", block: "start" }));
}

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
  const displayPassages = preparePassagesForDisplay(passages);
  elements.reader.replaceChildren();
  const article = document.createElement("article");
  article.className = "reader-content";
  article.innerHTML = `<p class="eyebrow">${segment.year} · ${displayPassages.length} teemalõiku</p><h2 id="reader-title"></h2><p class="reader-meta"></p><p class="reader-participants"></p><p class="speaker-note">Kõnelejatähised on automaatsed ega ole osalejatega nimeliselt kokku viidud.</p><div class="reader-actions"></div><div class="passage-list"></div>`;
  article.querySelector("h2").textContent = talk?.title || segment.title;
  article.querySelector(".reader-meta").textContent = [segment.organizer, talk?.stage].filter(Boolean).join(" · ");
  const participants = [talk?.moderator && `Moderaator: ${talk.moderator}`, talk?.panelists].filter(Boolean).join("\n");
  const participantLine = article.querySelector(".reader-participants");
  participantLine.textContent = participants ? `Osalejad: ${participants}` : "";
  participantLine.hidden = !participants;
  const actions = article.querySelector(".reader-actions");
  let audioPlayer;
  if (talk?.page) {
    const source = document.createElement("a");
    source.className = "primary-button";
    source.href = talk.page;
    source.target = "_blank";
    source.rel = "noreferrer";
    source.textContent = "Ava salvestuse leht ↗";
    actions.append(source);
  }
  if (talk) {
    const stats = document.createElement("a");
    stats.className = "secondary-button";
    stats.href = `diarization/${encodeURIComponent(talk.episode)}-stats.html`;
    stats.target = "_blank";
    stats.rel = "noreferrer";
    stats.textContent = "Kõnelejate statistika ↗";
    actions.append(stats);
  }
  if (talk?.audio) {
    audioPlayer = document.createElement("audio");
    audioPlayer.className = "transcript-audio-player";
    audioPlayer.controls = true;
    audioPlayer.preload = "metadata";
    audioPlayer.src = talk.audio;
    audioPlayer.setAttribute("aria-label", "Arutelu helipleier");
    actions.append(audioPlayer);
  }
  const list = article.querySelector(".passage-list");
  displayPassages.forEach((passage) => {
    const section = document.createElement("section");
    section.className = `reader-passage${passage.id === segment.id ? " selected" : ""}`;
    section.dataset.readerSegment = passage.id;
    const passageHead = document.createElement("div");
    passageHead.className = "passage-head";
    const topic = document.createElement("strong");
    topic.textContent = displayTopicName(passage.topic || passage.category);
    const playButton = document.createElement("button");
    playButton.type = "button";
    playButton.className = "passage-play-button";
    playButton.textContent = `▶ ${formatTime(passage.start)}`;
    playButton.setAttribute("aria-label", `Esita alates ${formatTime(passage.start)}`);
    playButton.disabled = !audioPlayer;
    playButton.addEventListener("click", () => {
      if (!audioPlayer) return;
      const seekAndPlay = () => {
        audioPlayer.currentTime = passage.start;
        audioPlayer.play().catch(() => {});
      };
      if (audioPlayer.readyState >= HTMLMediaElement.HAVE_METADATA) seekAndPlay();
      else audioPlayer.addEventListener("loadedmetadata", seekAndPlay, { once: true });
    });
    passageHead.append(topic, playButton);
    const text = document.createElement("p");
    addHighlightedText(text, passage.text, elements.query.value);
    section.append(passageHead);
    if (passage.speakers?.length) {
      const speakers = document.createElement("div");
      speakers.className = "passage-speakers";
      speakers.setAttribute("aria-label", "Lõigu kõneleja");
      passage.speakers.forEach((speaker) => {
        const chip = document.createElement("span");
        chip.className = "passage-speaker";
        const speakerLabel = speaker.name || (speaker.id === "UNASSIGNED" ? "Määramata" : speaker.id);
        chip.textContent = `${speakerLabel} · ${formatTime(speaker.seconds)}`;
        speakers.append(chip);
      });
      section.append(speakers);
    }
    section.append(text);
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
  if (!elements.talk.value) {
    setSearchMode();
    renderTalkCatalog();
    history.replaceState(null, "", stateToQuery(stateFromControls()));
    return;
  }
  setSearchMode();
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
  timer = setTimeout(() => {
    if (elements.talk.value) renderResults();
    else renderTalkCatalog();
  }, 180);
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
  document.getElementById("back-to-talks").addEventListener("click", () => {
    elements.talk.value = "";
    elements.query.value = "";
    elements.year.value = "";
    elements.category.value = "";
    elements.organizer.value = "";
    selectedSegmentId = "";
    setSearchMode();
    renderTalkCatalog();
    history.replaceState(null, "", window.location.pathname);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  document.getElementById("clear-filters").addEventListener("click", () => {
    [elements.query, elements.year, elements.category, elements.organizer, elements.talk].forEach((control) => { control.value = ""; });
    selectedSegmentId = "";
    renderResults();
  });
}

async function init() {
  Object.assign(elements, {
    query: document.getElementById("search-query"),
    searchLabel: document.getElementById("search-label"),
    searchHelp: document.getElementById("search-help"),
    filterGrid: document.querySelector(".filter-grid"),
    explorer: document.getElementById("explorer-layout"),
    talkCatalog: document.getElementById("talk-catalog"),
    year: document.getElementById("year-filter"),
    category: document.getElementById("category-filter"),
    organizer: document.getElementById("organizer-filter"),
    talk: document.getElementById("talk-filter"),
    talkList: document.getElementById("talk-list"),
    talkCount: document.getElementById("talk-count"),
    results: document.getElementById("search-results"),
    reader: document.getElementById("transcript-reader"),
    count: document.getElementById("result-count"),
    limit: document.getElementById("result-limit-note"),
  });
  try {
    data = await siteCommon.fetchJson("data/transcripts.json?v=speaker-turns1");
    data.segments.forEach((segment) => { segment._search = normalizeText(`${segment.title} ${segment.topic} ${segment.text} ${segment.organizer}`); });
    initializeControls(new URLSearchParams(window.location.search));
    renderResults();
  } catch (error) {
    siteCommon.showError(error);
    elements.results.innerHTML = '<p class="empty-state">Andmeid ei õnnestunud laadida.</p>';
  }
}

init();
