export function normalizeText(value) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("et")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

export function filterSegments(segments, state) {
  const query = normalizeText(state.query);
  return segments.filter((segment) => {
    if (state.year && String(segment.year) !== String(state.year)) return false;
    if (state.category && segment.category_id !== state.category) return false;
    if (state.organizer && segment.organizer !== state.organizer) return false;
    if (state.episode && segment.episode !== state.episode) return false;
    if (!query) return true;
    const haystack = segment._search || normalizeText(`${segment.title} ${segment.topic} ${segment.text} ${segment.organizer}`);
    return haystack.includes(query);
  });
}

export function stateToQuery(state) {
  const params = new URLSearchParams();
  const keys = [["query", "q"], ["year", "year"], ["category", "category"], ["organizer", "organizer"], ["episode", "episode"], ["segment", "segment"]];
  keys.forEach(([key, parameter]) => {
    if (state[key]) params.set(parameter, state[key]);
  });
  const query = params.toString();
  return query ? `?${query}` : "";
}
