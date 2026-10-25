export function buildCategoryBubbles(scatter, categories) {
  const coordinates = new Map();
  scatter.forEach((row) => {
    const current = coordinates.get(row.category_id) || { x: 0, y: 0, count: 0 };
    current.x += Number(row.x);
    current.y += Number(row.y);
    current.count += 1;
    coordinates.set(row.category_id, current);
  });

  return categories.flatMap((category) => {
    const coordinate = coordinates.get(category.id);
    if (!coordinate?.count) return [];
    return [{
      id: category.id,
      name: category.name,
      x: coordinate.x / coordinate.count,
      y: coordinate.y / coordinate.count,
      talks: category.talks,
      segments: category.segments,
      hours: category.hours,
      color: category.color,
    }];
  });
}

export function buildTopicCooccurrenceMatrix(scatter, categories) {
  const categoryNames = new Map(categories.map((category) => [category.id, category.name]));
  const topicsByTalk = new Map();
  scatter.forEach((row) => {
    const name = categoryNames.get(row.category_id);
    if (!name || row.category_id === "unassigned" || name === "Määramata") return;
    if (!topicsByTalk.has(row.episode)) topicsByTalk.set(row.episode, new Set());
    topicsByTalk.get(row.episode).add(row.category_id);
  });

  const talkTotals = new Map();
  topicsByTalk.forEach((topicIds) => topicIds.forEach((id) =>
    talkTotals.set(id, (talkTotals.get(id) || 0) + 1)));
  const categoryIds = [...talkTotals]
    .sort(([leftId, leftTotal], [rightId, rightTotal]) =>
      rightTotal - leftTotal
      || categoryNames.get(leftId).localeCompare(categoryNames.get(rightId), "et"))
    .map(([id]) => id);
  const positions = new Map(categoryIds.map((id, index) => [id, index]));
  const z = categoryIds.map((_, row) =>
    categoryIds.map((__, column) => row === column ? null : 0));

  topicsByTalk.forEach((topicIds) => {
    const indexes = [...topicIds].map((id) => positions.get(id)).sort((a, b) => a - b);
    indexes.forEach((left, offset) => {
      indexes.slice(offset + 1).forEach((right) => {
        z[left][right] += 1;
        z[right][left] += 1;
      });
    });
  });

  return {
    categoryIds,
    names: categoryIds.map((id) => categoryNames.get(id)),
    z,
  };
}

export function buildSubtopicCooccurrenceMatrix(scatter, categories) {
  const categoryNames = new Map(categories.map((category) => [category.id, category.name]));
  const subtopics = new Map();
  const subtopicsByTalk = new Map();
  scatter.forEach((row) => {
    const topic = row.topic || "Määramata";
    const categoryName = categoryNames.get(row.category_id);
    if (!categoryName || row.category_id === "unassigned" || topic === "Määramata") return;
    const id = `${row.category_id}\u0000${topic}`;
    const current = subtopics.get(id) || {
      categoryId: row.category_id,
      topic,
      episodes: new Set(),
    };
    current.episodes.add(row.episode);
    subtopics.set(id, current);
    if (!subtopicsByTalk.has(row.episode)) subtopicsByTalk.set(row.episode, new Set());
    subtopicsByTalk.get(row.episode).add(id);
  });

  const cooccurrenceTotals = new Map();
  subtopicsByTalk.forEach((topicIds) => {
    topicIds.forEach((id) =>
      cooccurrenceTotals.set(id, (cooccurrenceTotals.get(id) || 0) + topicIds.size - 1));
  });
  const ids = categories.flatMap((category) => [...subtopics]
    .filter(([id, subtopic]) =>
      subtopic.categoryId === category.id && (cooccurrenceTotals.get(id) || 0) >= 3)
    .sort(([, left], [, right]) =>
      right.episodes.size - left.episodes.size || left.topic.localeCompare(right.topic, "et"))
    .map(([id]) => id));
  const positions = new Map(ids.map((id, index) => [id, index]));
  const z = ids.map((_, row) => ids.map((__, column) => row === column ? null : 0));
  subtopicsByTalk.forEach((topicIds) => {
    const indexes = [...topicIds]
      .map((id) => positions.get(id))
      .filter((index) => index !== undefined)
      .sort((a, b) => a - b);
    indexes.forEach((left, offset) => {
      indexes.slice(offset + 1).forEach((right) => {
        z[left][right] += 1;
        z[right][left] += 1;
      });
    });
  });

  return {
    ids,
    labels: ids.map((id) => {
      const subtopic = subtopics.get(id);
      return `${categoryNames.get(subtopic.categoryId)} · ${subtopic.topic}`;
    }),
    z,
  };
}

export function buildCooccurrenceLayout(chartLayout, isSubtopics, itemCount) {
  const size = Math.max(900, itemCount * 18 + 420);
  return {
    ...chartLayout,
    ...(isSubtopics
      ? { autosize: false, width: size, height: size }
      : { autosize: true }),
    xaxis: {
      tickangle: isSubtopics ? -90 : -45,
      tickfont: { size: isSubtopics ? 8 : 10 },
      side: "bottom",
      automargin: true,
      fixedrange: true,
    },
    yaxis: {
      autorange: "reversed",
      automargin: true,
      fixedrange: true,
      tickfont: { size: isSubtopics ? 8 : 10 },
    },
    margin: isSubtopics
      ? { l: 370, r: 50, t: 24, b: 370 }
      : { l: 285, r: 30, t: 24, b: 280 },
  };
}

export function buildTopicBubbles(scatter, categories) {
  const groups = new Map();
  scatter.forEach((row) => {
    const name = row.topic || "Määramata";
    const key = `${row.category_id}\u0000${name}`;
    const current = groups.get(key) || {
      name,
      categoryId: row.category_id,
      x: 0,
      y: 0,
      episodes: new Set(),
      segments: 0,
      hours: 0,
    };
    current.x += Number(row.x);
    current.y += Number(row.y);
    current.episodes.add(row.episode);
    current.segments += 1;
    current.hours += (Number(row.end) - Number(row.start)) / 3600;
    groups.set(key, current);
  });

  return categories.flatMap((category) => [...groups.values()]
    .filter((group) => group.categoryId === category.id)
    .sort((left, right) => left.name.localeCompare(right.name, "et"))
    .map((group) => ({
      id: `${category.id}\u0000${group.name}`,
      name: group.name,
      categoryId: category.id,
      categoryName: category.name,
      x: group.x / group.segments,
      y: group.y / group.segments,
      talks: group.episodes.size,
      segments: group.segments,
      hours: group.hours,
      color: category.color,
    })));
}

export function buildTopicScatterTraces(bubbles, categories) {
  const largest = Math.max(...bubbles.map((row) => row.talks));
  const sizeReference = (2 * largest) / (58 ** 2);
  return categories.flatMap((category) => {
    const rows = bubbles.filter((row) => row.categoryId === category.id);
    if (!rows.length) return [];
    return [{
      type: "scatter",
      mode: "markers",
      name: category.name,
      x: rows.map((row) => row.x),
      y: rows.map((row) => row.y),
      customdata: rows.map((row) => [row.categoryId, row.categoryName, row.name, row.talks, row.segments, row.hours]),
      marker: {
        size: rows.map((row) => row.talks),
        sizemode: "area",
        sizeref: sizeReference,
        sizemin: 8,
        color: category.color,
        opacity: .8,
        line: { color: "#ffffff", width: 1.25 },
      },
      hovertemplate: "<b>%{customdata[2]}</b><br>%{customdata[1]}<br>%{customdata[3]} arutelu<br>%{customdata[4]} teemalõiku<br>%{customdata[5]:.1f} tundi<extra></extra>",
    }];
  });
}

export function buildTopicScatterLayout(chartLayout) {
  return {
    ...chartLayout,
    hovermode: "closest",
    legend: {
      orientation: "h",
      y: -0.1,
      font: { size: 10 },
      itemsizing: "constant",
    },
    xaxis: { visible: false },
    yaxis: { visible: false },
    margin: { l: 22, r: 22, t: 30, b: 140 },
  };
}

function passageTime(seconds) {
  const total = Math.max(0, Math.round(Number(seconds)));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function aggregateTopics(segments) {
  const topics = new Map();
  segments.forEach((segment) => {
    const current = topics.get(segment.topic) || {
      episodes: new Set(),
      hours: 0,
    };
    current.episodes.add(segment.episode);
    current.hours += (Number(segment.end) - Number(segment.start)) / 3600;
    topics.set(segment.topic, current);
  });
  return [...topics.entries()].sort(([left], [right]) => left.localeCompare(right, "et"));
}

function topicDisplayName(topic) {
  return topic === "Unassigned" ? "Määramata" : topic;
}

export function buildTreemapLevel(data, selection, metric) {
  if (!selection.categoryId) {
    return data.categories.flatMap((category) => {
      const topics = aggregateTopics(
        data.scatter.filter((segment) => segment.category_id === category.id),
      );
      const topicNodes = topics.map(([topic, totals]) => ({
        level: "topic",
        id: topic,
        treeId: `topic:${category.id}:${topic}`,
        parent: `category:${category.id}`,
        label: topicDisplayName(topic),
        value: metric === "talks" ? totals.episodes.size : totals.hours,
        displayValue: metric === "talks" ? totals.episodes.size : totals.hours,
        talks: totals.episodes.size,
        hours: totals.hours,
        color: category.color,
        categoryId: category.id,
      }));
      return [{
        level: "category",
        id: category.id,
        treeId: `category:${category.id}`,
        parent: "",
        label: category.name,
        value: topicNodes.reduce((sum, topic) => sum + topic.value, 0),
        displayValue: category[metric],
        talks: category.talks,
        hours: category.hours,
        color: category.color,
        categoryId: category.id,
      }, ...topicNodes];
    });
  }

  const category = data.categories.find((item) => item.id === selection.categoryId);
  const segments = data.scatter.filter((item) => item.category_id === selection.categoryId);
  if (!selection.topic) {
    return aggregateTopics(segments)
      .map(([topic, totals]) => ({
        level: "topic",
        id: topic,
        label: topicDisplayName(topic),
        value: metric === "talks" ? totals.episodes.size : totals.hours,
        displayValue: metric === "talks" ? totals.episodes.size : totals.hours,
        talks: totals.episodes.size,
        hours: totals.hours,
        color: category?.color,
      }));
  }

  return segments
    .filter((segment) => segment.topic === selection.topic)
    .map((segment) => {
      const hours = (Number(segment.end) - Number(segment.start)) / 3600;
      return {
        level: "passage",
        id: segment.id,
        label: `${segment.title} · ${passageTime(segment.start)}`,
        value: metric === "talks" ? 1 : hours,
        displayValue: metric === "talks" ? 1 : hours,
        talks: 1,
        hours,
        color: category?.color,
        episode: segment.episode,
        segment: segment.id,
      };
    })
    .sort((a, b) => a.label.localeCompare(b.label, "et"));
}

export function heatmapText(values) {
  return values.map((row) => row.map((value) => value ? String(value) : ""));
}

export function createLazyViewController(initialView, render) {
  let currentView = initialView;
  let mounted = false;
  let rendering = false;
  let pending = false;
  let activeRender = Promise.resolve();

  function renderLatest() {
    if (rendering) {
      pending = true;
      return activeRender;
    }
    rendering = true;
    activeRender = (async () => {
      try {
        do {
          pending = false;
          await render(currentView);
        } while (pending);
      } finally {
        rendering = false;
      }
    })();
    return activeRender;
  }

  return {
    setView(view) {
      currentView = view;
      return mounted ? renderLatest() : Promise.resolve();
    },
    mount() {
      if (mounted) return activeRender;
      mounted = true;
      return renderLatest();
    },
  };
}
