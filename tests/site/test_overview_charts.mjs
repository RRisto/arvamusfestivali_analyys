import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCategoryBubbles,
  buildCooccurrenceLayout,
  buildSubtopicCooccurrenceMatrix,
  buildTopicCooccurrenceMatrix,
  buildTopicBubbles,
  buildTopicScatterLayout,
  buildTopicScatterTraces,
  buildTreemapLevel,
  createLazyViewController,
  heatmapText,
} from "../../site/assets/overview-core.mjs";

test("subtopic heatmap layout allocates an explicit square canvas for every row", () => {
  const layout = buildCooccurrenceLayout({ font: { family: "sans-serif" } }, true, 236);

  assert.equal(layout.autosize, false);
  assert.equal(layout.width, 4668);
  assert.equal(layout.height, 4668);
  assert.equal(layout.xaxis.tickangle, -90);
  assert.deepEqual(buildCooccurrenceLayout({}, false, 20), {
    autosize: true,
    xaxis: {
      tickangle: -45,
      tickfont: { size: 10 },
      side: "bottom",
      automargin: true,
      fixedrange: true,
    },
    yaxis: {
      autorange: "reversed",
      automargin: true,
      fixedrange: true,
      tickfont: { size: 10 },
    },
    margin: { l: 285, r: 30, t: 24, b: 280 },
  });
});

test("subtopic co-occurrence matrix omits subtopics with fewer than three co-occurrences", () => {
  const categories = [
    { id: "education", name: "Haridus" },
    { id: "ai", name: "Tehisintellekt" },
    { id: "unassigned", name: "Määramata" },
  ];
  const scatter = [
    { episode: "e1", category_id: "education", topic: "Digioskused" },
    { episode: "e1", category_id: "education", topic: "Digioskused" },
    { episode: "e1", category_id: "ai", topic: "Digioskused" },
    { episode: "e2", category_id: "education", topic: "Digioskused" },
    { episode: "e2", category_id: "education", topic: "Õpetajad" },
    { episode: "e2", category_id: "ai", topic: "Digioskused" },
    { episode: "e3", category_id: "ai", topic: "Robotid" },
    { episode: "e4", category_id: "unassigned", topic: "Muu" },
    { episode: "e5", category_id: "education", topic: "Määramata" },
  ];

  assert.deepEqual(buildSubtopicCooccurrenceMatrix(scatter, categories), {
    ids: [
      "education\u0000Digioskused",
      "ai\u0000Digioskused",
    ],
    labels: [
      "Haridus · Digioskused",
      "Tehisintellekt · Digioskused",
    ],
    z: [
      [null, 2],
      [2, null],
    ],
  });
});

test("topic co-occurrence matrix counts each high-level pair once per talk", () => {
  const categories = [
    { id: "education", name: "Haridus" },
    { id: "ai", name: "Tehisintellekt" },
    { id: "health", name: "Tervis" },
    { id: "unassigned", name: "Määramata" },
  ];
  const scatter = [
    { episode: "e1", category_id: "education" },
    { episode: "e1", category_id: "education" },
    { episode: "e1", category_id: "ai" },
    { episode: "e2", category_id: "education" },
    { episode: "e2", category_id: "ai" },
    { episode: "e2", category_id: "health" },
    { episode: "e3", category_id: "health" },
    { episode: "e3", category_id: "unassigned" },
  ];

  assert.deepEqual(buildTopicCooccurrenceMatrix(scatter, categories), {
    categoryIds: ["education", "ai", "health"],
    names: ["Haridus", "Tehisintellekt", "Tervis"],
    z: [
      [null, 2, 1],
      [2, null, 1],
      [1, 1, null],
    ],
  });
});

test("topic bubbles aggregate subtopics and inherit their main-topic color", () => {
  const categories = [
    { id: "education", name: "Haridus", color: "gold" },
    { id: "health", name: "Tervis", color: "green" },
  ];
  const scatter = [
    { category_id: "education", topic: "Digioskused", episode: "e1", x: 0, y: 2, start: 0, end: 60 },
    { category_id: "education", topic: "Digioskused", episode: "e1", x: 2, y: 4, start: 60, end: 180 },
    { category_id: "education", topic: "Digioskused", episode: "e2", x: 4, y: 6, start: 0, end: 60 },
    { category_id: "health", topic: "Digioskused", episode: "e3", x: -2, y: 8, start: 0, end: 180 },
  ];

  assert.deepEqual(buildTopicBubbles(scatter, categories), [
    {
      id: "education\u0000Digioskused",
      name: "Digioskused",
      categoryId: "education",
      categoryName: "Haridus",
      x: 2,
      y: 4,
      talks: 2,
      segments: 3,
      hours: 1 / 15,
      color: "gold",
    },
    {
      id: "health\u0000Digioskused",
      name: "Digioskused",
      categoryId: "health",
      categoryName: "Tervis",
      x: -2,
      y: 8,
      talks: 1,
      segments: 1,
      hours: 1 / 20,
      color: "green",
    },
  ]);
});

test("topic scatter traces show bubbles without permanent text labels", () => {
  const categories = [{ id: "education", name: "Haridus", color: "gold" }];
  const bubbles = [{
    id: "education\u0000Digioskused",
    name: "Digioskused",
    categoryId: "education",
    categoryName: "Haridus",
    x: 2,
    y: 4,
    talks: 2,
    segments: 3,
    hours: 0.1,
    color: "gold",
  }];

  assert.deepEqual(buildTopicScatterTraces(bubbles, categories), [{
    type: "scatter",
    mode: "markers",
    name: "Haridus",
    x: [2],
    y: [4],
    customdata: [["education", "Haridus", "Digioskused", 2, 3, 0.1]],
    marker: {
      size: [2],
      sizemode: "area",
      sizeref: 4 / (58 ** 2),
      sizemin: 8,
      color: "gold",
      opacity: 0.8,
      line: { color: "#ffffff", width: 1.25 },
    },
    hovertemplate: "<b>%{customdata[2]}</b><br>%{customdata[1]}<br>%{customdata[3]} arutelu<br>%{customdata[4]} teemalõiku<br>%{customdata[5]:.1f} tundi<extra></extra>",
  }]);
});

test("topic scatter keeps variable talk-count bubbles but uses constant legend symbols", () => {
  const bubbles = [
    { categoryId: "education", categoryName: "Haridus", name: "Digioskused", x: 1, y: 2, talks: 2, segments: 3, hours: 0.1 },
    { categoryId: "education", categoryName: "Haridus", name: "Õpetajad", x: 3, y: 4, talks: 5, segments: 6, hours: 0.2 },
  ];
  const categories = [{ id: "education", name: "Haridus", color: "gold" }];

  const traces = buildTopicScatterTraces(bubbles, categories);
  const layout = buildTopicScatterLayout({ font: { family: "sans-serif" } });

  assert.deepEqual(traces[0].marker.size, [2, 5]);
  assert.equal(layout.legend.itemsizing, "constant");
});

const hierarchyData = {
  categories: [
    { id: "education", name: "Haridus", talks: 2, segments: 3, hours: 0.1, color: "gold" },
  ],
  scatter: [
    { id: "s1", episode: "e1", title: "Õpetajad", start: 60, end: 120, topic: "Õpetaja töö", category_id: "education" },
    { id: "s2", episode: "e1", title: "Õpetajad", start: 180, end: 300, topic: "Õpetaja töö", category_id: "education" },
    { id: "s3", episode: "e2", title: "Digikool", start: 30, end: 90, topic: "Digioskused", category_id: "education" },
  ],
};

test("treemap overview nests visible subtopics inside each high-level category", () => {
  assert.deepEqual(buildTreemapLevel(hierarchyData, { categoryId: "" }, "talks"), [
    {
      level: "category",
      id: "education",
      treeId: "category:education",
      parent: "",
      label: "Haridus",
      value: 2,
      displayValue: 2,
      talks: 2,
      hours: 0.1,
      color: "gold",
      categoryId: "education",
    },
    {
      level: "topic",
      id: "Digioskused",
      treeId: "topic:education:Digioskused",
      parent: "category:education",
      label: "Digioskused",
      value: 1,
      displayValue: 1,
      talks: 1,
      hours: 1 / 60,
      color: "gold",
      categoryId: "education",
    },
    {
      level: "topic",
      id: "Õpetaja töö",
      treeId: "topic:education:Õpetaja töö",
      parent: "category:education",
      label: "Õpetaja töö",
      value: 1,
      displayValue: 1,
      talks: 1,
      hours: 3 / 60,
      color: "gold",
      categoryId: "education",
    },
  ]);
});

test("treemap category displays its unique talk count when subtopics overlap", () => {
  const data = {
    categories: [
      { id: "education", name: "Haridus", talks: 1, segments: 2, hours: 2 / 60, color: "gold" },
    ],
    scatter: [
      { id: "s1", episode: "e1", title: "Kool", start: 0, end: 60, topic: "Õpetajad", category_id: "education" },
      { id: "s2", episode: "e1", title: "Kool", start: 60, end: 120, topic: "Õpilased", category_id: "education" },
    ],
  };

  const [category] = buildTreemapLevel(data, { categoryId: "" }, "talks");
  assert.equal(category.value, 2);
  assert.equal(category.displayValue, 1);
});

test("treemap drills from a high-level category into aggregated subtopics", () => {
  assert.deepEqual(buildTreemapLevel(hierarchyData, { categoryId: "education" }, "talks"), [
    { level: "topic", id: "Digioskused", label: "Digioskused", value: 1, displayValue: 1, talks: 1, hours: 1 / 60, color: "gold" },
    { level: "topic", id: "Õpetaja töö", label: "Õpetaja töö", value: 1, displayValue: 1, talks: 1, hours: 3 / 60, color: "gold" },
  ]);
});

test("treemap subtopic opens concrete timestamped transcript passages", () => {
  assert.deepEqual(
    buildTreemapLevel(
      hierarchyData,
      { categoryId: "education", topic: "Õpetaja töö" },
      "hours",
    ),
    [
      { level: "passage", id: "s1", label: "Õpetajad · 1:00", value: 1 / 60, displayValue: 1 / 60, talks: 1, hours: 1 / 60, color: "gold", episode: "e1", segment: "s1" },
      { level: "passage", id: "s2", label: "Õpetajad · 3:00", value: 2 / 60, displayValue: 2 / 60, talks: 1, hours: 2 / 60, color: "gold", episode: "e1", segment: "s2" },
    ],
  );
});

test("category bubbles use segment centroids and category totals", () => {
  const categories = [
    { id: "education", name: "Haridus", talks: 8, segments: 3, hours: 2.5, color: "gold" },
    { id: "nature", name: "Loodus", talks: 4, segments: 1, hours: 1, color: "green" },
  ];
  const scatter = [
    { category_id: "education", x: 0, y: 2 },
    { category_id: "education", x: 2, y: 4 },
    { category_id: "nature", x: -3, y: 5 },
  ];

  assert.deepEqual(buildCategoryBubbles(scatter, categories), [
    { id: "education", name: "Haridus", x: 1, y: 3, talks: 8, segments: 3, hours: 2.5, color: "gold" },
    { id: "nature", name: "Loodus", x: -3, y: 5, talks: 4, segments: 1, hours: 1, color: "green" },
  ]);
});

test("heatmap labels show counts but leave zero cells quiet", () => {
  assert.deepEqual(heatmapText([[0, 2], [11, 0]]), [["", "2"], ["11", ""]]);
});

test("view changes before lazy mount render only the latest view once", async () => {
  const rendered = [];
  const controller = createLazyViewController("categories", (view) => rendered.push(view));

  await controller.setView("segments");
  assert.deepEqual(rendered, []);

  await controller.mount();
  assert.deepEqual(rendered, ["segments"]);

  await controller.setView("categories");
  assert.deepEqual(rendered, ["segments", "categories"]);
});

test("view changes during rendering wait and then render only the latest view", async () => {
  const rendered = [];
  let finishFirst;
  const controller = createLazyViewController("categories", (view) => {
    rendered.push(view);
    if (view === "categories") {
      return new Promise((resolve) => { finishFirst = resolve; });
    }
    return undefined;
  });

  const mounting = controller.mount();
  controller.setView("segments");
  assert.deepEqual(rendered, ["categories"]);

  finishFirst();
  await mounting;
  assert.deepEqual(rendered, ["categories", "segments"]);
});
