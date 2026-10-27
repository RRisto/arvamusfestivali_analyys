(function () {
  "use strict";

  let comparison;
  let overview;
  let metric = "time_share";
  let differenceView = "share";
  let subtopicView = "share";
  let selectedCategory = "";

  const labels = {
    talk_coverage: "arutelude osakaal",
    segment_share: "teemalõikude osakaal",
    time_share: "salvestusaja osakaal",
  };

  const absoluteLabels = {
    talk_coverage: { title: "arutelude arv", unit: "arutelu", digits: 0 },
    segment_share: { title: "teemalõikude arv", unit: "lõiku", digits: 0 },
    time_share: { title: "salvestusaeg", unit: "min", digits: 1 },
  };

  function categoryRows() {
    const counts = new Map();
    const totals = { 2025: 0, 2026: 0 };
    for (const segment of overview.scatter) {
      const year = String(segment.year);
      if (year !== "2025" && year !== "2026") continue;
      const key = `${segment.category_id}\u0000${year}`;
      counts.set(key, (counts.get(key) || 0) + 1);
      totals[year] += 1;
    }
    return comparison.categories.map((category) => {
      const share2025 = totals[2025] ? (counts.get(`${category.id}\u00002025`) || 0) / totals[2025] * 100 : 0;
      const share2026 = totals[2026] ? (counts.get(`${category.id}\u00002026`) || 0) / totals[2026] * 100 : 0;
      return {
        ...category,
        segment_count_2025: counts.get(`${category.id}\u00002025`) || 0,
        segment_count_2026: counts.get(`${category.id}\u00002026`) || 0,
        segment_share_2025: share2025,
        segment_share_2026: share2026,
        segment_share_delta_pp: share2026 - share2025,
      };
    });
  }

  function deltaKey() {
    return `${metric}_delta_pp`;
  }

  function absoluteValue(item, year) {
    if (metric === "segment_share") return item[`segment_count_${year}`];
    return item[`${metric === "time_share" ? "minutes" : "talks"}_${year}`];
  }

  function absoluteDelta(item) {
    return absoluteValue(item, 2026) - absoluteValue(item, 2025);
  }

  function formatAbsoluteTemplate(valueIndex) {
    return `%{customdata[${valueIndex}]:${absoluteLabels[metric].digits === 0 ? ",.0f" : ",.1f"}}`;
  }

  function transcriptUrl(category, year) {
    const params = new URLSearchParams({ category: category.id, year: String(year) });
    return `transcripts.html?${params}`;
  }

  function renderSummary() {
    const sorted = [...categoryRows()].sort((a, b) => b[deltaKey()] - a[deltaKey()]);
    const rise = sorted[0];
    const fall = sorted[sorted.length - 1];
    document.getElementById("largest-rise").textContent = rise.name;
    document.getElementById("largest-rise-value").textContent = `+${siteCommon.formatNumber(rise[deltaKey()], 2)} pp`;
    document.getElementById("largest-fall").textContent = fall.name;
    document.getElementById("largest-fall-value").textContent = `${siteCommon.formatNumber(fall[deltaKey()], 2)} pp`;
  }

  function renderDistribution() {
    const categories = [...categoryRows()].sort((a, b) => b[`${metric}_2026`] - a[`${metric}_2026`]);
    const names = categories.map((item) => item.name);
    const common = { type: "bar", orientation: "h", customdata: categories.map((item) => item.id), hovertemplate: `%{y}<br>%{x:.1f}% ${labels[metric]}<extra></extra>` };
    Plotly.react("distribution-chart", [
      { ...common, name: "2025", x: categories.map((item) => item[`${metric}_2025`]), y: names, marker: { color: "#b8c9c3" } },
      { ...common, name: "2026", x: categories.map((item) => item[`${metric}_2026`]), y: names, marker: { color: "#d95e32" } },
    ], {
      ...siteCommon.chartLayout,
      barmode: "group",
      height: Math.max(650, names.length * 31),
      xaxis: { title: `${labels[metric]} (%)`, ticksuffix: "%", gridcolor: "#e7ece9" },
      yaxis: { autorange: "reversed", automargin: true, ticklabelstandoff: 16 },
      legend: { orientation: "h", y: 1.08 },
      margin: { l: 280, r: 30, t: 60, b: 60 },
    }, siteCommon.plotConfig).then((plot) => {
      plot.removeAllListeners("plotly_click");
      plot.on("plotly_click", (event) => {
        const point = event.points[0];
        window.location.href = transcriptUrl(categories[point.pointIndex], point.curveNumber === 0 ? 2025 : 2026);
      });
    });
  }

  function renderDifference() {
    const rows = categoryRows().sort((a, b) => differenceView === "share"
      ? a[deltaKey()] - b[deltaKey()]
      : absoluteValue(a, 2026) - absoluteValue(b, 2026));
    const unit = absoluteLabels[metric].unit;
    const customdata = rows.map((item) => [
      item[`${metric}_2025`], item[`${metric}_2026`],
      absoluteValue(item, 2025), absoluteValue(item, 2026), absoluteDelta(item), item[deltaKey()],
    ]);
    const traces = differenceView === "share"
      ? [{
        type: "bar",
        orientation: "h",
        x: rows.map((item) => item[deltaKey()]),
        y: rows.map((item) => item.name),
        customdata,
        marker: { color: rows.map((item) => item[deltaKey()] >= 0 ? "#d95e32" : "#286454") },
        hovertemplate: `%{y}<br>2025 osakaal: %{customdata[0]:.1f}%<br>2026 osakaal: %{customdata[1]:.1f}%<br>Osakaalu muutus: %{x:+.1f} pp<br>Absoluutmaht: ${formatAbsoluteTemplate(2)} → ${formatAbsoluteTemplate(3)} ${unit} (${formatAbsoluteTemplate(4)} ${unit})<extra></extra>`,
      }]
      : [2025, 2026].map((year, index) => ({
        type: "bar",
        orientation: "h",
        name: String(year),
        x: rows.map((item) => absoluteValue(item, year)),
        y: rows.map((item) => item.name),
        customdata,
        texttemplate: metric === "time_share" ? "%{x:,.1f}" : "%{x:,.0f}",
        textposition: "outside",
        cliponaxis: false,
        marker: { color: index === 0 ? "#b8c9c3" : "#d95e32" },
        hovertemplate: `%{y}<br>2025. aasta: ${formatAbsoluteTemplate(2)} ${unit}<br>2026. aasta: ${formatAbsoluteTemplate(3)} ${unit}<br>Absoluutmahu muutus: %{customdata[4]:+.${absoluteLabels[metric].digits}f} ${unit}<br>Osakaalu muutus: %{customdata[5]:+.1f} pp<extra></extra>`,
      }));

    Plotly.react("difference-chart", traces, {
      ...siteCommon.chartLayout,
      barmode: differenceView === "absolute" ? "group" : "relative",
      height: Math.max(650, rows.length * 31),
      xaxis: differenceView === "share"
        ? { title: `2026 − 2025, ${labels[metric]} (protsendipunkti)`, ticksuffix: " pp", zeroline: true, zerolinewidth: 2, zerolinecolor: "#17322b", gridcolor: "#e7ece9" }
        : { title: `${absoluteLabels[metric].title} (${unit})`, ticksuffix: ` ${unit}`, gridcolor: "#e7ece9" },
      yaxis: { automargin: true, ticklabelstandoff: 12 },
      showlegend: differenceView === "absolute",
      legend: { orientation: "h", y: 1.08 },
      margin: { l: 220, r: 20, t: differenceView === "absolute" ? 60 : 30, b: 60 },
    }, siteCommon.plotConfig).then((plot) => {
      plot.removeAllListeners("plotly_click");
      plot.on("plotly_click", (event) => {
        const item = rows[event.points[0].pointIndex];
        window.location.href = transcriptUrl(item, differenceView === "absolute" ? Number(event.points[0].data.name) : 2026);
      });
    });
  }

  function renderSubtopicDifference() {
    const category = comparison.categories.find((item) => item.id === selectedCategory);
    if (!category || !overview) return;

    const segments = overview.scatter.filter((row) => row.category_id === category.id);
    const topicRows = new Map();
    const categoryTotals = {
      2025: { seconds: 0, talks: new Set() },
      2026: { seconds: 0, talks: new Set() },
    };
    categoryTotals[2025].segments = 0;
    categoryTotals[2026].segments = 0;
    for (const segment of segments) {
      const year = String(segment.year);
      if (year !== "2025" && year !== "2026") continue;
      const seconds = Number(segment.end) - Number(segment.start);
      categoryTotals[year].seconds += seconds;
      categoryTotals[year].talks.add(segment.episode);
      categoryTotals[year].segments += 1;

      const name = segment.topic || "Määramata";
      if (!topicRows.has(name)) {
        topicRows.set(name, {
          name,
          seconds: { 2025: 0, 2026: 0 },
          talks: { 2025: new Set(), 2026: new Set() },
          segments: { 2025: 0, 2026: 0 },
        });
      }
      const row = topicRows.get(name);
      row.seconds[year] += seconds;
      row.talks[year].add(segment.episode);
      row.segments[year] += 1;
    }

    const rows = [...topicRows.values()].map((row) => {
      const shareForYear = (year) => {
        if (metric === "time_share") {
          return categoryTotals[year].seconds ? row.seconds[year] / categoryTotals[year].seconds * 100 : 0;
        }
        if (metric === "segment_share") {
          return categoryTotals[year].segments ? row.segments[year] / categoryTotals[year].segments * 100 : 0;
        }
        return categoryTotals[year].talks.size ? row.talks[year].size / categoryTotals[year].talks.size * 100 : 0;
      };
      const share2025 = shareForYear(2025);
      const share2026 = shareForYear(2026);
      const absoluteForYear = (year) => metric === "time_share"
        ? row.seconds[year] / 60
        : metric === "segment_share" ? row.segments[year] : row.talks[year].size;
      const absolute2025 = absoluteForYear(2025);
      const absolute2026 = absoluteForYear(2026);
      return { ...row, share2025, share2026, delta: share2026 - share2025,
        absolute2025, absolute2026, absoluteDelta: absolute2026 - absolute2025 };
    });

    const unit = absoluteLabels[metric].unit;
    const makeSubtopicCustomdata = (row) => [row.share2025, row.share2026, row.absolute2025,
      row.absolute2026, row.absoluteDelta, row.delta];
    const chartRows = [...rows].sort((a, b) => subtopicView === "share"
      ? a.delta - b.delta
      : a.absolute2026 - b.absolute2026);
    const customdata = chartRows.map(makeSubtopicCustomdata);
    const traces = subtopicView === "share"
      ? [{
        type: "bar",
        orientation: "h",
        x: chartRows.map((row) => row.delta),
        y: chartRows.map((row) => row.name),
        customdata,
        marker: { color: chartRows.map((row) => row.delta >= 0 ? "#d95e32" : "#286454") },
        hovertemplate: `%{y}<br>2025 osakaal: %{customdata[0]:.1f}%<br>2026 osakaal: %{customdata[1]:.1f}%<br>Osakaalu muutus: %{x:+.1f} pp<br>Absoluutmaht: %{customdata[2]:.${absoluteLabels[metric].digits}f} → %{customdata[3]:.${absoluteLabels[metric].digits}f} ${unit} (%{customdata[4]:+.${absoluteLabels[metric].digits}f} ${unit})<extra></extra>`,
      }]
      : [2025, 2026].map((year, index) => ({
        type: "bar",
        orientation: "h",
        name: String(year),
        x: chartRows.map((row) => year === 2025 ? row.absolute2025 : row.absolute2026),
        y: chartRows.map((row) => row.name),
        customdata,
        texttemplate: metric === "time_share" ? "%{x:,.1f}" : "%{x:,.0f}",
        textposition: "outside",
        cliponaxis: false,
        marker: { color: index === 0 ? "#b8c9c3" : "#d95e32" },
        hovertemplate: `%{y}<br>2025. aasta: %{customdata[2]:.${absoluteLabels[metric].digits}f} ${unit}<br>2026. aasta: %{customdata[3]:.${absoluteLabels[metric].digits}f} ${unit}<br>Absoluutmahu muutus: %{customdata[4]:+.${absoluteLabels[metric].digits}f} ${unit}<br>Osakaalu muutus: %{customdata[5]:+.1f} pp<extra></extra>`,
      }));

    Plotly.react("subtopic-difference-chart", traces, {
      ...siteCommon.chartLayout,
      barmode: subtopicView === "absolute" ? "group" : "relative",
      height: Math.max(320, chartRows.length * 28 + 70),
      xaxis: subtopicView === "absolute"
        ? { title: `${absoluteLabels[metric].title} (${unit})`, ticksuffix: ` ${unit}`, gridcolor: "#e7ece9" }
        : { title: `2026 − 2025, ${labels[metric]} (protsendipunkti)`, ticksuffix: " pp", zeroline: true, zerolinewidth: 2, zerolinecolor: "#17322b", gridcolor: "#e7ece9" },
      yaxis: { automargin: true, ticklabelstandoff: 8 },
      showlegend: subtopicView === "absolute",
      legend: { orientation: "h", y: 1.08 },
      margin: { l: 200, r: 20, t: subtopicView === "absolute" ? 50 : 16, b: 42 },
    }, siteCommon.plotConfig);
  }

  function populateSubtopicCategory() {
    const select = document.getElementById("subtopic-category");
    const categories = [...comparison.categories].sort((a, b) => a.name.localeCompare(b.name, "et"));
    select.replaceChildren(...categories.map((category) => {
      const option = document.createElement("option");
      option.value = category.id;
      option.textContent = category.name;
      return option;
    }));
    selectedCategory = select.value;
    select.addEventListener("change", () => {
      selectedCategory = select.value;
      renderSubtopicDifference();
    });
  }

  function render() {
    document.getElementById("talk-coverage-help").hidden = metric !== "talk_coverage";
    renderSummary();
    renderDistribution();
    renderDifference();
    renderSubtopicDifference();
  }

  async function init() {
    try {
      comparison = await siteCommon.fetchJson("data/comparison.json");
      overview = await siteCommon.fetchJson("data/overview.json");
      document.getElementById("talks-2025").textContent = siteCommon.formatNumber(comparison.cohorts["2025"].talks);
      document.getElementById("talks-2026").textContent = siteCommon.formatNumber(comparison.cohorts["2026"].talks);
      populateSubtopicCategory();
      document.querySelectorAll("[data-compare-metric]").forEach((button) => button.addEventListener("click", () => {
        metric = button.dataset.compareMetric;
        document.querySelectorAll("[data-compare-metric]").forEach((candidate) => {
          const active = candidate === button;
          candidate.classList.toggle("active", active);
          candidate.setAttribute("aria-pressed", String(active));
        });
        render();
      }));
      document.querySelectorAll("[data-chart-view]").forEach((button) => button.addEventListener("click", () => {
        const group = button.dataset.chartView;
        const view = button.dataset.view;
        if (group === "difference") differenceView = view;
        else subtopicView = view;
        document.querySelectorAll("[data-chart-view]").forEach((candidate) => {
          if (candidate.dataset.chartView !== group) return;
          const active = candidate === button;
          candidate.classList.toggle("active", active);
          candidate.setAttribute("aria-pressed", String(active));
        });
        if (group === "difference") renderDifference();
        else renderSubtopicDifference();
      }));
      render();
    } catch (error) {
      siteCommon.showError(error);
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();
