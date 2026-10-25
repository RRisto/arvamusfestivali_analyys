(function () {
  "use strict";

  let comparison;
  let overview;
  let metric = "time_share";
  let selectedCategory = "";

  const labels = {
    talk_coverage: "arutelude osakaal",
    segment_share: "teemalõikude osakaal",
    time_share: "salvestusaja osakaal",
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
        segment_share_2025: share2025,
        segment_share_2026: share2026,
        segment_share_delta_pp: share2026 - share2025,
      };
    });
  }

  function deltaKey() {
    return `${metric}_delta_pp`;
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
    document.getElementById("largest-rise-value").textContent = `+${siteCommon.formatNumber(rise[deltaKey()], 1)} pp`;
    document.getElementById("largest-fall").textContent = fall.name;
    document.getElementById("largest-fall-value").textContent = `${siteCommon.formatNumber(fall[deltaKey()], 1)} pp`;
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
    const categories = [...categoryRows()].sort((a, b) => a[deltaKey()] - b[deltaKey()]);
    const values = categories.map((item) => item[deltaKey()]);
    Plotly.react("difference-chart", [{
      type: "bar",
      orientation: "h",
      x: values,
      y: categories.map((item) => item.name),
      customdata: categories.map((item) => item.id),
      marker: { color: values.map((value) => value >= 0 ? "#d95e32" : "#286454") },
      hovertemplate: "%{y}<br>%{x:+.1f} pp<extra></extra>",
    }], {
      ...siteCommon.chartLayout,
      height: Math.max(650, categories.length * 31),
      xaxis: { title: `2026 − 2025, ${labels[metric]} (protsendipunkti)`, ticksuffix: " pp", zeroline: true, zerolinewidth: 2, zerolinecolor: "#17322b", gridcolor: "#e7ece9" },
      yaxis: { automargin: true, ticklabelstandoff: 16 },
      margin: { l: 280, r: 30, t: 35, b: 60 },
    }, siteCommon.plotConfig).then((plot) => {
      plot.removeAllListeners("plotly_click");
      plot.on("plotly_click", (event) => {
        const item = categories[event.points[0].pointIndex];
        window.location.href = transcriptUrl(item, 2026);
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
      return { ...row, share2025, share2026, delta: share2026 - share2025 };
    }).sort((a, b) => a.delta - b.delta);

    Plotly.react("subtopic-difference-chart", [{
      type: "bar",
      orientation: "h",
      x: rows.map((row) => row.delta),
      y: rows.map((row) => row.name),
      customdata: rows.map((row) => [row.share2025, row.share2026]),
      marker: { color: rows.map((row) => row.delta >= 0 ? "#d95e32" : "#286454") },
      hovertemplate: "%{y}<br>2025: %{customdata[0]:.2f}%<br>2026: %{customdata[1]:.2f}%<br>Muutus: %{x:+.2f} pp<extra></extra>",
    }], {
      ...siteCommon.chartLayout,
      height: Math.max(420, rows.length * 28),
      xaxis: { title: `2026 − 2025, ${labels[metric]} (protsendipunkti)`, ticksuffix: " pp", zeroline: true, zerolinewidth: 2, zerolinecolor: "#17322b", gridcolor: "#e7ece9" },
      yaxis: { automargin: true, ticklabelstandoff: 16 },
      margin: { l: 280, r: 30, t: 35, b: 60 },
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
      render();
    } catch (error) {
      siteCommon.showError(error);
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();
