import {
  buildCooccurrenceLayout,
  buildSubtopicCooccurrenceMatrix,
  buildTopicBubbles,
  buildTopicCooccurrenceMatrix,
  buildTopicScatterLayout,
  buildTopicScatterTraces,
  buildTreemapLevel,
  createLazyViewController,
} from "./overview-core.mjs?v=treemap-segment-metric1";

(async function () {
  "use strict";

  const common = window.siteCommon;
  try {
    const data = await common.fetchJson("data/overview.json");
    const yearFilter = document.getElementById("overview-year");
    let viewData = buildYearView(data, yearFilter.value);
    function updateMetrics() {
      document.getElementById("metric-talks").textContent = common.formatNumber(
        new Set(viewData.scatter.map((item) => item.episode)).size,
      );
      document.getElementById("metric-segments").textContent = common.formatNumber(viewData.scatter.length);
      document.getElementById("metric-categories").textContent = common.formatNumber(
        viewData.categories.filter((category) => category.segments > 0).length,
      );
      document.getElementById("metric-subtopics").textContent = common.formatNumber(
        new Set(viewData.scatter.map((item) => item.topic).filter(Boolean)).size,
      );
    }
    updateMetrics();

    let treemapMetric = "talks";
    let treemapReady = false;
    const treemapSelection = { categoryId: "", topic: "" };
    const treemap = document.getElementById("treemap-chart");
    const treemapBreadcrumb = document.getElementById("treemap-breadcrumb");

    function renderTreemapBreadcrumb() {
      treemapBreadcrumb.replaceChildren();
      const category = viewData.categories.find((item) => item.id === treemapSelection.categoryId);
      const hasSelection = Boolean(treemapSelection.categoryId || treemapSelection.topic);
      const crumbs = hasSelection
        ? [{ label: "Kõik põhiteemad", categoryId: "", topic: "" }]
        : [];
      if (category) crumbs.push({ label: category.name, categoryId: category.id, topic: "" });
      if (treemapSelection.topic) {
        const label = treemapSelection.topic === "Unassigned" ? "Määramata" : treemapSelection.topic;
        crumbs.push({ label, categoryId: category.id, topic: treemapSelection.topic });
      }
      crumbs.forEach((crumb, index) => {
        if (index) {
          const separator = document.createElement("span");
          separator.className = "breadcrumb-separator";
          separator.textContent = "›";
          separator.setAttribute("aria-hidden", "true");
          treemapBreadcrumb.append(separator);
        }
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = crumb.label;
        if (index === crumbs.length - 1) button.setAttribute("aria-current", "page");
        button.addEventListener("click", () => {
          if (index === crumbs.length - 1) return;
          treemapSelection.categoryId = crumb.categoryId;
          treemapSelection.topic = crumb.topic;
          renderTreemap();
        });
        treemapBreadcrumb.append(button);
      });
    }

    function renderTreemap() {
      treemap.classList.remove("chart-loading");
      if (!treemap.classList.contains("js-plotly-plot")) treemap.textContent = "";
      const isHours = treemapMetric === "hours";
      const unit = isHours ? "tundi" : treemapMetric === "segments" ? "teemalõiku" : "arutelu";
      const nodes = buildTreemapLevel(viewData, treemapSelection, treemapMetric);
      const totalValue = buildTreemapLevel(viewData, { categoryId: "", topic: "" }, treemapMetric)
        .filter((row) => row.level === "category")
        .reduce((sum, row) => sum + row.value, 0);
      renderTreemapBreadcrumb();
      document.getElementById("treemap-help").textContent = treemapSelection.topic
        ? "Klõpsa lõigul, et avada täpne koht transkriptis ja ajastatud helilink."
          : treemapSelection.categoryId
            ? "Klõpsa alamteemal, et näha selle konkreetseid kohti transkriptides."
          : "Alamteemad on põhiteemade sees kohe nähtavad. Klõpsa alamteemal, et näha konkreetseid kohti transkriptides.";
      Plotly.react(treemap, [{
        type: "treemap",
        labels: nodes.map((row) => row.label),
        parents: nodes.map((row) => row.parent || ""),
        values: nodes.map((row) => row.value),
        ids: nodes.map((row) => row.treeId || `${row.level}:${row.id}`),
        customdata: nodes.map((row) => [row.level, row.id, row.talks, row.hours, row.episode, row.segment, row.categoryId, row.displayValue, row.segments, totalValue ? row.value / totalValue * 100 : 0]),
        branchvalues: "total",
        texttemplate: `<b>%{label}</b><br>%{customdata[7]:${isHours ? ".1f" : ".0f"}} ${unit}`,
        hovertemplate: "<b>%{label}</b><br>%{customdata[2]} arutelu<br>%{customdata[8]} teemalõiku<br>%{customdata[3]:.1f} tundi<br>Osakaal kõigi teemade kogumahust: %{customdata[9]:.1f}%<extra></extra>",
        marker: { colors: nodes.map((row) => row.color), line: { color: "#ffffff", width: 2 } },
        tiling: { pad: 3 },
      }], { ...common.chartLayout, margin: { l: 8, r: 8, t: 8, b: 8 } }, common.plotConfig).then((plot) => {
        plot.removeAllListeners("plotly_click");
        plot.on("plotly_click", (event) => {
          const [level, id, , , episode, segment, categoryId] = event.points[0]?.customdata || [];
          if (level === "category") {
            treemapSelection.categoryId = id;
            treemapSelection.topic = "";
            renderTreemap();
          } else if (level === "topic") {
            treemapSelection.categoryId = categoryId || treemapSelection.categoryId;
            treemapSelection.topic = id;
            renderTreemap();
          } else if (level === "passage" && episode && segment) {
            location.href = `transcripts.html?episode=${encodeURIComponent(episode)}&segment=${encodeURIComponent(segment)}`;
          }
        });
      });
    }
    document.querySelectorAll("[data-metric]").forEach((button) => {
      button.addEventListener("click", () => {
        treemapMetric = button.dataset.metric;
        document.querySelectorAll("[data-metric]").forEach((candidate) => {
          const selected = candidate === button;
          candidate.classList.toggle("active", selected);
          candidate.setAttribute("aria-pressed", String(selected));
        });
        if (treemapReady) renderTreemap();
      });
    });
    common.observeOnce(treemap, () => {
      treemapReady = true;
      renderTreemap();
    });

    const scatter = document.getElementById("scatter-chart");
    let activeScatterView = "topics";
    let scatterReady = false;
    function renderScatter(scatterView) {
      scatter.classList.remove("chart-loading");
      if (!scatter.classList.contains("js-plotly-plot")) scatter.textContent = "";
      if (scatterView === "topics") {
        const bubbles = buildTopicBubbles(viewData.scatter, viewData.categories);
        document.getElementById("scatter-description").textContent = "Iga mull on üks alamteema. Asukoht näitab sarnasust teiste teemadega, suurus arutelude arvu ja värv põhiteemat.";
        document.getElementById("scatter-help").textContent = "Mullil klõpsamine avab vastavad lõigud transkriptiotsingus.";
        const traces = buildTopicScatterTraces(bubbles, viewData.categories);
        return Plotly.react(scatter, traces, buildTopicScatterLayout(common.chartLayout), common.plotConfig).then((plot) => {
          plot.removeAllListeners("plotly_click");
          plot.on("plotly_click", (event) => {
            const [category, , topic] = event.points[0]?.customdata || [];
            if (category && topic) {
              const params = new URLSearchParams({ category, q: topic });
              if (yearFilter.value !== "all") params.set("year", yearFilter.value);
              location.href = `transcripts.html?${params}`;
            }
          });
        });
      }

      document.getElementById("scatter-description").textContent = "Iga punkt on üks semantiline lõik. Lähedus näitab tekstilist sarnasust, värv põhiteemat. Kahemõõtmeline kaugus on ligikaudne.";
      document.getElementById("scatter-help").textContent = "Hõljuta lisainfo nägemiseks. Punkt avab valitud lõigu ja ajastatud helilingi.";
      const grouped = new Map();
      viewData.scatter.forEach((row) => {
        if (!grouped.has(row.category_id)) grouped.set(row.category_id, []);
        grouped.get(row.category_id).push(row);
      });
      const traces = [...grouped.entries()].map(([categoryId, rows]) => {
        const category = viewData.categories.find((item) => item.id === categoryId);
        return {
          type: "scattergl",
          mode: "markers",
          name: category?.name || rows[0].category,
          x: rows.map((row) => row.x),
          y: rows.map((row) => row.y),
          customdata: rows.map((row) => [row.id, row.title, row.year, row.topic, row.organizer, row.start]),
          marker: { size: 7, opacity: .78, color: category?.color || "#6b7280" },
          hovertemplate: "<b>%{customdata[1]}</b> · %{customdata[2]}<br>%{customdata[3]}<br>%{customdata[4]}<br>Algus %{customdata[5]:.0f} s<extra></extra>",
        };
      });
      return Plotly.react(scatter, traces, {
        ...common.chartLayout,
        hovermode: "closest",
        legend: { orientation: "h", y: -0.12, font: { size: 10 } },
        xaxis: { visible: false },
        yaxis: { visible: false },
        margin: { l: 14, r: 14, t: 16, b: 110 },
      }, common.plotConfig).then((plot) => {
        plot.removeAllListeners("plotly_click");
        plot.on("plotly_click", (event) => {
          const segment = event.points[0]?.customdata?.[0];
          const row = viewData.scatter.find((item) => item.id === segment);
          if (row) location.href = `transcripts.html?episode=${encodeURIComponent(row.episode)}&segment=${encodeURIComponent(row.id)}`;
        });
      });
    }
    const scatterController = createLazyViewController("topics", renderScatter);
    document.querySelectorAll("[data-scatter-view]").forEach((button) => {
      button.addEventListener("click", () => {
        activeScatterView = button.dataset.scatterView;
        scatterController.setView(button.dataset.scatterView);
        document.querySelectorAll("[data-scatter-view]").forEach((candidate) => {
          const selected = candidate === button;
          candidate.classList.toggle("active", selected);
          candidate.setAttribute("aria-pressed", String(selected));
        });
      });
    });
    common.observeOnce(scatter, () => {
      scatterReady = true;
      scatterController.mount();
    });

    const cooccurrenceChart = document.getElementById("cooccurrence-chart");
    const cooccurrenceScroll = document.getElementById("cooccurrence-scroll");
    let cooccurrenceView = "categories";
    let cooccurrenceReady = false;
    function renderCooccurrence() {
      cooccurrenceChart.classList.remove("chart-loading");
      if (!cooccurrenceChart.classList.contains("js-plotly-plot")) cooccurrenceChart.textContent = "";
      const isSubtopics = cooccurrenceView === "subtopics";
      const matrix = isSubtopics
        ? buildSubtopicCooccurrenceMatrix(viewData.scatter, viewData.categories)
        : buildTopicCooccurrenceMatrix(viewData.scatter, viewData.categories);
      const labels = isSubtopics ? matrix.labels : matrix.names;
      const layout = buildCooccurrenceLayout(common.chartLayout, isSubtopics, labels.length);
      cooccurrenceScroll.classList.toggle("subtopics-active", isSubtopics);
      cooccurrenceChart.style.width = isSubtopics ? `${layout.width}px` : "";
      cooccurrenceChart.style.height = isSubtopics ? `${layout.height}px` : "";
      cooccurrenceChart.setAttribute("aria-label", isSubtopics
        ? "Kõigi alamteemade koosesinemise heatmap"
        : "Põhiteemade koosesinemise heatmap");
      document.getElementById("cooccurrence-description").textContent = isSubtopics
        ? `Maatriksis on ${labels.length} alamteemat, millel on teiste alamteemadega kokku vähemalt 3 koosesinemist. Iga paar loetakse ühe arutelu kohta ainult üks kord.`
        : "Iga lahter näitab, mitmes erinevas arutelus kaks teemat koos esinesid. „Määramata” teemaga lõigud on välja jäetud.";
      document.getElementById("cooccurrence-help").textContent = isSubtopics
        ? "Keri maatriksit mõlemas suunas. Hõljuta lahtril alamteemade ja täpse arvu nägemiseks."
        : "Tumedam lahter tähendab rohkem ühiseid arutelusid.";
      const hover = matrix.z.map((row, rowIndex) => row.map((talks, columnIndex) =>
        rowIndex === columnIndex
          ? ""
          : `<b>${labels[rowIndex]}</b><br>+ ${labels[columnIndex]}<br>${talks} arutelu`));
      Plotly.react(cooccurrenceChart, [{
        type: "heatmap",
        x: labels,
        y: labels,
        z: matrix.z,
        customdata: hover,
        xgap: 1,
        ygap: 1,
        colorscale: [[0, "#f3f7f5"], [.25, "#d1e3dc"], [.6, "#8eb7a8"], [1, "#376f5d"]],
        colorbar: { title: "Arutelusid" },
        hoverongaps: false,
        hovertemplate: "%{customdata}<extra></extra>",
      }], layout, common.plotConfig);
    }
    document.querySelectorAll("[data-cooccurrence-view]").forEach((button) => {
      button.addEventListener("click", () => {
        cooccurrenceView = button.dataset.cooccurrenceView;
        document.querySelectorAll("[data-cooccurrence-view]").forEach((candidate) => {
          const selected = candidate === button;
          candidate.classList.toggle("active", selected);
          candidate.setAttribute("aria-pressed", String(selected));
        });
        if (cooccurrenceReady) renderCooccurrence();
      });
    });
    common.observeOnce(cooccurrenceChart, () => {
      cooccurrenceReady = true;
      renderCooccurrence();
    });

    const organizerChart = document.getElementById("organizer-chart");
    const organizerCount = document.getElementById("organizer-count");
    const organizerToggle = document.getElementById("organizer-toggle");
    let showAllOrganizers = false;
    let organizerReady = false;
    function renderOrganizer() {
      organizerChart.classList.remove("chart-loading");
      if (!organizerChart.classList.contains("js-plotly-plot")) organizerChart.textContent = "";
      const organizerEpisodes = new Map();
      for (const row of viewData.scatter) {
        if (!organizerEpisodes.has(row.organizer)) organizerEpisodes.set(row.organizer, new Set());
        organizerEpisodes.get(row.organizer).add(row.episode);
      }
      const sortedOrganizers = [...organizerEpisodes.keys()].sort((a, b) => {
        if (a === "Korraldaja teadmata") return 1;
        if (b === "Korraldaja teadmata") return -1;
        return organizerEpisodes.get(b).size - organizerEpisodes.get(a).size || a.localeCompare(b, "et");
      });
      const organizers = showAllOrganizers ? sortedOrganizers : sortedOrganizers.slice(0, 20);
      organizerCount.textContent = `Kuvatakse ${organizers.length} / ${sortedOrganizers.length} korraldajat, järjestatud arutelude arvu järgi.`;
      organizerToggle.hidden = sortedOrganizers.length <= 20;
      organizerToggle.textContent = showAllOrganizers ? "Näita 20 suurimat" : `Näita kõiki (${sortedOrganizers.length})`;
      organizerToggle.setAttribute("aria-expanded", String(showAllOrganizers));
      const categories = viewData.categories.filter((category) => viewData.organizer_topics.some((row) => row.category === category.id));
      const lookup = new Map(viewData.organizer_topics.map((row) => [`${row.organizer}\u0000${row.category}`, row.talks]));
      const z = organizers.map((organizer) => categories.map((category) => lookup.get(`${organizer}\u0000${category.id}`) || 0));
      Plotly.react(organizerChart, [{
        type: "heatmap",
        x: categories.map((row) => row.name),
        y: organizers,
        z,
        customdata: organizers.map((organizer) => categories.map((category) => [organizer, category.id, category.name])),
        colorscale: [[0, "#edf3f0"], [.35, "#c5dcd3"], [1, "#8eb7a8"]],
        colorbar: { title: "Arutelusid" },
        hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[2]}<br>%{z} arutelu<extra></extra>",
      }], {
        ...common.chartLayout,
        height: Math.max(560, 26 * organizers.length + 200),
        xaxis: { tickangle: -35, side: "bottom", automargin: true },
        yaxis: { autorange: "reversed", automargin: true, ticklabelstandoff: 10 },
        margin: { l: 230, r: 30, t: 20, b: 180 },
      }, common.plotConfig).then((plot) => {
        plot.removeAllListeners("plotly_click");
        plot.on("plotly_click", (event) => {
          const [organizer, category] = event.points[0]?.customdata || [];
          if (organizer && category) {
            const params = new URLSearchParams({ category, organizer });
            if (yearFilter.value !== "all") params.set("year", yearFilter.value);
            location.href = `transcripts.html?${params}`;
          }
        });
      });
    }
    common.observeOnce(organizerChart, () => {
      organizerReady = true;
      renderOrganizer();
    });
    organizerToggle.addEventListener("click", () => {
      showAllOrganizers = !showAllOrganizers;
      if (organizerReady) renderOrganizer();
    });

    yearFilter.addEventListener("change", () => {
      viewData = buildYearView(data, yearFilter.value);
      treemapSelection.categoryId = "";
      treemapSelection.topic = "";
      updateMetrics();
      if (treemapReady) renderTreemap();
      if (scatterReady) scatterController.setView(activeScatterView);
      if (cooccurrenceReady) renderCooccurrence();
      if (organizerReady) renderOrganizer();
    });
  } catch (error) {
    common.showError(error);
  }
})();

function buildYearView(data, year) {
  const scatter = year === "all"
    ? data.scatter
    : data.scatter.filter((row) => String(row.year) === String(year));
  const categories = data.categories.map((category) => {
    const rows = scatter.filter((row) => row.category_id === category.id);
    return {
      ...category,
      segments: rows.length,
      talks: new Set(rows.map((row) => row.episode)).size,
      hours: rows.reduce(
        (total, row) => total + (Number(row.end) - Number(row.start)) / 3600,
        0,
      ),
    };
  });
  const organizerEpisodes = new Map();
  scatter.forEach((row) => {
    const key = `${row.organizer}\u0000${row.category_id}`;
    if (!organizerEpisodes.has(key)) organizerEpisodes.set(key, new Set());
    organizerEpisodes.get(key).add(row.episode);
  });
  const organizer_topics = [...organizerEpisodes].map(([key, episodes]) => {
    const separator = key.indexOf("\u0000");
    return {
      organizer: key.slice(0, separator),
      category: key.slice(separator + 1),
      talks: episodes.size,
    };
  });
  return { ...data, scatter, categories, organizer_topics };
}
