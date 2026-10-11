import {
  buildTopicBubbles,
  buildTopicScatterLayout,
  buildTopicScatterTraces,
  buildTreemapLevel,
  createLazyViewController,
  heatmapText,
} from "./overview-core.mjs?v=topic-bubbles3";

(async function () {
  "use strict";

  const common = window.siteCommon;
  try {
    const data = await common.fetchJson("data/overview.json");
    document.getElementById("metric-talks").textContent = common.formatNumber(data.metrics.talks);
    document.getElementById("metric-segments").textContent = common.formatNumber(data.metrics.segments);
    document.getElementById("metric-categories").textContent = common.formatNumber(data.metrics.categories);

    let treemapMetric = "talks";
    let treemapReady = false;
    const treemapSelection = { categoryId: "", topic: "" };
    const treemap = document.getElementById("treemap-chart");
    const treemapBreadcrumb = document.getElementById("treemap-breadcrumb");

    function renderTreemapBreadcrumb() {
      treemapBreadcrumb.replaceChildren();
      const category = data.categories.find((item) => item.id === treemapSelection.categoryId);
      const crumbs = [
        { label: "Kõik põhiteemad", categoryId: "", topic: "" },
      ];
      if (category) crumbs.push({ label: category.name, categoryId: category.id, topic: "" });
      if (treemapSelection.topic) {
        crumbs.push({ label: treemapSelection.topic, categoryId: category.id, topic: treemapSelection.topic });
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
      const isTalks = treemapMetric === "talks";
      const nodes = buildTreemapLevel(data, treemapSelection, treemapMetric);
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
        customdata: nodes.map((row) => [row.level, row.id, row.talks, row.hours, row.episode, row.segment, row.categoryId, row.displayValue]),
        branchvalues: "total",
        texttemplate: `<b>%{label}</b><br>%{customdata[7]:${isTalks ? ".0f" : ".1f"}} ${isTalks ? "arutelu" : "tundi"}`,
        hovertemplate: "<b>%{label}</b><br>%{customdata[2]} arutelu<br>%{customdata[3]:.1f} tundi<extra></extra>",
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
    function renderScatter(scatterView) {
      scatter.classList.remove("chart-loading");
      if (!scatter.classList.contains("js-plotly-plot")) scatter.textContent = "";
      if (scatterView === "topics") {
        const bubbles = buildTopicBubbles(data.scatter, data.categories);
        document.getElementById("scatter-description").textContent = "Iga mull on üks alamteema. Asukoht põhineb selle lõikude keskpunktil, suurus arutelude arvul ja värv põhiteemal.";
        document.getElementById("scatter-help").textContent = "Hõljuta kõigi alamteemade nimede nägemiseks. Mull avab vastavad lõigud transkriptiotsingus.";
        const traces = buildTopicScatterTraces(bubbles, data.categories);
        return Plotly.react(scatter, traces, buildTopicScatterLayout(common.chartLayout), common.plotConfig).then((plot) => {
          plot.removeAllListeners("plotly_click");
          plot.on("plotly_click", (event) => {
            const [category, , topic] = event.points[0]?.customdata || [];
            if (category && topic) {
              const params = new URLSearchParams({ category, q: topic });
              location.href = `transcripts.html?${params}`;
            }
          });
        });
      }

      document.getElementById("scatter-description").textContent = "Iga punkt on üks semantiline lõik. Lähedus näitab tekstilist sarnasust, värv põhiteemat. Kahemõõtmeline kaugus on ligikaudne.";
      document.getElementById("scatter-help").textContent = "Hõljuta lisainfo nägemiseks. Punkt avab valitud lõigu ja ajastatud helilingi.";
      const grouped = new Map();
      data.scatter.forEach((row) => {
        if (!grouped.has(row.category_id)) grouped.set(row.category_id, []);
        grouped.get(row.category_id).push(row);
      });
      const traces = [...grouped.entries()].map(([categoryId, rows]) => {
        const category = data.categories.find((item) => item.id === categoryId);
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
          const row = data.scatter.find((item) => item.id === segment);
          if (row) location.href = `transcripts.html?episode=${encodeURIComponent(row.episode)}&segment=${encodeURIComponent(row.id)}`;
        });
      });
    }
    const scatterController = createLazyViewController("topics", renderScatter);
    document.querySelectorAll("[data-scatter-view]").forEach((button) => {
      button.addEventListener("click", () => {
        scatterController.setView(button.dataset.scatterView);
        document.querySelectorAll("[data-scatter-view]").forEach((candidate) => {
          const selected = candidate === button;
          candidate.classList.toggle("active", selected);
          candidate.setAttribute("aria-pressed", String(selected));
        });
      });
    });
    common.observeOnce(scatter, () => scatterController.mount());

    const organizerChart = document.getElementById("organizer-chart");
    common.observeOnce(organizerChart, () => {
      organizerChart.classList.remove("chart-loading");
      organizerChart.textContent = "";
      const organizers = [...new Set(data.organizer_topics.map((row) => row.organizer))];
      const categories = data.categories.filter((category) => data.organizer_topics.some((row) => row.category === category.id));
      const lookup = new Map(data.organizer_topics.map((row) => [`${row.organizer}\u0000${row.category}`, row.talks]));
      const z = organizers.map((organizer) => categories.map((category) => lookup.get(`${organizer}\u0000${category.id}`) || 0));
      Plotly.newPlot(organizerChart, [{
        type: "heatmap",
        x: categories.map((row) => row.name),
        y: organizers,
        z,
        text: heatmapText(z),
        texttemplate: "%{text}",
        textfont: { color: "#102e26", size: 11 },
        customdata: organizers.map((organizer) => categories.map((category) => [organizer, category.id, category.name])),
        colorscale: [[0, "#edf3f0"], [.35, "#c5dcd3"], [1, "#8eb7a8"]],
        colorbar: { title: "Arutelusid" },
        hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[2]}<br>%{z} arutelu<extra></extra>",
      }], {
        ...common.chartLayout,
        xaxis: { tickangle: -35, side: "bottom", automargin: true },
        yaxis: { automargin: true },
        margin: { l: 230, r: 30, t: 20, b: 180 },
      }, common.plotConfig);
      organizerChart.on("plotly_click", (event) => {
        const [organizer, category] = event.points[0]?.customdata || [];
        if (organizer && category) location.href = `transcripts.html?category=${encodeURIComponent(category)}&organizer=${encodeURIComponent(organizer)}`;
      });
    });
  } catch (error) {
    common.showError(error);
  }
})();
