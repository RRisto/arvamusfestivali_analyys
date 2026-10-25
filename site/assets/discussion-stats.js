"use strict";

const episodeId = window.DISCUSSION_STATS_ID;
const dataUrl = `../data/discussion-stats/${encodeURIComponent(episodeId)}.json`;
const audio = document.getElementById("audio");
let topicDetail = "detailed";
const LIGHT_TO_DARK = [
  [0, "#f7fbff"],
  [0.2, "#deebf7"],
  [0.4, "#9ecae1"],
  [0.6, "#6baed6"],
  [0.8, "#3182bd"],
  [1, "#08306b"],
];

function renderParticipation(data) {
  const participation = data.participation;
  const traces = Object.entries(participation.speaker_shares).map(([speaker, shares]) => ({
    x: participation.minutes,
    y: shares,
    name: speaker,
    type: "scatter",
    mode: "lines",
    stackgroup: "speakers",
    hovertemplate: `${speaker}<br>%{x:.1f} min · %{y:.0%}<extra></extra>`,
  }));
  Plotly.newPlot("participation", traces, {
    autosize: true,
    height: 360,
    margin: { l: 55, r: 20, t: 22, b: 55 },
    xaxis: { title: "Salvestise minut", fixedrange: false },
    yaxis: { title: "Kõneaja osakaal", range: [0, 1], tickformat: ".0%" },
    legend: { orientation: "h", y: -0.24 },
    hovermode: "x unified",
  }, { responsive: true, displaylogo: false }).then((chart) => {
    chart.on("plotly_click", (event) => {
      if (event.points?.length) {
        audio.currentTime = Math.max(0, event.points[0].x * 60);
        audio.play().catch(() => {});
      }
    });
  });

  const metrics = participation.metrics;
  const metricTraces = [
    ["dominant_speaker_share", "Valdava kõneleja osakaal"],
    ["gini", "Kõneaja ebavõrdsus (Gini)"],
    ["effective_speaker_share", "Aktiivsete kõnelejate osakaal"],
  ].map(([key, name]) => ({
    x: participation.minutes,
    y: metrics[key],
    name,
    type: "scatter",
    mode: "lines+markers",
    connectgaps: false,
  }));
  Plotly.newPlot("equality", metricTraces, {
    autosize: true,
    height: 310,
    margin: { l: 55, r: 20, t: 22, b: 55 },
    xaxis: { title: "Salvestise minut" },
    yaxis: { title: "Näitajad", range: [0, 1], tickformat: ".0%" },
    legend: { orientation: "h", y: -0.24 },
    hovermode: "x unified",
  }, { responsive: true, displaylogo: false });
}

function renderHeatmap(data) {
  const speakerTopic = data.speaker_topic[topicDetail];
  if (!speakerTopic.topics.length || !speakerTopic.speakers.length) {
    document.getElementById("speaker-topic").hidden = true;
    document.getElementById("no-topics").hidden = false;
  } else {
    document.getElementById("speaker-topic").hidden = false;
    document.getElementById("no-topics").hidden = true;
    Plotly.newPlot("speaker-topic", [{
      x: speakerTopic.topics,
      y: speakerTopic.speakers,
      z: speakerTopic.shares,
      type: "heatmap",
      zmin: 0,
      zmax: 1,
      colorscale: LIGHT_TO_DARK,
      colorbar: { title: "Teema kõneaja osakaal", tickformat: ".0%" },
      hovertemplate: "%{y}<br>%{x}<br>%{z:.1%}<extra></extra>",
    }], {
      autosize: true,
      height: Math.max(330, 44 * speakerTopic.speakers.length + 120),
      margin: { l: 90, r: 30, t: 16, b: 190 },
      xaxis: { tickangle: -40, automargin: true },
      yaxis: { automargin: true, autorange: "reversed" },
    }, { responsive: true, displaylogo: false });
  }
}

function setTopicDetail(detail) {
  topicDetail = detail;
  document.getElementById("detailed-topics").setAttribute("aria-pressed", String(detail === "detailed"));
  document.getElementById("high-level-topics").setAttribute("aria-pressed", String(detail === "high_level"));
  if (window.discussionData) renderHeatmap(window.discussionData);
}

function renderTransitions(data) {
  const transitions = data.transitions;
  if (!transitions.speakers.length) {
    document.getElementById("transitions").textContent = "Kõnelejate üleminekuid pole saadaval.";
    return;
  }
  Plotly.newPlot("transitions", [{
    x: transitions.speakers,
    y: transitions.speakers,
    z: transitions.shares,
    type: "heatmap",
    zmin: 0,
    zmax: 1,
    colorscale: LIGHT_TO_DARK,
    colorbar: { title: "Üleminekute osakaal", tickformat: ".0%" },
    hovertemplate: "Kõnelejalt: %{y}<br>Kõnelejale: %{x}<br>%{z:.1%}<extra></extra>",
  }], {
    autosize: true,
    height: Math.max(330, 48 * transitions.speakers.length + 120),
    margin: { l: 80, r: 30, t: 16, b: 90 },
    xaxis: { title: "Järgmine kõneleja", tickangle: -35, automargin: true },
    yaxis: { title: "Praegune kõneleja", automargin: true, autorange: "reversed" },
  }, { responsive: true, displaylogo: false });
}

async function main() {
  try {
    const response = await fetch(dataUrl);
    if (!response.ok) throw new Error(`Statistika laadimine ebaõnnestus (${response.status}).`);
    const data = await response.json();
    window.discussionData = data;
    document.title = `${data.title} — arutelu statistika`;
    document.getElementById("organizer").textContent = data.organizer || "";
    audio.src = data.audio_url;
    document.getElementById("summary").innerHTML = [
      `${data.speaker_count} kõnelejat`,
      `${data.speech_minutes.toFixed(1)} min kõneaega`,
      `${data.speaker_switches} kõnelejate vahetust`,
    ].map((item) => `<span class="pill">${item}</span>`).join("");
    renderParticipation(data);
    renderHeatmap(data);
    renderTransitions(data);
  } catch (error) {
    document.getElementById("summary").textContent = error.message || "Statistika laadimine ebaõnnestus.";
  }
}

document.getElementById("detailed-topics").addEventListener("click", () => setTopicDetail("detailed"));
document.getElementById("high-level-topics").addEventListener("click", () => setTopicDetail("high_level"));
main();
