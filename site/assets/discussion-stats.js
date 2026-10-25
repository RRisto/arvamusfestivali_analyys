"use strict";

const episodeId = window.DISCUSSION_STATS_ID;
const dataUrl = `../data/discussion-stats/${encodeURIComponent(episodeId)}.json`;
const audio = document.getElementById("audio");

function percent(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

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
    xaxis: { title: "Recording minute", fixedrange: false },
    yaxis: { title: "Speech share", range: [0, 1], tickformat: ".0%" },
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
    ["dominant_speaker_share", "Dominant speaker share"],
    ["gini", "Gini"],
    ["effective_speaker_share", "Effective / detected speakers"],
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
    xaxis: { title: "Recording minute" },
    yaxis: { title: "Metric", range: [0, 1], tickformat: ".0%" },
    legend: { orientation: "h", y: -0.24 },
    hovermode: "x unified",
  }, { responsive: true, displaylogo: false });
}

function renderHeatmaps(data) {
  const speakerTopic = data.speaker_topic;
  if (!speakerTopic.topics.length || !speakerTopic.speakers.length) {
    document.getElementById("speaker-topic").hidden = true;
    document.getElementById("no-topics").hidden = false;
  } else {
    Plotly.newPlot("speaker-topic", [{
      x: speakerTopic.topics,
      y: speakerTopic.speakers,
      z: speakerTopic.shares,
      type: "heatmap",
      zmin: 0,
      zmax: 1,
      colorscale: "Blues",
      colorbar: { title: "Topic share", tickformat: ".0%" },
      hovertemplate: "%{y}<br>%{x}<br>%{z:.1%}<extra></extra>",
    }], {
      autosize: true,
      height: Math.max(330, 44 * speakerTopic.speakers.length + 120),
      margin: { l: 90, r: 30, t: 16, b: 145 },
      xaxis: { tickangle: -40, automargin: true },
      yaxis: { automargin: true, autorange: "reversed" },
    }, { responsive: true, displaylogo: false });
  }

  const transitions = data.transitions;
  if (!transitions.speakers.length) {
    document.getElementById("transitions").textContent = "No speaker transitions are available.";
    return;
  }
  Plotly.newPlot("transitions", [{
    x: transitions.speakers,
    y: transitions.speakers,
    z: transitions.shares,
    type: "heatmap",
    zmin: 0,
    zmax: 1,
    colorscale: "YlGnBu",
    colorbar: { title: "Transition share", tickformat: ".0%" },
    hovertemplate: "Current: %{y}<br>Next: %{x}<br>%{z:.1%}<extra></extra>",
  }], {
    autosize: true,
    height: Math.max(330, 48 * transitions.speakers.length + 120),
    margin: { l: 80, r: 30, t: 16, b: 90 },
    xaxis: { title: "Next speaker", tickangle: -35, automargin: true },
    yaxis: { title: "Current speaker", automargin: true, autorange: "reversed" },
  }, { responsive: true, displaylogo: false });
}

async function main() {
  try {
    const response = await fetch(dataUrl);
    if (!response.ok) throw new Error(`Could not load statistics (${response.status})`);
    const data = await response.json();
    document.title = `${data.title} — discussion statistics`;
    document.getElementById("organizer").textContent = data.organizer || "";
    audio.src = data.audio_url;
    document.getElementById("summary").innerHTML = [
      `${data.speaker_count} speakers`,
      `${data.speech_minutes.toFixed(1)} minutes of speech`,
      `${data.speaker_switches} speaker transitions`,
    ].map((item) => `<span class="pill">${item}</span>`).join("");
    renderParticipation(data);
    renderHeatmaps(data);
  } catch (error) {
    document.getElementById("summary").textContent = error.message;
  }
}

main();
