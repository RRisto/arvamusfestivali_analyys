"""Inspectable Plotly views of topic assignments and model correspondence."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from html import escape

import plotly.graph_objects as go
from plotly.colors import qualitative

from .types import Passage, TopicRun

_COLORS = qualitative.Safe
_OUTLIER_COLOR = "#777777"
_HOVER = (
    "<b>%{customdata[0]}</b><br>Episode %{customdata[1]}<br>"
    "%{customdata[2]:.1f}–%{customdata[3]:.1f} s<br>"
    "%{customdata[4]}<br>%{customdata[5]}<extra></extra>"
)


def _validate(run: TopicRun, passages: Sequence[Passage]) -> None:
    ids = tuple(passage.passage_id for passage in passages)
    if run.passage_ids != ids:
        raise ValueError("run and passage IDs and order must match")
    if len(set(ids)) != len(ids):
        raise ValueError("passage IDs must be unique")


def _topic_label(topic: int) -> str:
    return "Outlier (−1)" if topic == -1 else f"Topic {topic}"


def _topic_color(topic: int) -> str:
    return _OUTLIER_COLOR if topic == -1 else _COLORS[topic % len(_COLORS)]


def _source(passage: Passage) -> list[str | float]:
    excerpt = passage.text.strip().replace("\n", " ")
    if len(excerpt) > 240:
        excerpt = excerpt[:239].rstrip() + "…"
    return [
        passage.title,
        passage.episode_id,
        passage.start_seconds,
        passage.end_seconds,
        excerpt,
        passage.timestamped_audio_url,
    ]


def _group_source(passages: Sequence[Passage]) -> list[int | str]:
    count = len(passages)
    if not passages:
        return [0, "0 contributing passages", "[]"]
    lines = [f"{count} contributing passage{'s' if count != 1 else ''}:"]
    for index, passage in enumerate(passages, start=1):
        title, episode, start, end, excerpt, audio_url = _source(passage)
        lines.append(
            f"{index}. {escape(str(title))} · {escape(str(episode))} · "
            f"{start:g}–{end:g} s · {escape(str(excerpt))} · "
            f"{escape(str(audio_url))}"
        )
    return [count, "<br>".join(lines), json.dumps([_source(item) for item in passages])]


def _base(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_white",
        font={"size": 13},
        margin={"l": 90, "r": 35, "t": 95, "b": 75},
        hoverlabel={"align": "left"},
    )
    return figure


def plot_semantic_map(run: TopicRun, passages: Sequence[Passage]) -> go.Figure:
    """Show one run's 2-D reduction; positions are comparable only within this run."""
    _validate(run, passages)
    if run.reduced_embeddings.shape[1] < 2:
        raise ValueError("semantic map requires a two-dimensional projection")
    figure = _base(f"{run.model_key}: semantic map of passage topics")
    for topic in sorted({int(value) for value in run.topics}):
        indices = [index for index, value in enumerate(run.topics) if int(value) == topic]
        figure.add_trace(go.Scatter(
            x=[float(run.reduced_embeddings[index, 0]) for index in indices],
            y=[float(run.reduced_embeddings[index, 1]) for index in indices],
            mode="markers",
            name=_topic_label(topic),
            marker={"color": _topic_color(topic), "size": 10, "opacity": 0.85},
            customdata=[_source(passages[index]) for index in indices],
            hovertemplate=_HOVER,
        ))
    figure.update_xaxes(title="Projection dimension 1")
    figure.update_yaxes(title="Projection dimension 2")
    figure.add_annotation(
        text=(
            "Separate model projections are not geometrically aligned; "
            "compare topic assignments, not absolute positions."
        ),
        xref="paper", yref="paper", x=0, y=-0.2, showarrow=False,
        xanchor="left", align="left", font={"size": 11},
    )
    return figure


def plot_topic_sizes(run: TopicRun, passages: Sequence[Passage]) -> go.Figure:
    """Count passages per topic, retaining source provenance for each bar."""
    _validate(run, passages)
    grouped: dict[int, list[Passage]] = defaultdict(list)
    for topic, passage in zip(run.topics, passages, strict=True):
        grouped[int(topic)].append(passage)
    topics = sorted(grouped)
    figure = _base(f"{run.model_key}: passage counts by topic")
    figure.add_trace(go.Bar(
        x=[_topic_label(topic) for topic in topics],
        y=[len(grouped[topic]) for topic in topics],
        marker_color=[_topic_color(topic) for topic in topics],
        customdata=[_group_source(grouped[topic]) for topic in topics],
        hovertemplate="%{x}: %{y} passages<br>%{customdata[1]}<extra></extra>",
    ))
    figure.update_xaxes(title="Topic ID")
    figure.update_yaxes(title="Passage count", rangemode="tozero")
    return figure


def plot_episode_topic_heatmap(
    run: TopicRun, passages: Sequence[Passage], *, normalized: bool = False
) -> go.Figure:
    """Aggregate passage durations by episode and topic, optionally by row share."""
    _validate(run, passages)
    episodes = list(dict.fromkeys(passage.episode_id for passage in passages))
    topics = sorted({int(topic) for topic in run.topics})
    grouped: dict[tuple[str, int], list[Passage]] = defaultdict(list)
    for topic, passage in zip(run.topics, passages, strict=True):
        grouped[(passage.episode_id, int(topic))].append(passage)
    seconds = [
        [sum(item.end_seconds - item.start_seconds for item in grouped[(episode, topic)])
         for topic in topics]
        for episode in episodes
    ]
    values = []
    for row in seconds:
        total = sum(row)
        values.append([value / total if total else 0.0 for value in row] if normalized else row)
    figure = _base(
        f"{run.model_key}: episode topic {'duration share' if normalized else 'duration'}"
    )
    figure.add_trace(go.Heatmap(
        x=[_topic_label(topic) for topic in topics],
        y=episodes,
        z=values,
        customdata=[[_group_source(grouped[(episode, topic)]) for topic in topics]
                    for episode in episodes],
        colorscale="Blues",
        colorbar={"title": "Share" if normalized else "Seconds"},
        hovertemplate="Episode %{y}<br>%{x}: %{z:.2f}<br>"
        "%{customdata[1]}<extra></extra>",
    ))
    figure.update_xaxes(title="Topic ID")
    figure.update_yaxes(title="Episode ID")
    return figure


def plot_episode_timeline(run: TopicRun, passages: Sequence[Passage]) -> go.Figure:
    """Place passage intervals on a common minute axis for each recording."""
    _validate(run, passages)
    figure = _base(f"{run.model_key}: passage topic timeline by episode")
    for topic in sorted({int(value) for value in run.topics}):
        selected = [passage for value, passage in zip(run.topics, passages, strict=True)
                    if int(value) == topic]
        figure.add_trace(go.Bar(
            x=[(item.end_seconds - item.start_seconds) / 60 for item in selected],
            base=[item.start_seconds / 60 for item in selected],
            y=[f"{item.episode_id} · {item.passage_id}" for item in selected],
            orientation="h",
            name=_topic_label(topic),
            marker_color=_topic_color(topic),
            customdata=[_source(item) for item in selected],
            hovertemplate=_HOVER,
        ))
    figure.update_layout(barmode="overlay", height=max(400, 85 + 32 * len(passages)))
    figure.update_xaxes(title="Recording time (minutes)", rangemode="tozero")
    figure.update_yaxes(title="Episode · passage", autorange="reversed")
    return figure


def plot_topic_correspondence(
    runs: Mapping[str, TopicRun], passages: Sequence[Passage]
) -> go.Figure:
    """Link two runs by counts of passages assigned to both non-outlier topics."""
    if len(runs) != 2:
        raise ValueError("topic correspondence requires exactly two model runs")
    (first_key, first), (second_key, second) = runs.items()
    if first.model_key != first_key or second.model_key != second_key:
        raise ValueError("run model keys must match mapping keys")
    _validate(first, passages)
    _validate(second, passages)
    overlap: dict[tuple[int, int], list[Passage]] = defaultdict(list)
    for left, right, passage in zip(first.topics, second.topics, passages, strict=True):
        if int(left) != -1 and int(right) != -1:
            overlap[(int(left), int(right))].append(passage)
    left_topics = sorted({left for left, _ in overlap})
    right_topics = sorted({right for _, right in overlap})
    labels = [f"{first_key} · {_topic_label(topic)}" for topic in left_topics]
    labels += [f"{second_key} · {_topic_label(topic)}" for topic in right_topics]
    left_indices = {topic: index for index, topic in enumerate(left_topics)}
    right_indices = {
        topic: index + len(left_topics) for index, topic in enumerate(right_topics)
    }
    pairs = sorted(overlap)
    figure = _base(f"{first_key} ↔ {second_key}: shared-passage topic correspondence")
    figure.add_trace(go.Sankey(
        node={"label": labels, "color": [
            _topic_color(topic) for topic in (*left_topics, *right_topics)
        ]},
        link={
            "source": [left_indices[left] for left, _ in pairs],
            "target": [right_indices[right] for _, right in pairs],
            "value": [len(overlap[pair]) for pair in pairs],
            "customdata": [_group_source(overlap[pair]) for pair in pairs],
            "hovertemplate": "%{value} shared passages<br>"
                             "%{customdata[1]}<extra></extra>",
        },
    ))
    figure.add_annotation(
        text="Links count shared passage assignments; outliers and zero-overlap links are omitted.",
        xref="paper", yref="paper", x=0, y=-0.12, showarrow=False, xanchor="left",
    )
    return figure
