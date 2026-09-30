"""Behavioral contracts for inspectable topic-comparison figures."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from arvamusfestivali_transcripts.topic_analysis.plotting import (
    plot_episode_timeline,
    plot_episode_topic_heatmap,
    plot_semantic_map,
    plot_topic_correspondence,
    plot_topic_sizes,
)
from arvamusfestivali_transcripts.topic_analysis.types import Passage, TopicRun


def make_passages() -> tuple[Passage, ...]:
    return tuple(
        Passage(
            passage_id=f"p{index}",
            episode_id=f"episode-{index // 2}",
            duplicate_episode_ids=(f"episode-{index // 2}",),
            title=f"Recording {index // 2}",
            start_seconds=(index % 2) * 60.0,
            end_seconds=(index % 2 + 1) * 60.0,
            text=f"Passage {index} " + "detail " * 50,
            audio_sha256=f"{index + 1:x}" * 64,
            audio_url=f"https://example.test/audio/{index // 2}.mp3",
            word_count=52,
            cue_count=1,
        )
        for index in range(4)
    )


def make_run(
    model_key: str = "qwen", topics: tuple[int, ...] = (0, 1, 0, -1)
) -> TopicRun:
    return TopicRun(
        model_key=model_key,
        topics=np.array(topics),
        probabilities=None,
        reduced_embeddings=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        topic_info=pd.DataFrame({"Topic": [-1, 0, 1], "Name": ["Outlier", "Education", "Health"]}),
        representative_passages={},
        cluster_persistence=(),
        passage_ids=("p0", "p1", "p2", "p3"),
    )


@pytest.mark.parametrize(
    "builder",
    [plot_semantic_map, plot_topic_sizes, plot_episode_topic_heatmap, plot_episode_timeline],
)
def test_single_run_figures_show_model_and_timestamped_source(builder) -> None:
    figure = builder(make_run(), make_passages())

    assert isinstance(figure, go.Figure)
    payload = figure.to_plotly_json()
    assert "qwen" in payload["layout"]["title"]["text"].lower()
    serialized = json.dumps(payload)
    assert "#t=" in serialized
    assert "episode-" in serialized
    assert "Passage 0" in serialized


def test_semantic_map_labels_projection_as_unaligned() -> None:
    figure = plot_semantic_map(make_run(), make_passages())

    annotations = [item.text for item in figure.layout.annotations]
    assert any("not geometrically aligned" in text for text in annotations)
    assert figure.layout.xaxis.title.text
    assert figure.layout.yaxis.title.text


def test_semantic_map_truncates_excerpts_and_keeps_outlier_visible() -> None:
    figure = plot_semantic_map(make_run(), make_passages())

    assert sum(len(trace.x) for trace in figure.data) == 4
    assert any(trace.name == "Outlier (−1)" for trace in figure.data)
    excerpts = [row[4] for trace in figure.data for row in trace.customdata]
    assert all(len(excerpt) <= 240 for excerpt in excerpts)


def test_heatmap_uses_seconds_by_default_and_normalizes_each_episode_on_request() -> None:
    run = make_run()
    passages = make_passages()

    seconds = plot_episode_topic_heatmap(run, passages)
    normalized = plot_episode_topic_heatmap(run, passages, normalized=True)

    assert seconds.data[0].z == ([0, 60.0, 60.0], [60.0, 60.0, 0])
    assert normalized.data[0].z == ([0.0, 0.5, 0.5], [0.5, 0.5, 0.0])


def test_timeline_uses_common_minute_axis_for_passage_intervals() -> None:
    figure = plot_episode_timeline(make_run(), make_passages())

    intervals = sorted(
        (base, width)
        for trace in figure.data
        for base, width in zip(trace.base, trace.x, strict=True)
    )
    assert intervals == [(0.0, 1.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    assert "minute" in figure.layout.xaxis.title.text.lower()


def test_correspondence_counts_shared_passages_and_omits_zero_overlap() -> None:
    figure = plot_topic_correspondence(
        {"qwen": make_run(), "bge": make_run("bge", (2, 2, 3, -1))}, make_passages()
    )

    assert isinstance(figure, go.Figure)
    assert "qwen" in figure.layout.title.text.lower()
    assert "bge" in figure.layout.title.text.lower()
    assert sorted(figure.data[0].link.value) == [1, 1, 1]
    assert all(value > 0 for value in figure.data[0].link.value)
    assert "#t=" in json.dumps(figure.to_plotly_json())


def test_figures_reject_reordered_passages() -> None:
    with pytest.raises(ValueError, match="passage IDs and order"):
        plot_semantic_map(make_run(), tuple(reversed(make_passages())))


def test_correspondence_rejects_mismatched_run_order() -> None:
    bge = make_run("bge")
    object.__setattr__(bge, "passage_ids", tuple(reversed(bge.passage_ids)))
    with pytest.raises(ValueError, match="passage IDs and order"):
        plot_topic_correspondence({"qwen": make_run(), "bge": bge}, make_passages())


def test_all_outliers_produce_valid_empty_topic_subsets() -> None:
    run = make_run(topics=(-1, -1, -1, -1))
    passages = make_passages()

    for figure in (
        plot_topic_sizes(run, passages),
        plot_episode_topic_heatmap(run, passages),
        plot_episode_timeline(run, passages),
        plot_topic_correspondence(
            {"qwen": run, "bge": make_run("bge", (-1, -1, -1, -1))}, passages
        ),
    ):
        assert isinstance(figure, go.Figure)
        assert figure.layout.title.text


def _assert_readable_aggregate_hover(hovertemplate: str, row: tuple, count: int) -> None:
    assert "%{customdata[1]}" in hovertemplate
    assert f"{count} contributing passages" in row[1]
    assert "<br>" in row[1]
    assert row[1].count("episode-0") == count
    assert row[1].count("Recording 0") == count
    assert "0–60" in row[1]
    assert "60–120" in row[1]


def test_topic_size_hover_lists_every_contributing_passage() -> None:
    figure = plot_topic_sizes(make_run(topics=(0, 0, 1, 1)), make_passages())
    trace = figure.data[0]

    _assert_readable_aggregate_hover(trace.hovertemplate, trace.customdata[0], 2)
    assert "#t=0" in trace.customdata[0][1]
    assert "#t=60" in trace.customdata[0][1]


def test_heatmap_hover_lists_every_contributing_passage_in_cell() -> None:
    figure = plot_episode_topic_heatmap(make_run(topics=(0, 0, 1, 1)), make_passages())
    trace = figure.data[0]

    _assert_readable_aggregate_hover(trace.hovertemplate, trace.customdata[0][0], 2)
    assert "#t=0" in trace.customdata[0][0][1]
    assert "#t=60" in trace.customdata[0][0][1]


def test_correspondence_hover_lists_every_contributing_passage_in_link() -> None:
    figure = plot_topic_correspondence(
        {"qwen": make_run(topics=(0, 0, 1, 1)), "bge": make_run("bge", (2, 2, 3, 3))},
        make_passages(),
    )
    link = figure.data[0].link

    _assert_readable_aggregate_hover(link.hovertemplate, link.customdata[0], 2)
    assert "#t=0" in link.customdata[0][1]
    assert "#t=60" in link.customdata[0][1]
