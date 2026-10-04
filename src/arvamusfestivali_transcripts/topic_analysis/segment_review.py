"""Render saved native topic assignments for human inspection in a notebook."""

from __future__ import annotations

import hashlib
import json
from html import escape
from urllib.parse import urlsplit

import pandas as pd


def document_catalog(segments: pd.DataFrame) -> pd.DataFrame:
    """Stable zero-based document indices shared by both segmentation modes."""
    catalog = segments[["episode_id", "talk_name"]].drop_duplicates("episode_id").copy()
    catalog["episode_id"] = catalog.episode_id.astype(str)
    catalog["_sort"] = catalog.talk_name.str.casefold()
    return catalog.sort_values(["_sort", "episode_id"]).drop(columns="_sort").reset_index(drop=True)


def _time(seconds: float) -> str:
    value = int(seconds)
    hours, rest = divmod(value, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _color(key: str, topic: int) -> str:
    if topic == -1:
        return "#64748b"
    hue = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 360
    return f"hsl({hue},55%,42%)"


def segment_review_html(
    segments: pd.DataFrame,
    document_index: int,
    *,
    mode: str = "both",
    only_unassigned: bool = False,
    weak_only: bool = False,
    weak_threshold: float = 0.5,
    search: str = "",
) -> str:
    """Show full texts, timestamps, names and membership strength without changing labels."""
    if mode not in {"both", "semantic", "fixed"}:
        raise ValueError("Mode must be both, semantic or fixed")
    catalog = document_catalog(segments)
    if not 0 <= document_index < len(catalog):
        raise ValueError("Document index outside the catalog")
    doc = catalog.iloc[document_index]
    native = segments[segments.episode_id.astype(str) == doc.episode_id].copy()
    native = native.sort_values(["start_seconds", "end_seconds", "segment_key"])
    modes = ["semantic", "fixed"] if mode == "both" else [mode]
    end = max(float(native.end_seconds.max()), 1.0)
    selected = native
    if only_unassigned:
        selected = selected[selected.topic_id == -1]
    if weak_only:
        selected = selected[(selected.topic_id == -1) | (selected.confidence <= weak_threshold)]
    if search.strip():
        selected = selected[
            selected.text.str.contains(search, case=False, regex=False, na=False)
            | selected.topic_name.str.contains(search, case=False, regex=False, na=False)
        ]
    out = [
        '<div class="segment-review"><style>'
        ".segment-review{font:15px system-ui;color:#172033;line-height:1.5}"
        ".segment-review .columns{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(340px,1fr));"
        "gap:20px}.segment-review .card{border:1px solid #dce2ea;"
        "border-left:5px solid var(--topic);"
        "border-radius:8px;padding:14px;margin:12px 0;background:#fff}"
        ".segment-review .meta{color:#526077;font-size:13px}"
        ".segment-review .text{white-space:pre-wrap;overflow-wrap:anywhere;max-height:440px;"
        "overflow-y:auto;margin-top:12px}.segment-review .track{position:relative;height:46px;"
        "background:#eef2f6;border-radius:5px;margin:8px 0}"
        ".segment-review .interval{position:absolute;height:19px;min-width:2px;border-radius:2px}"
        ".segment-review h3{margin:10px 0}.segment-review details{margin-top:10px}"
        "@media(max-width:750px){.segment-review .columns{display:block}}"
        "</style>",
        f"<h2>Document {document_index}: {escape(doc.talk_name)}</h2>",
        f'<p class="meta">Episode {escape(doc.episode_id)} · '
        f"Analyzed recording ends at {_time(end)} · "
        "Topic IDs are scoped to each model. Gray indicates unassigned content.</p>",
    ]
    out.append('<div class="columns">')
    for current_mode in modes:
        all_rows = native[native.segmentation == current_mode]
        shown = selected[selected.segmentation == current_mode]
        topic_count = all_rows.loc[all_rows.topic_id >= 0, "topic_id"].nunique()
        out.append(
            f"<section><h3>{current_mode.title()} segmentation</h3>"
            f"<p>{len(all_rows)} segments · {topic_count}"
            f" assigned topics · {int((all_rows.topic_id == -1).sum())} unassigned</p>"
            '<div class="track" aria-label="Full recording topic timeline">'
        )
        # Alternate rows keep the overlapping fixed windows visible.
        for number, row in enumerate(all_rows.itertuples()):
            color = _color(row.topic_key, row.topic_id)
            tip = f"{number + 1}: {_time(row.start_seconds)}–{_time(row.end_seconds)} · "
            tip += f"{row.topic_id}: {row.topic_name}"
            out.append(
                f'<span class="interval" title="{escape(tip, quote=True)}" '
                f'style="left:{row.start_seconds / end * 100:.4f}%;'
                f"width:{(row.end_seconds - row.start_seconds) / end * 100:.4f}%;"
                f'top:{2 + number % 2 * 22}px;background:{color}"></span>'
            )
        out.append(
            f'</div><p class="meta">00:00 → {_time(end)} · '
            "Hover over the timeline for names and timestamps.</p>"
            f"<p>Showing {len(shown)} of {len(all_rows)} segments.</p>"
        )
        if shown.empty:
            out.append("<p>No segments match the filters.</p>")
        numbering = {key: i + 1 for i, key in enumerate(all_rows.segment_key)}
        for row in shown.itertuples():
            color = _color(row.topic_key, row.topic_id)
            confidence = "Unassigned" if row.topic_id == -1 else f"{row.confidence:.3f}"
            audio = str(row.audio_link)
            audio_link = (
                f'<a href="{escape(audio, quote=True)}" target="_blank" '
                'rel="noopener noreferrer">Listen at timestamp</a>'
                if urlsplit(audio).scheme in {"http", "https"}
                else ""
            )
            jev_html = ""
            if pd.notna(getattr(row, "jev_topic_id", None)):
                probabilities = json.loads(row.jev_probabilities)
                vocabulary = segments.loc[segments.segmentation == "semantic"].drop_duplicates(
                    "topic_id"
                )
                names = dict(zip(vocabulary.topic_id.astype(str), vocabulary.topic_name))
                names["-1"] = "Unassigned"
                alternatives = sorted(
                    probabilities.items(), key=lambda item: item[1], reverse=True
                )[:3]
                alternatives_html = "; ".join(
                    f"{escape(names.get(key, key))} ({probability:.1%})"
                    for key, probability in alternatives
                )
                jev_html = (
                    f"<p><strong>Jev: {int(row.jev_topic_id)} · "
                    f"{escape(row.jev_topic_name)}</strong><br>"
                    f'<span class="meta">Choice probability: {row.jev_probability:.1%} · '
                    f"Jev confidence: {row.jev_confidence:.3f} · {escape(row.jev_model)}<br>"
                    f"Top alternatives: {alternatives_html}</span></p>"
                )
                if pd.notna(getattr(row, "jev_high_level_name", None)):
                    jev_html += (
                        f'<p class="meta">High-level category (dictionary): '
                        f"{escape(row.jev_high_level_name)}</p>"
                    )
            elif "jev_topic_id" in segments.columns and current_mode == "semantic":
                jev_html = '<p class="meta">Jev: not sampled</p>'
            out.append(
                f'<article class="card" style="--topic:{color}">'
                f"<strong>Segment {numbering[row.segment_key]} · "
                f"{_time(row.start_seconds)}–{_time(row.end_seconds)}</strong>"
                f"<h3>{int(row.topic_id)} · {escape(row.topic_name)}</h3>"
                f'<div class="meta">Membership strength: {confidence} · {audio_link}</div>'
                f'{jev_html}<div class="text">{escape(row.text)}</div>'
                "<details><summary>Original keywords and segment metadata</summary>"
                f"<p>{escape(str(getattr(row, 'topic_name_keywords', row.topic_name)))}</p>"
                f'<p class="meta">Model: {escape(row.model_key)}<br>'
                f"Segment: {escape(row.segment_key)}<br>"
                f"Exact interval: {row.start_seconds:.2f}–{row.end_seconds:.2f} seconds</p>"
                "</details></article>"
            )
        out.append("</section>")
    out.append(
        '</div><p class="meta">Membership strength is HDBSCAN cluster membership, '
        "not a calibrated probability that the topic is correct. Full segment text is "
        "shown in scrollable cards. Filters affect cards; timelines show all segments.</p></div>"
    )
    return "".join(out)


def semantic_length_review_html(
    original: pd.DataFrame,
    shorter: pd.DataFrame,
    document_index: int,
) -> str:
    """Compare native original segments with shorter Jev-only segments on the same talk."""
    catalog = document_catalog(original)
    if not 0 <= document_index < len(catalog):
        raise ValueError("Document index outside the catalog")
    doc = catalog.iloc[document_index]
    rows = shorter[shorter.episode_id.astype(str) == doc.episode_id].sort_values("start_seconds")
    if rows.empty:
        return "<p>This document is outside the shorter-segment pilot.</p>"
    vocabulary = original[original.segmentation == "semantic"].drop_duplicates("topic_id")
    names = dict(zip(vocabulary.topic_id.astype(str), vocabulary.topic_name))
    names["-1"] = "Unassigned"
    old = original[
        (original.episode_id.astype(str) == doc.episode_id) & (original.segmentation == "semantic")
    ].sort_values("start_seconds")
    numbering = {row.segment_key: i + 1 for i, row in enumerate(old.itertuples())}
    old_lookup = old.set_index("segment_key")
    out = [
        '<div class="segment-review"><style>'
        ".length-comparison{display:grid;grid-template-columns:1fr 1fr;gap:24px}"
        ".length-comparison>section{min-width:0}"
        "@media(max-width:900px){.length-comparison{display:block}}"
        "</style><p>Original and shorter native boundaries. Shorter segments have Jev "
        "predictions only; original cluster references below are temporal overlaps.</p>"
        '<div class="length-comparison"><section><h2>Original semantic segments</h2>',
        segment_review_html(original, document_index, mode="semantic"),
        "</section><section><h2>Shorter semantic segments</h2>",
        f"<p>{len(rows)} segments · median {rows.duration_seconds.median():.0f}s · "
        f"maximum {rows.duration_seconds.max():.0f}s</p>",
    ]
    for number, row in enumerate(rows.itertuples(), 1):
        color = _color(row.jev_topic_key, int(row.jev_topic_id))
        alternatives = sorted(
            json.loads(row.jev_probabilities).items(), key=lambda item: item[1], reverse=True
        )[:3]
        top = "; ".join(f"{escape(names.get(key, key))} ({p:.1%})" for key, p in alternatives)
        overlaps = []
        for overlap in json.loads(row.original_segment_overlaps):
            key = overlap["segment_key"]
            old_row = old_lookup.loc[key]
            overlaps.append(
                f"Original segment {numbering[key]}: {escape(old_row.topic_name)} "
                f"({overlap['overlap_seconds']:.0f}s overlap)"
            )
        audio = str(row.audio_link)
        link = (
            f'<a href="{escape(audio, quote=True)}" target="_blank" '
            'rel="noopener noreferrer">Listen at timestamp</a>'
            if urlsplit(audio).scheme in {"http", "https"}
            else ""
        )
        out.append(
            f'<article class="card" style="--topic:{color}">'
            f"<strong>Short segment {number} · {_time(row.start_seconds)}–"
            f"{_time(row.end_seconds)}</strong>"
            f"<h3>Jev: {int(row.jev_topic_id)} · {escape(row.jev_topic_name)}</h3>"
            f'<p class="meta">Choice probability: {row.jev_probability:.1%} · '
            f"Jev confidence: {row.jev_confidence:.3f} · {link}<br>"
            f"Top alternatives: {top}</p>"
            f'<div class="text">{escape(row.text)}</div>'
            "<details><summary>Original overlaps and provenance</summary>"
            f"<p>{'<br>'.join(overlaps)}</p>"
            f'<p class="meta">{escape(row.segment_key)}<br>'
            f"{row.cue_count} original transcript cues · {escape(row.jev_model)}</p>"
            "</details></article>"
        )
    out.append("</section></div></div>")
    return "".join(out)


def segmentation_experiment_review_html(
    original: pd.DataFrame,
    variants: dict[str, pd.DataFrame],
    document_index: int,
    *,
    left: str = "cue5",
    right: str = "sentence5",
    start_seconds: float = 0,
    end_seconds: float | None = None,
    assignment_source: str = "both",
) -> str:
    """Compare any two experiments on a shared recording/time window, preserving native text."""
    if assignment_source not in {"both", "topic_model", "jev"}:
        raise ValueError("Assignment source must be both, topic_model or jev")
    catalog = document_catalog(original)
    if not 0 <= document_index < len(catalog):
        raise ValueError("Document index outside the catalog")
    if start_seconds < 0 or (end_seconds is not None and end_seconds <= start_seconds):
        raise ValueError("Time window must be nonnegative and ordered")
    doc = catalog.iloc[document_index]
    vocabulary = original[original.segmentation == "semantic"].drop_duplicates("topic_id")
    names = dict(zip(vocabulary.topic_id.astype(str), vocabulary.topic_name))
    names["-1"] = "Unassigned"
    labels = {
        "original": "Original semantic",
        "short3": "Cue cuts · 3 minutes",
        "cue5": "Cue cuts · 5 minutes",
        "sentence3": "Sentence cuts · 3 minutes",
        "sentence5": "Sentence cuts · 5 minutes",
    }
    out = [
        '<div class="experiment-review"><style>'
        ".experiment-review{font:15px system-ui;color:#172033;line-height:1.5}"
        ".experiment-review .pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}"
        ".experiment-review .card{border:1px solid #ddd;border-left:5px solid var(--topic);"
        "border-radius:8px;padding:14px;margin:12px 0;background:#fff}"
        ".experiment-review .text{white-space:pre-wrap;max-height:420px;overflow:auto;"
        "overflow-wrap:anywhere}.experiment-review .meta{font-size:13px;color:#526077}"
        "@media(max-width:900px){.experiment-review .pair{display:block}}"
        "</style>",
        f"<h2>Document {document_index}: {escape(doc.talk_name)}</h2>"
        "<p>Same recording and time window; each card retains its full native text. "
        "Topic model IDs belong to each independent fit. "
        "Sentence endings use an ASR punctuation heuristic. Jev probabilities are "
        'primary-topic alternatives, not measured accuracy.</p><div class="pair">',
    ]
    for key in [left, right]:
        if key not in variants:
            raise ValueError(f"Unknown experiment: {key}")
        all_rows = variants[key]
        all_rows = all_rows[
            (all_rows.episode_id.astype(str) == doc.episode_id)
            & (all_rows.segmentation == "semantic")
        ].sort_values("start_seconds")
        shown = all_rows[all_rows.end_seconds > start_seconds]
        if end_seconds is not None:
            shown = shown[shown.start_seconds < end_seconds]
        out.append(
            f"<section><h3>{escape(labels.get(key, key))}</h3>"
            f"<p>Showing {len(shown)} of {len(all_rows)} segments.</p>"
        )
        numbering = {row.segment_key: i + 1 for i, row in enumerate(all_rows.itertuples())}
        for row in shown.itertuples():
            has_jev = pd.notna(getattr(row, "jev_topic_id", None))
            color = _color(row.jev_topic_key, int(row.jev_topic_id)) if has_jev else "#64748b"
            has_cluster = pd.notna(getattr(row, "cluster_topic_id", None))
            if has_cluster and assignment_source != "jev":
                color = _color(row.cluster_topic_key, int(row.cluster_topic_id))
            cluster_label = ""
            if has_cluster and assignment_source != "jev":
                membership = (
                    "Unassigned"
                    if row.cluster_topic_id == -1
                    else f"{row.cluster_membership_strength:.3f}"
                )
                cluster_label = (
                    f"<h3>Topic model: {int(row.cluster_topic_id)} · "
                    f"{escape(row.cluster_topic_name)}</h3>"
                    f'<p class="meta">Cluster membership strength: {membership}<br>'
                    f"Fit: {escape(row.cluster_model_key)}</p>"
                )
            original_label = (
                f"<p>Original full-corpus cluster: {int(row.topic_id)} · "
                f"{escape(row.topic_name)}</p>"
                if key == "original"
                else ""
            )
            prediction = "<p>Jev: not sampled</p>"
            if has_jev:
                top = sorted(
                    json.loads(row.jev_probabilities).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
                alternatives = "; ".join(f"{escape(names.get(t, t))} ({p:.1%})" for t, p in top)
                prediction = (
                    f"<h3>Jev: {int(row.jev_topic_id)} · {escape(row.jev_topic_name)}</h3>"
                    f'<p class="meta">Choice probability: {row.jev_probability:.1%} · '
                    f"Confidence: {row.jev_confidence:.3f}<br>{alternatives}</p>"
                )
            if assignment_source == "topic_model":
                prediction = "" if has_cluster else "<p>Topic-model refit not available.</p>"
            ending = getattr(row, "ends_at_sentence", None)
            ending_label = (
                f" · Sentence-ending punctuation: {'yes' if ending else 'no'}"
                if ending is not None
                else ""
            )
            audio = str(row.audio_link)
            link = (
                f'<a href="{escape(audio, quote=True)}" target="_blank" '
                'rel="noopener noreferrer">Listen</a>'
                if urlsplit(audio).scheme in {"http", "https"}
                else ""
            )
            out.append(
                f'<article class="card" style="--topic:{color}">'
                f"<strong>Segment {numbering[row.segment_key]} · {_time(row.start_seconds)}–"
                f'{_time(row.end_seconds)}</strong><p class="meta">'
                f"{row.end_seconds - row.start_seconds:.0f}s{ending_label} · {link}</p>"
                f"{original_label}{cluster_label}{prediction}"
                f'<div class="text">{escape(row.text)}</div>'
                f"<details><summary>Segment provenance</summary>"
                f"<p>{escape(row.segment_key)}</p></details></article>"
            )
        if shown.empty:
            out.append("<p>No pilot segments in this recording/time window.</p>")
        out.append("</section>")
    out.append("</div></div>")
    return "".join(out)
