"""Render saved native topic assignments for human inspection in a notebook."""

from __future__ import annotations

import hashlib
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
            out.append(
                f'<article class="card" style="--topic:{color}">'
                f"<strong>Segment {numbering[row.segment_key]} · "
                f"{_time(row.start_seconds)}–{_time(row.end_seconds)}</strong>"
                f"<h3>{int(row.topic_id)} · {escape(row.topic_name)}</h3>"
                f'<div class="meta">Membership strength: {confidence} · {audio_link}</div>'
                f'<div class="text">{escape(row.text)}</div>'
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
