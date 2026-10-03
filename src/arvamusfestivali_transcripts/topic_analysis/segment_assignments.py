"""Join fitted topic labels to native segments and link temporal overlap across models."""

from __future__ import annotations

import json

import pandas as pd


def name_segment_topics(
    assignments: pd.DataFrame, topic_info: pd.DataFrame, *, model_key: str
) -> pd.DataFrame:
    """Preserve segment provenance and attach automatic names and membership strength."""
    required = {
        "passage_id",
        "episode_id",
        "title",
        "start_seconds",
        "end_seconds",
        "topic_id",
        "membership_strength",
        "text",
        "audio_link",
    }
    if missing := required - set(assignments.columns):
        raise ValueError(f"Missing assignment columns: {sorted(missing)}")
    if not {"Topic", "Name"}.issubset(topic_info.columns):
        raise ValueError("Topic table must contain Topic and Name")
    if assignments.passage_id.duplicated().any() or topic_info.Topic.duplicated().any():
        raise ValueError("Segment IDs and topic IDs must be unique within their tables")
    if not set(assignments.topic_id).issubset(set(topic_info.Topic)):
        raise ValueError("Assignments contain unknown topic IDs")
    if (assignments.end_seconds <= assignments.start_seconds).any():
        raise ValueError("Segment end must follow its start")
    confidence = assignments.membership_strength.dropna()
    if not confidence.between(0, 1).all():
        raise ValueError("Membership strength must lie between zero and one")
    if assignments.loc[assignments.topic_id != -1, "membership_strength"].isna().any():
        raise ValueError("Assigned segments require membership strength")

    names = topic_info.set_index("Topic").Name
    result = assignments.copy()
    result.insert(0, "model_key", model_key)
    result["topic_key"] = model_key + ":" + result.topic_id.astype(str)
    result["topic_label"] = result.topic_id.map(names)
    result["topic_name"] = result.topic_label.str.replace(r"^-?\d+_", "", regex=True).str.replace(
        "_", ", ", regex=False
    )
    result.loc[result.topic_id == -1, "topic_name"] = "Unassigned"
    if "LLM_Name" in topic_info.columns:
        llm_names = topic_info.set_index("Topic").LLM_Name
        result["topic_name_keywords"] = result.topic_name
        result["topic_name"] = result.topic_id.map(llm_names).fillna(result.topic_name)
    result["confidence"] = result.membership_strength.where(result.topic_id != -1)
    result["confidence_type"] = "hdbscan_membership_strength"
    result["talk_name"] = result.title
    result["duration_seconds"] = result.end_seconds - result.start_seconds
    return result


def link_overlapping_segments(segments: pd.DataFrame, other_segments: pd.DataFrame) -> pd.DataFrame:
    """Keep exact opposite-model labels; temporal overlap is not a second prediction.

    Coverage fractions may sum above one when the other model uses overlapping windows.
    Membership strength belongs to the other native segment, not the target segment.
    """
    other_by_episode = {
        episode_id: group.sort_values("start_seconds")
        for episode_id, group in other_segments.groupby("episode_id", sort=False)
    }
    links = []
    for segment in segments.itertuples():
        matches = []
        candidates = other_by_episode.get(segment.episode_id)
        if candidates is not None:
            candidates = candidates[
                (candidates.start_seconds < segment.end_seconds)
                & (candidates.end_seconds > segment.start_seconds)
            ]
            for other in candidates.itertuples():
                overlap = min(segment.end_seconds, other.end_seconds) - max(
                    segment.start_seconds, other.start_seconds
                )
                matches.append(
                    {
                        "model_key": other.model_key,
                        "passage_id": other.passage_id,
                        "topic_id": int(other.topic_id),
                        "topic_key": other.topic_key,
                        "topic_name": other.topic_name,
                        "confidence": None
                        if pd.isna(other.confidence)
                        else float(other.confidence),
                        "start_seconds": float(other.start_seconds),
                        "end_seconds": float(other.end_seconds),
                        "overlap_seconds": float(overlap),
                        "fraction_of_target_segment": float(overlap / segment.duration_seconds),
                    }
                )
        links.append(json.dumps(matches, ensure_ascii=False, allow_nan=False))
    result = segments.copy()
    result["other_model_overlaps"] = links
    return result
