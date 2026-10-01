"""Local deterministic semantic segmentation on cue-aligned atomic blocks."""

from __future__ import annotations

from collections.abc import Sequence

from .types import CanonicalEpisode, Passage, SemanticSegmentationConfig


def _passage(episode: CanonicalEpisode, indices: Sequence[int]) -> Passage:
    cues = tuple(episode.cues[index] for index in indices)
    start = cues[0].start_seconds
    end = cues[-1].end_seconds
    return Passage(
        passage_id=(
            f"{episode.episode_id}:{round(start * 1000):012d}-{round(end * 1000):012d}"
        ),
        episode_id=episode.episode_id,
        duplicate_episode_ids=episode.duplicate_episode_ids,
        title=episode.title,
        start_seconds=start,
        end_seconds=end,
        text=" ".join(cue.text for cue in cues),
        audio_sha256=episode.audio_sha256,
        audio_url=episode.audio_url,
        word_count=sum(len(cue.text.split()) for cue in cues),
        cue_count=len(cues),
    )


def build_atomic_blocks(
    episode: CanonicalEpisode,
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> tuple[Passage, ...]:
    """Accumulate whole cues into short, deterministic, non-overlapping blocks."""
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    for index, cue in enumerate(episode.cues):
        if current:
            prospective_duration = cue.end_seconds - episode.cues[current[0]].start_seconds
            if prospective_duration > config.atomic_max_seconds:
                groups.append(tuple(current))
                current = []
        current.append(index)
        duration = cue.end_seconds - episode.cues[current[0]].start_seconds
        if duration >= config.atomic_target_seconds:
            groups.append(tuple(current))
            current = []
    if current:
        groups.append(tuple(current))
    return tuple(_passage(episode, indices) for indices in groups)


def build_atomic_blocks_many(
    episodes: Sequence[CanonicalEpisode],
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> tuple[Passage, ...]:
    """Build atomic blocks while preserving the caller's episode order."""
    if not episodes:
        raise ValueError("episodes must be nonempty")
    return tuple(
        block
        for episode in episodes
        for block in build_atomic_blocks(episode, config)
    )
