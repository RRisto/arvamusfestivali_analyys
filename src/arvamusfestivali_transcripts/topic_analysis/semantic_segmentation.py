"""Local deterministic semantic segmentation on cue-aligned atomic blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .types import (
    CanonicalEpisode,
    EmbeddingResult,
    Passage,
    SemanticBoundary,
    SemanticSegmentationConfig,
    SemanticSegmentationResult,
)


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


def _validate_blocks(blocks: Sequence[Passage]) -> tuple[Passage, ...]:
    items = tuple(blocks)
    if not items:
        raise ValueError("atomic blocks must be nonempty")
    episode_ids = {item.episode_id for item in items}
    if len(episode_ids) != 1:
        raise ValueError("atomic blocks must belong to one episode")
    if len({item.passage_id for item in items}) != len(items):
        raise ValueError("atomic block IDs must be unique")
    for left, right in zip(items, items[1:]):
        if left.start_seconds >= right.start_seconds or left.end_seconds > right.start_seconds:
            raise ValueError("atomic blocks must be ordered and nonoverlapping")
    return items


def _validate_embeddings(blocks: Sequence[Passage], embeddings: np.ndarray) -> np.ndarray:
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(blocks)
        or embeddings.shape[1] < 1
    ):
        raise ValueError("embedding rows must match atomic blocks")
    if not np.issubdtype(embeddings.dtype, np.number) or not np.isfinite(embeddings).all():
        raise ValueError("atomic block embeddings must be finite")
    matrix = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-4):
        raise ValueError("atomic block embeddings must be normalized")
    return matrix


def _context_centroid(
    blocks: Sequence[Passage],
    embeddings: np.ndarray,
    start_seconds: float,
    end_seconds: float,
) -> np.ndarray | None:
    weighted = np.zeros(embeddings.shape[1], dtype=np.float64)
    total_weight = 0.0
    for block, vector in zip(blocks, embeddings, strict=True):
        weight = max(
            0.0,
            min(block.end_seconds, end_seconds) - max(block.start_seconds, start_seconds),
        )
        if weight:
            weighted += weight * vector
            total_weight += weight
    if not total_weight:
        return None
    norm = float(np.linalg.norm(weighted))
    return None if norm <= 1e-12 else weighted / norm


def score_semantic_boundaries(
    blocks: Sequence[Passage],
    embeddings: np.ndarray,
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> tuple[SemanticBoundary, ...]:
    """Score every adjacent block boundary by left/right contextual cosine distance."""
    items = _validate_blocks(blocks)
    matrix = _validate_embeddings(items, embeddings)
    boundaries = []
    for index in range(1, len(items)):
        timestamp = items[index].start_seconds
        left = _context_centroid(
            items,
            matrix,
            timestamp - config.context_seconds,
            timestamp,
        )
        right = _context_centroid(
            items,
            matrix,
            timestamp,
            timestamp + config.context_seconds,
        )
        score = 0.0 if left is None or right is None else max(0.0, 1.0 - float(left @ right))
        boundaries.append(
            SemanticBoundary(
                episode_id=items[0].episode_id,
                timestamp_seconds=timestamp,
                left_block_id=items[index - 1].passage_id,
                right_block_id=items[index].passage_id,
                score=score,
            )
        )
    return tuple(boundaries)


def _validate_boundaries(
    blocks: Sequence[Passage], boundaries: Sequence[SemanticBoundary]
) -> tuple[SemanticBoundary, ...]:
    diagnostics = tuple(boundaries)
    if len(diagnostics) != max(0, len(blocks) - 1):
        raise ValueError("boundaries must describe every adjacent atomic block pair")
    for index, boundary in enumerate(diagnostics):
        left = blocks[index]
        right = blocks[index + 1]
        if (
            boundary.episode_id != left.episode_id
            or boundary.left_block_id != left.passage_id
            or boundary.right_block_id != right.passage_id
            or boundary.timestamp_seconds != right.start_seconds
        ):
            raise ValueError("boundaries must match atomic block IDs and order")
    return diagnostics


def _containing_interval(
    timestamp: float, cuts: set[float], start: float, end: float
) -> tuple[float, float]:
    ordered = [start, *sorted(cuts), end]
    for left, right in zip(ordered, ordered[1:]):
        if left < timestamp < right:
            return left, right
    return timestamp, timestamp


def select_semantic_boundaries(
    blocks: Sequence[Passage],
    boundaries: Sequence[SemanticBoundary],
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> tuple[SemanticBoundary, ...]:
    """Select semantic peaks, then add deterministic cuts required by maximum duration."""
    items = _validate_blocks(blocks)
    diagnostics = _validate_boundaries(items, boundaries)
    if not diagnostics:
        return ()
    scores = np.array([item.score for item in diagnostics], dtype=np.float64)
    threshold = float(np.quantile(scores, config.boundary_quantile))
    candidates = []
    for index, boundary in enumerate(diagnostics):
        previous = scores[index - 1] if index else -np.inf
        following = scores[index + 1] if index + 1 < len(scores) else -np.inf
        if (
            boundary.score > 1e-12
            and boundary.score >= threshold
            and boundary.score > previous
            and boundary.score >= following
        ):
            candidates.append(boundary)

    episode_start = items[0].start_seconds
    episode_end = items[-1].end_seconds
    selected: set[float] = set()
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.timestamp_seconds)):
        left, right = _containing_interval(
            candidate.timestamp_seconds,
            selected,
            episode_start,
            episode_end,
        )
        if (
            candidate.timestamp_seconds - left >= config.min_segment_seconds
            and right - candidate.timestamp_seconds >= config.min_segment_seconds
        ):
            selected.add(candidate.timestamp_seconds)

    forced: set[float] = set()
    while True:
        ordered = [episode_start, *sorted(selected), episode_end]
        oversized = next(
            (
                (left, right)
                for left, right in zip(ordered, ordered[1:])
                if right - left > config.max_segment_seconds
            ),
            None,
        )
        if oversized is None:
            break
        left, right = oversized
        limit = left + config.max_segment_seconds
        eligible = [
            boundary
            for boundary in diagnostics
            if (
                left + config.min_segment_seconds <= boundary.timestamp_seconds <= limit
                and right - boundary.timestamp_seconds >= config.min_segment_seconds
                and boundary.timestamp_seconds not in selected
            )
        ]
        if not eligible:
            break
        positive = [boundary for boundary in eligible if boundary.score > 1e-12]
        if positive:
            chosen = min(positive, key=lambda item: (-item.score, item.timestamp_seconds))
        else:
            chosen = max(eligible, key=lambda item: item.timestamp_seconds)
        selected.add(chosen.timestamp_seconds)
        forced.add(chosen.timestamp_seconds)

    return tuple(
        replace(
            boundary,
            selected=boundary.timestamp_seconds in selected,
            forced=boundary.timestamp_seconds in forced,
        )
        for boundary in diagnostics
    )


def segment_episode_semantically(
    episode: CanonicalEpisode,
    blocks: Sequence[Passage],
    embeddings: np.ndarray,
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> SemanticSegmentationResult:
    """Select boundaries and rebuild final passages from the episode's original cues."""
    items = _validate_blocks(blocks)
    if items[0].episode_id != episode.episode_id:
        raise ValueError("atomic blocks must match the requested episode")
    matrix = _validate_embeddings(items, embeddings)
    boundaries = select_semantic_boundaries(
        items,
        score_semantic_boundaries(items, matrix, config),
        config,
    )
    cuts = {item.timestamp_seconds for item in boundaries if item.selected}
    legal_cuts = {cue.start_seconds for cue in episode.cues[1:]}
    if not cuts.issubset(legal_cuts):
        raise ValueError("selected boundaries must align to original cue boundaries")

    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    for index, cue in enumerate(episode.cues):
        if current and cue.start_seconds in cuts:
            groups.append(tuple(current))
            current = []
        current.append(index)
    if current:
        groups.append(tuple(current))
    return SemanticSegmentationResult(
        passages=tuple(_passage(episode, indices) for indices in groups),
        boundaries=boundaries,
    )


def segment_episodes_semantically(
    episodes: Sequence[CanonicalEpisode],
    blocks: Sequence[Passage],
    embedding_result: EmbeddingResult,
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> SemanticSegmentationResult:
    """Segment multiple episodes without changing block or embedding order."""
    episode_items = tuple(episodes)
    block_items = tuple(blocks)
    if not episode_items:
        raise ValueError("episodes must be nonempty")
    expected_ids = tuple(item.passage_id for item in block_items)
    if embedding_result.passage_ids != expected_ids:
        raise ValueError("embedding passage IDs and order must match atomic blocks")

    passages: list[Passage] = []
    boundaries: list[SemanticBoundary] = []
    offset = 0
    for episode in episode_items:
        start = offset
        while offset < len(block_items) and block_items[offset].episode_id == episode.episode_id:
            offset += 1
        if offset == start:
            raise ValueError("atomic block episode order must match episodes")
        result = segment_episode_semantically(
            episode,
            block_items[start:offset],
            embedding_result.embeddings[start:offset],
            config,
        )
        passages.extend(result.passages)
        boundaries.extend(result.boundaries)
    if offset != len(block_items):
        raise ValueError("atomic block episode order must match episodes")
    return SemanticSegmentationResult(tuple(passages), tuple(boundaries))
