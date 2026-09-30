"""Build timestamped passages without splitting transcript cues."""

from __future__ import annotations

from collections.abc import Sequence

from .types import CanonicalEpisode, ChunkingConfig, Passage


def chunk_episode(
    episode: CanonicalEpisode, config: ChunkingConfig = ChunkingConfig()
) -> tuple[Passage, ...]:
    """Accumulate whole cues into passages with deterministic cue-level overlap."""
    cues = episode.cues
    cue_words = tuple(len(cue.text.split()) for cue in cues)
    chunks: list[tuple[int, ...]] = []
    carried: tuple[int, ...] = ()
    next_index = 0

    while next_index < len(cues):
        indices = list(carried)
        word_count = sum(cue_words[index] for index in indices)

        # Always add a new cue before closing; carried cues alone are not a passage.
        while next_index < len(cues):
            indices.append(next_index)
            word_count += cue_words[next_index]
            next_index += 1
            duration = cues[indices[-1]].end_seconds - cues[indices[0]].start_seconds
            should_close = (
                (duration >= config.target_seconds and word_count >= config.min_words)
                or duration >= config.max_seconds
                or (duration >= config.min_seconds and word_count >= config.max_words)
            )
            if should_close:
                break

        chunks.append(tuple(indices))
        end = cues[indices[-1]].end_seconds
        overlap_start = end - config.overlap_seconds
        carried = tuple(index for index in indices if cues[index].start_seconds >= overlap_start)

    if len(chunks) > 1:
        tail = chunks[-1]
        tail_duration = cues[tail[-1]].end_seconds - cues[tail[0]].start_seconds
        merged_duration = cues[tail[-1]].end_seconds - cues[chunks[-2][0]].start_seconds
        if (
            tail_duration <= config.merge_tail_seconds
            and merged_duration <= config.max_merged_seconds
        ):
            chunks[-2] = tuple(dict.fromkeys((*chunks[-2], *tail)))
            chunks.pop()

    passages = []
    for indices in chunks:
        start = cues[indices[0]].start_seconds
        end = cues[indices[-1]].end_seconds
        passages.append(
            Passage(
                passage_id=(
                    f"{episode.episode_id}:{round(start * 1000):012d}-{round(end * 1000):012d}"
                ),
                episode_id=episode.episode_id,
                duplicate_episode_ids=episode.duplicate_episode_ids,
                title=episode.title,
                start_seconds=start,
                end_seconds=end,
                text=" ".join(cues[index].text for index in indices),
                audio_sha256=episode.audio_sha256,
                audio_url=episode.audio_url,
                word_count=sum(cue_words[index] for index in indices),
                cue_count=len(indices),
            )
        )
    return tuple(passages)


def chunk_episodes(
    episodes: Sequence[CanonicalEpisode], config: ChunkingConfig = ChunkingConfig()
) -> tuple[Passage, ...]:
    """Chunk episodes in caller-provided order."""
    return tuple(passage for episode in episodes for passage in chunk_episode(episode, config))
