# Semantic Transcript Segmentation Design

## Goal

Add a local, deterministic semantic segmentation path alongside the existing fixed-duration chunker. Both paths must produce the existing `Passage` contract so the same embedding, BERTopic, evaluation, export, and plotting code can compare them.

The feature is successful when a researcher can inspect likely topic-change boundaries for an episode, compare fixed and semantic passage sets, and run Qwen and BGE over identical passages within each segmentation mode without sending transcript text to a cloud service.

## Scope

The implementation will:

- preserve the existing fixed chunker unchanged;
- build small, non-overlapping, cue-aligned atomic blocks;
- use cached BGE embeddings of those blocks to score semantic changes;
- select deterministic boundaries subject to minimum and maximum segment lengths;
- merge original cues into timestamped semantic `Passage` objects;
- expose boundary diagnostics and a Plotly boundary chart;
- add a notebook that compares fixed and semantic segmentation using Qwen and BGE;
- reuse the existing content-addressed embedding cache.

The implementation will not use an LLM for boundary selection, change BERTopic's clustering algorithm, replace the existing embedding-comparison notebook, or add a general-purpose change-point library.

## Why Atomic Blocks Instead of Individual Cues

The 2026 corpus contains 297,248 cues. Their median duration is 4.72 seconds and the 90th percentile is 7 seconds. Individual cues are too short to carry reliable topic meaning and would make boundary embedding unnecessarily expensive.

The semantic path will therefore group whole cues into atomic blocks targeting 30 seconds. Atomic blocks are analysis units only: final boundaries remain aligned to original cue boundaries, and the final passages contain the original transcript text.

## Public Contracts

`SemanticSegmentationConfig` will contain validated deterministic settings:

- `atomic_target_seconds=30`
- `atomic_max_seconds=45`
- `context_seconds=60`
- `min_segment_seconds=90`
- `max_segment_seconds=600`
- `boundary_quantile=0.85`

`SemanticBoundary` will describe one candidate boundary with episode ID, timestamp, semantic-change score, whether it was selected, and whether it was forced by the maximum-duration constraint.

The semantic module will expose focused operations:

- build atomic blocks from one episode or a sequence of episodes;
- score and select boundaries for one episode from ordered normalized block embeddings;
- build semantic passages for one episode or a sequence of episodes;
- return diagnostics in the same deterministic order as the source episodes.

All functions will validate passage/embedding count and order. A caller cannot accidentally apply vectors from one episode or model run to different blocks.

## Boundary Scoring

For each boundary between adjacent atomic blocks:

1. collect blocks whose time spans fall within `context_seconds` on the left and right;
2. compute a duration-weighted centroid for each side;
3. normalize both centroids;
4. calculate cosine distance, `1 - cosine_similarity`;
5. retain local maxima as candidates.

The episode-relative threshold is the configured quantile of finite boundary scores. A candidate must meet that threshold and be a local maximum. Ties are resolved by timestamp, giving deterministic results.

Candidate boundaries are considered from highest score to lowest. A boundary is accepted only when it preserves `min_segment_seconds` on both sides of its current containing segment. After semantic candidates are selected, any interval longer than `max_segment_seconds` is split at the highest-scoring eligible boundary. If no scored boundary can satisfy the maximum, the cue-aligned atomic boundary nearest the permitted limit is inserted and marked `forced=True`.

Episodes shorter than the minimum duration produce one passage. Flat or identical embeddings produce no semantic cuts, although maximum-duration enforcement can still create forced cuts.

## Cache Behavior

Atomic blocks use the existing `Passage`, `EmbeddingCache`, and `embed_passages` path. Their explicit atomic `ChunkingConfig` creates a distinct cache identity from the existing fixed passages.

Final semantic passages are also content-addressed by audio hash, timestamps, and canonicalized text. Changing a boundary setting therefore reuses embeddings for unchanged passages and computes only new text spans. Existing fixed-passage cache directories remain valid and untouched.

Boundary selection itself is inexpensive and will be recomputed from cached atomic embeddings. No separate boundary-result cache is required.

## Notebook Workflow

A new `notebooks/compare_segmentation_modes.ipynb` will leave the user's executed `compare_embedding_models.ipynb` untouched.

The notebook will:

1. load and deduplicate episodes using the existing corpus code;
2. select a small deterministic sample by default;
3. create fixed passages and semantic atomic blocks;
4. embed atomic blocks with BGE and select semantic boundaries;
5. show passage counts, duration distributions, boundary diagnostics, and an episode boundary plot;
6. embed both passage sets with Qwen and BGE;
7. fit BERTopic independently for each `(segmentation mode, embedding model)` pair;
8. compare fixed versus semantic results within each model and Qwen versus BGE within each mode;
9. retain timestamps and audio links for manual validation.

The default notebook run will be small. Configuration will make the episode count, device, batch sizes, cache path, and segmentation settings explicit. A deterministic fake-embedding mode will support an offline executable smoke test.

## Visualization

`plot_semantic_boundaries` will display semantic-change score against episode time. Selected semantic cuts and maximum-duration-forced cuts will use distinct markers. Hover text will include timestamp, score, selection reason, and neighbouring transcript excerpts when available.

This diagnostic is intentionally separate from the existing topic timeline: it evaluates whether segmentation boundaries are credible before BERTopic quality is considered.

## Errors and Edge Cases

- Invalid duration ordering, quantiles, or context sizes raise `ValueError` at configuration construction.
- Empty episode collections and empty block sequences are rejected with actionable errors.
- Embedding matrices must be finite, two-dimensional, normalized, ordered, and have one row per atomic block.
- Atomic blocks never split cues and never overlap.
- Extremely long single cues remain indivisible; the resulting forced segment may exceed the configured maximum and the diagnostic records that no legal cue boundary existed.
- Duplicate recordings continue to be removed by the existing SHA-based canonical corpus loader.
- Gemini is not used by semantic segmentation.

## Testing and Verification

Test-driven implementation will cover:

- cue-aligned non-overlapping atomic block construction;
- a synthetic A-to-B embedding shift producing the expected boundary;
- local-maximum and quantile filtering;
- minimum-segment enforcement;
- deterministic tie handling;
- forced maximum-duration cuts;
- short episodes and flat embeddings;
- mismatched block IDs, order, shapes, and non-finite vectors;
- construction of final text, timestamps, provenance, and audio links;
- boundary plotting;
- existing embedding-cache reuse;
- an executable notebook dry-run smoke test.

Completion requires the focused red/green tests, the full pytest suite, Ruff, and notebook execution with deterministic fake embeddings.

