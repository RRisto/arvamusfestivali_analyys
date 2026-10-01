# Semantic Transcript Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local deterministic semantic segmenter, boundary diagnostics, and a notebook that compares semantic and fixed segmentation with Qwen and BGE.

**Architecture:** Whole transcript cues are first accumulated into non-overlapping approximately 30-second atomic `Passage` objects. Cached BGE embeddings of those blocks drive contextual cosine-distance boundary scores; deterministic peak selection and duration constraints then merge original cues into semantic `Passage` objects that flow through the existing embedding and BERTopic pipeline.

**Tech Stack:** Python 3.12, NumPy, Plotly, sentence-transformers/BGE-M3, BERTopic, pytest, nbformat/nbclient, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-01-semantic-segmentation-design.md`

## Global Constraints

- Preserve `chunk_episode` and `chunk_episodes` behavior and the existing `compare_embedding_models.ipynb` file.
- Semantic boundary selection is local and deterministic; Gemini and other cloud APIs are excluded.
- BGE is the default boundary model; Qwen and BGE receive identical final passages within a segmentation mode.
- Atomic blocks never overlap and never split transcript cues.
- Defaults are 30-second atomic targets, 45-second atomic maxima, 60-second contexts, 90-second minimum segments, 600-second maximum segments, and a 0.85 boundary-score quantile.
- Existing content-addressed embeddings remain reusable and cache files are never deleted.
- Production code is written only after its focused test has failed for the expected missing behavior.

---

### Task 1: Semantic Contracts and Atomic Blocks

**Files:**
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/types.py`
- Create: `src/arvamusfestivali_transcripts/topic_analysis/semantic_segmentation.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Create: `tests/topic_analysis/test_semantic_segmentation.py`
- Modify: `tests/topic_analysis/test_types.py`

**Interfaces:**
- Consumes: `CanonicalEpisode`, `Cue`, `ChunkingConfig`, and `Passage` from `types.py`.
- Produces: `SemanticSegmentationConfig`, `SemanticBoundary`, `SemanticSegmentationResult`, `build_atomic_blocks`, and `build_atomic_blocks_many`.

- [ ] **Step 1: Write failing validation and atomic-block tests**

Add tests equivalent to:

```python
def test_semantic_config_rejects_incompatible_durations() -> None:
    with pytest.raises(ValueError, match="semantic segment durations"):
        SemanticSegmentationConfig(min_segment_seconds=700, max_segment_seconds=600)
    with pytest.raises(ValueError, match="boundary_quantile"):
        SemanticSegmentationConfig(boundary_quantile=1.1)


def test_atomic_blocks_are_cue_aligned_nonoverlapping_and_deterministic() -> None:
    episode = episode_with_cues(count=8, seconds_per_cue=10, words_per_cue=5)
    config = SemanticSegmentationConfig(atomic_target_seconds=30, atomic_max_seconds=45)
    blocks = build_atomic_blocks(episode, config)
    assert [(p.start_seconds, p.end_seconds) for p in blocks] == [
        (0, 30), (30, 60), (60, 80),
    ]
    assert [p.cue_count for p in blocks] == [3, 3, 2]
    assert "word0" in blocks[0].text and "word3" not in blocks[0].text
    assert build_atomic_blocks(episode, config) == blocks


def test_atomic_blocks_keep_an_indivisible_long_cue() -> None:
    episode = episode_with_custom_cues([(0, 120, "long cue"), (120, 130, "tail")])
    blocks = build_atomic_blocks(episode, SemanticSegmentationConfig())
    assert [(p.start_seconds, p.end_seconds) for p in blocks] == [(0, 120), (120, 130)]
```

Also assert that all three new public records are frozen and tuple fields reject lists.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/topic_analysis/test_types.py tests/topic_analysis/test_semantic_segmentation.py -q
```

Expected: collection fails because the semantic types and module do not exist.

- [ ] **Step 3: Implement contracts and atomic construction**

Implement frozen dataclasses with these public fields:

```python
@dataclass(frozen=True, slots=True)
class SemanticSegmentationConfig:
    atomic_target_seconds: float = 30.0
    atomic_max_seconds: float = 45.0
    context_seconds: float = 60.0
    min_segment_seconds: float = 90.0
    max_segment_seconds: float = 600.0
    boundary_quantile: float = 0.85

    @property
    def atomic_chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            min_seconds=self.atomic_target_seconds,
            target_seconds=self.atomic_target_seconds,
            max_seconds=self.atomic_max_seconds,
            min_words=1,
            max_words=1_000_000,
            overlap_seconds=0,
            merge_tail_seconds=0,
            max_merged_seconds=self.atomic_max_seconds,
        )


@dataclass(frozen=True, slots=True)
class SemanticBoundary:
    episode_id: str
    timestamp_seconds: float
    left_block_id: str
    right_block_id: str
    score: float
    selected: bool = False
    forced: bool = False


@dataclass(frozen=True, slots=True)
class SemanticSegmentationResult:
    passages: tuple[Passage, ...]
    boundaries: tuple[SemanticBoundary, ...]
```

Validate finite positive ordered durations, `0 <= boundary_quantile <= 1`, tuple fields, finite nonnegative boundary timestamps/scores, `selected or not forced`, unique passage IDs, and increasing non-overlapping passage intervals per episode.

Implement:

```python
def build_atomic_blocks(
    episode: CanonicalEpisode,
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> tuple[Passage, ...]: ...

def build_atomic_blocks_many(
    episodes: Sequence[CanonicalEpisode],
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> tuple[Passage, ...]: ...
```

Accumulate whole cues until the span reaches `atomic_target_seconds`; allow an indivisible cue to exceed `atomic_max_seconds`; close before adding a new cue when doing so would exceed `atomic_max_seconds`; never overlap blocks. Construct IDs with the existing millisecond timestamp format.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same focused command and expect all tests to pass.

- [ ] **Step 5: Commit**

```powershell
git add src/arvamusfestivali_transcripts/topic_analysis/types.py src/arvamusfestivali_transcripts/topic_analysis/semantic_segmentation.py src/arvamusfestivali_transcripts/topic_analysis/__init__.py tests/topic_analysis/test_types.py tests/topic_analysis/test_semantic_segmentation.py
git commit -m "feat: build semantic segmentation blocks"
```

---

### Task 2: Contextual Boundary Scoring and Selection

**Files:**
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/semantic_segmentation.py`
- Modify: `tests/topic_analysis/test_semantic_segmentation.py`

**Interfaces:**
- Consumes: ordered atomic `Passage` objects, a matching finite normalized `numpy.ndarray`, and `SemanticSegmentationConfig`.
- Produces: `score_semantic_boundaries(blocks, embeddings, config) -> tuple[SemanticBoundary, ...]` and `select_semantic_boundaries(blocks, boundaries, config) -> tuple[SemanticBoundary, ...]`.

- [ ] **Step 1: Write failing score and selection tests**

Use explicit normalized vectors so the expected semantic shift is visible:

```python
def test_contextual_score_finds_a_to_b_shift() -> None:
    blocks = six_thirty_second_blocks()
    vectors = np.array([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
    ], dtype=np.float32)
    config = SemanticSegmentationConfig(
        context_seconds=60, min_segment_seconds=60,
        max_segment_seconds=600, boundary_quantile=0.8,
    )
    scored = score_semantic_boundaries(blocks, vectors, config)
    selected = select_semantic_boundaries(blocks, scored, config)
    assert max(scored, key=lambda item: item.score).timestamp_seconds == 90
    assert [item.timestamp_seconds for item in selected if item.selected] == [90]
    assert not any(item.forced for item in selected)


def test_flat_embeddings_only_receive_required_maximum_cuts() -> None:
    blocks = twelve_sixty_second_blocks()
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (12, 1))
    config = SemanticSegmentationConfig(
        context_seconds=60, min_segment_seconds=90,
        max_segment_seconds=300, boundary_quantile=0.85,
    )
    selected = select_semantic_boundaries(
        blocks, score_semantic_boundaries(blocks, vectors, config), config
    )
    cuts = [item.timestamp_seconds for item in selected if item.selected]
    assert cuts == [300, 600]
    assert all(item.forced for item in selected if item.selected)
```

Add separate tests for minimum-duration rejection, deterministic equal-score ties, mismatched row counts, non-finite vectors, non-normalized vectors, mixed episodes, and an overlong indivisible block.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/topic_analysis/test_semantic_segmentation.py -q
```

Expected: import or attribute failures for the missing scoring functions.

- [ ] **Step 3: Implement contextual scoring**

For boundary index `i`, gather left blocks intersecting `[timestamp-context_seconds, timestamp]` and right blocks intersecting `[timestamp, timestamp+context_seconds]`. Weight each normalized embedding by the number of seconds of its block inside that context. Normalize each weighted centroid and use `max(0.0, 1.0 - dot(left, right))`.

Reject embeddings unless shape is `(len(blocks), dimension)`, values are finite, row norms are within `1e-4` of one, passage IDs are unique, all blocks belong to one episode, and intervals are increasing and non-overlapping.

- [ ] **Step 4: Implement deterministic selection and forced cuts**

Build local-maximum candidates using `score > 1e-12`, `score >= quantile`, and tie rule `score > previous and score >= next`. The positive-score requirement prevents a flat sequence from inventing a semantic cut. Consider candidates in `(-score, timestamp)` order. Accept a candidate only if both sides of its current containing interval satisfy `min_segment_seconds`.

Then inspect intervals in chronological order. While an interval exceeds `max_segment_seconds`, choose the highest-score unused legal block boundary at or before `interval_start + max_segment_seconds`; break score ties by earliest timestamp. If all legal scored boundaries are unavailable, use the latest atomic boundary at or before that limit. Mark a maximum-enforcement cut `forced=True`. Do not invent a cut inside an atomic block.

Return every scored boundary in timestamp order, updating `selected` and `forced` with `dataclasses.replace`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Task 2 test command and expect all semantic segmentation tests to pass.

- [ ] **Step 6: Commit**

```powershell
git add src/arvamusfestivali_transcripts/topic_analysis/semantic_segmentation.py tests/topic_analysis/test_semantic_segmentation.py
git commit -m "feat: detect semantic transcript boundaries"
```

---

### Task 3: Assemble Semantic Passages and Orchestrate Episodes

**Files:**
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/semantic_segmentation.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Modify: `tests/topic_analysis/test_semantic_segmentation.py`
- Modify: `tests/topic_analysis/test_cache.py`

**Interfaces:**
- Consumes: canonical episodes, atomic blocks, and the existing `EmbeddingResult` from the BGE atomic-block run.
- Produces: `segment_episode_semantically(...) -> SemanticSegmentationResult` and `segment_episodes_semantically(...) -> SemanticSegmentationResult`.

- [ ] **Step 1: Write failing passage-assembly and cache tests**

Add tests equivalent to:

```python
def test_segment_episode_merges_original_cues_at_selected_cut() -> None:
    episode = six_cue_topic_shift_episode()
    blocks = build_atomic_blocks(episode, semantic_config_for_test())
    result = segment_episode_semantically(episode, blocks, shift_vectors(), semantic_config_for_test())
    assert [(p.start_seconds, p.end_seconds) for p in result.passages] == [(0, 90), (90, 180)]
    assert result.passages[0].text == " ".join(cue.text for cue in episode.cues[:3])
    assert result.passages[1].text == " ".join(cue.text for cue in episode.cues[3:])
    assert all(p.audio_sha256 == episode.audio_sha256 for p in result.passages)
    assert all("#t=" in p.timestamped_audio_url for p in result.passages)


def test_many_episode_orchestrator_rejects_reordered_embedding_ids() -> None:
    episodes, blocks, embedding_result = two_episode_fixture()
    reordered = replace(
        embedding_result,
        passage_ids=tuple(reversed(embedding_result.passage_ids)),
        embeddings=embedding_result.embeddings[::-1],
    )
    with pytest.raises(ValueError, match="IDs and order"):
        segment_episodes_semantically(episodes, blocks, reordered, SemanticSegmentationConfig())
```

Extend the cache test to embed the same atomic blocks twice with `DeterministicHashEmbedder`, asserting the second result has all cache hits and identical vectors.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/topic_analysis/test_semantic_segmentation.py tests/topic_analysis/test_cache.py -q
```

Expected: failures because orchestration functions do not exist.

- [ ] **Step 3: Implement episode assembly**

Implement:

```python
def segment_episode_semantically(
    episode: CanonicalEpisode,
    blocks: Sequence[Passage],
    embeddings: np.ndarray,
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> SemanticSegmentationResult: ...

def segment_episodes_semantically(
    episodes: Sequence[CanonicalEpisode],
    blocks: Sequence[Passage],
    embedding_result: EmbeddingResult,
    config: SemanticSegmentationConfig = SemanticSegmentationConfig(),
) -> SemanticSegmentationResult: ...
```

Use selected timestamps to partition the original cue tuple. Every final `Passage` derives text, word count, cue count, title, duplicate IDs, audio hash, and audio URL from the episode and its exact cues. Reject a selected timestamp that is not an original cue boundary. Keep episode and passage order deterministic.

For the multi-episode function, require `embedding_result.passage_ids == tuple(block.passage_id for block in blocks)`, partition blocks and matrix rows by episode without reordering, call the one-episode function, and concatenate results.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 3 command and expect all focused tests to pass.

- [ ] **Step 5: Commit**

```powershell
git add src/arvamusfestivali_transcripts/topic_analysis/semantic_segmentation.py src/arvamusfestivali_transcripts/topic_analysis/__init__.py tests/topic_analysis/test_semantic_segmentation.py tests/topic_analysis/test_cache.py
git commit -m "feat: assemble semantic transcript passages"
```

---

### Task 4: Boundary Diagnostic Visualization

**Files:**
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/plotting.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Modify: `tests/topic_analysis/test_plotting.py`

**Interfaces:**
- Consumes: one `CanonicalEpisode`, its ordered atomic blocks, and `SemanticBoundary` diagnostics.
- Produces: `plot_semantic_boundaries(episode, blocks, boundaries) -> plotly.graph_objects.Figure`.

- [ ] **Step 1: Write failing figure tests**

```python
def test_boundary_plot_distinguishes_semantic_and_forced_cuts() -> None:
    episode, blocks, boundaries = boundary_plot_fixture()
    figure = plot_semantic_boundaries(episode, blocks, boundaries)
    assert isinstance(figure, go.Figure)
    assert "semantic" in figure.layout.title.text.lower()
    names = {trace.name for trace in figure.data}
    assert {"Boundary score", "Selected semantic cut", "Forced maximum cut"} <= names
    payload = json.dumps(figure.to_plotly_json())
    assert "left excerpt" in payload.lower()
    assert "#t=" in payload


def test_boundary_plot_rejects_other_episode_blocks() -> None:
    episode, blocks, boundaries = boundary_plot_fixture()
    foreign = replace(blocks[0], episode_id="other")
    with pytest.raises(ValueError, match="episode"):
        plot_semantic_boundaries(episode, (foreign, *blocks[1:]), boundaries)
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/topic_analysis/test_plotting.py -q`; expect the new import to fail.

- [ ] **Step 3: Implement the Plotly diagnostic**

Add one line-plus-marker trace for all boundary scores and two selected-cut traces split by `forced`. Use timestamp minutes on the x-axis, score on the y-axis, the existing white theme, and vertical dashed shapes for selected cuts. Hover custom data must contain left/right block excerpts and `episode.audio_url#t=<timestamp>`.

Validate block episode IDs, ordered unique block IDs, matching boundary left/right IDs, and boundary timestamps.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 4 command and expect all plotting tests to pass.

- [ ] **Step 5: Commit**

```powershell
git add src/arvamusfestivali_transcripts/topic_analysis/plotting.py src/arvamusfestivali_transcripts/topic_analysis/__init__.py tests/topic_analysis/test_plotting.py
git commit -m "feat: visualize semantic boundaries"
```

---

### Task 5: Fixed-versus-Semantic Comparison Notebook

**Files:**
- Create: `notebooks/compare_segmentation_modes.ipynb`
- Create: `tests/topic_analysis/test_segmentation_notebook.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all public semantic segmentation APIs, existing embedders/cache, BERTopic fitting, evaluation, and plotting APIs.
- Produces: an executable notebook with nested mappings keyed first by `fixed`/`semantic`, then by `qwen`/`bge`.

- [ ] **Step 1: Write the failing notebook structure test**

The test reads the notebook with `nbformat` and asserts:

```python
for public_name in (
    "build_atomic_blocks_many",
    "segment_episodes_semantically",
    "plot_semantic_boundaries",
    "chunk_episodes",
    "embed_passages",
    "fit_topic_model",
):
    assert public_name in source
assert 'SEGMENTATION_MODES = ("fixed", "semantic")' in source
assert 'MODEL_KEYS = ("qwen", "bge")' in source
assert "GeminiEmbedder" not in source
assert all(cell.execution_count is None for cell in code_cells)
assert all(cell.outputs == [] for cell in code_cells)
```

Add an offline execution test modelled on `test_notebook_executes_offline_and_exports_manifest`: copy six fixture archives, set `TOPIC_ANALYSIS_PROJECT_ROOT`, replace `DRY_RUN_WITH_FAKE_EMBEDDINGS = False` with `True`, execute via `NotebookClient`, and assert both segmentation modes produce passages, both local model keys produce runs, and a boundary figure is created.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/topic_analysis/test_segmentation_notebook.py -q
```

Expected: failure because the notebook does not exist.

- [ ] **Step 3: Create the thin notebook**

Create ordered cells for:

1. purpose/privacy/cost explanation;
2. imports and project-root resolution;
3. explicit configuration, including `EPISODE_COUNT=6`, empty explicit IDs, `SEGMENTATION_MODES=("fixed", "semantic")`, `MODEL_KEYS=("qwen", "bge")`, CPU device, batch sizes, cache/result roots, fixed and semantic configs, and fake-mode flag;
4. corpus load and deterministic episode selection;
5. fixed passages;
6. atomic blocks and cached BGE boundary embeddings;
7. semantic segmentation and duration/count comparison table;
8. boundary-score plot for the first selected episode;
9. passage-set embedding loops for both modes and both models;
10. BERTopic loops with mode-qualified model keys such as `fixed:qwen`;
11. metrics table and fixed-versus-semantic visual comparisons;
12. concise instructions for switching to all episodes and CUDA.

In fake mode, use `DeterministicHashEmbedder` for atomic and final embeddings. After `embed_passages`, use `dataclasses.replace(result, model_key=f"{mode}:{model_key}")` so the logical Qwen/BGE and fixed/semantic run identities remain distinct even though the synthetic adapter is shared. Choose a test-compatible topic configuration. Keep implementation logic in package modules rather than notebook functions.

- [ ] **Step 4: Add README usage**

Document the new notebook, explain that BGE boundary embeddings are cached separately, show the existing repository-visible Jupyter launch commands, and state that semantic segmentation does not use Gemini.

- [ ] **Step 5: Run notebook tests and verify GREEN**

Run the Task 5 test command and expect both structural and executable smoke tests to pass.

- [ ] **Step 6: Run complete verification**

Run:

```powershell
python -m pytest -q
python -m ruff check src tests
python -m pytest tests/topic_analysis/test_segmentation_notebook.py -q
git diff --check
```

Expected: zero test failures, zero Ruff findings, successful notebook execution, and no whitespace errors.

- [ ] **Step 7: Commit**

```powershell
git add notebooks/compare_segmentation_modes.ipynb tests/topic_analysis/test_segmentation_notebook.py README.md
git commit -m "feat: compare fixed and semantic segmentation"
```
