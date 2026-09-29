# Topic-model Embedding Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable analysis package and Jupyter notebook that compare Qwen3, BGE-M3, and optional Gemini embeddings as inputs to BERTopic on six diverse, deduplicated festival transcripts.

**Architecture:** Keep the notebook as an orchestration and inspection surface. Put corpus preparation, deterministic chunking, model adapters, per-passage caching, BERTopic fitting, evaluation, and plotting in focused modules under `topic_analysis`; every model receives the same passages and controlled downstream configuration.

**Tech Stack:** Python 3.12, uv, pytest, NumPy, pandas, scikit-learn, Sentence Transformers, PyTorch, BERTopic, UMAP, HDBSCAN, Plotly, JupyterLab, Google Gen AI SDK.

**Spec:** `docs/superpowers/specs/2026-09-29-topic-model-comparison-design.md`

## Global Constraints

- Deduplicate recordings by `episode.audio_sha256`; retain every duplicate episode ID and source path as provenance.
- Default to six deterministically selected recordings, with explicit episode-ID override in the notebook.
- Preserve timestamps and audio links for every passage.
- Use `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3` dense output, and optional `gemini-embedding-2`.
- Read Gemini credentials only from `GEMINI_API_KEY`; skip Gemini cleanly when absent.
- Support CPU execution and conservative batches on the target RTX 2050 with 4 GB VRAM.
- Keep generated embeddings and results under ignored `data/topic-analysis/` directories.
- Do not download models or call external APIs in the normal test suite.
- Hold BERTopic, UMAP, and HDBSCAN settings constant for the controlled comparison.
- Automatic metrics support manual review; they do not declare a model winner.

---

## File map

**Create**

- `src/arvamusfestivali_transcripts/topic_analysis/__init__.py` — public analysis API.
- `src/arvamusfestivali_transcripts/topic_analysis/types.py` — immutable corpus, passage, embedding, topic-run, and review types.
- `src/arvamusfestivali_transcripts/topic_analysis/corpus.py` — validated archive loading, SHA deduplication, and deterministic diversity selection.
- `src/arvamusfestivali_transcripts/topic_analysis/chunking.py` — cue-boundary passage construction.
- `src/arvamusfestivali_transcripts/topic_analysis/cache.py` — cache identities and atomic per-passage vector storage.
- `src/arvamusfestivali_transcripts/topic_analysis/embedders.py` — common adapter contract and Qwen, BGE, Gemini implementations.
- `src/arvamusfestivali_transcripts/topic_analysis/modelling.py` — controlled BERTopic/UMAP/HDBSCAN fitting.
- `src/arvamusfestivali_transcripts/topic_analysis/evaluation.py` — metrics, cross-model comparisons, reviews, and JSON/CSV export.
- `src/arvamusfestivali_transcripts/topic_analysis/plotting.py` — Plotly figures for notebook inspection.
- `notebooks/compare_embedding_models.ipynb` — thin executable experiment notebook.
- `tests/topic_analysis/conftest.py` — compact archive and passage fixtures.
- `tests/topic_analysis/test_corpus.py`
- `tests/topic_analysis/test_chunking.py`
- `tests/topic_analysis/test_cache.py`
- `tests/topic_analysis/test_embedders.py`
- `tests/topic_analysis/test_modelling.py`
- `tests/topic_analysis/test_evaluation.py`
- `tests/topic_analysis/test_plotting.py`
- `tests/topic_analysis/test_notebook.py`

**Modify**

- `pyproject.toml` — add the optional `topic-analysis` dependency group.
- `uv.lock` — lock optional analysis dependencies.
- `.gitignore` — ignore generated analysis cache and result files.
- `README.md` — document installation and notebook execution.

---

### Task 1: Analysis dependency group and shared domain types

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.gitignore`
- Create: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Create: `src/arvamusfestivali_transcripts/topic_analysis/types.py`
- Create: `tests/topic_analysis/conftest.py`
- Create: `tests/topic_analysis/test_types.py`

**Interfaces:**
- Consumes: existing Python 3.12 package configuration.
- Produces: `Cue`, `CanonicalEpisode`, `Passage`, `EmbeddingResult`, `TopicRun`, `ManualTopicReview`, and `ChunkingConfig` dataclasses used by all later tasks.

- [ ] **Step 1: Write failing type-contract tests**

Create `tests/topic_analysis/test_types.py` with tests that instantiate a passage and reject invalid intervals:

```python
from dataclasses import FrozenInstanceError

import pytest

from arvamusfestivali_transcripts.topic_analysis.types import Passage


def test_passage_is_immutable_and_builds_timestamp_link() -> None:
    passage = Passage(
        passage_id="ep-1:000000120-000000240",
        episode_id="ep-1",
        duplicate_episode_ids=("ep-1", "ep-2"),
        title="Education debate",
        start_seconds=120.0,
        end_seconds=240.0,
        text="Teachers need enough time to prepare lessons.",
        audio_sha256="a" * 64,
        audio_url="https://example.test/audio.mp3",
        word_count=7,
        cue_count=1,
    )

    assert passage.timestamped_audio_url == "https://example.test/audio.mp3#t=120"
    with pytest.raises(FrozenInstanceError):
        passage.title = "Changed"  # type: ignore[misc]


def test_passage_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="passage interval"):
        Passage(
            passage_id="bad",
            episode_id="ep-1",
            duplicate_episode_ids=("ep-1",),
            title="Bad interval",
            start_seconds=10.0,
            end_seconds=10.0,
            text="text",
            audio_sha256="a" * 64,
            audio_url="https://example.test/audio.mp3",
            word_count=1,
            cue_count=1,
        )
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_types.py -q`

Expected: FAIL during collection with `ModuleNotFoundError` for `topic_analysis`.

- [ ] **Step 3: Add the optional dependency group and ignores**

Add this group to `pyproject.toml`, then run `uv lock`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.8",
]
topic-analysis = [
    "bertopic>=0.17,<0.18",
    "google-genai>=1,<2",
    "hdbscan>=0.8.40,<0.9",
    "ipykernel>=6,<8",
    "jupyterlab>=4,<5",
    "nbclient>=0.10,<1",
    "nbformat>=5,<6",
    "numpy>=2,<3",
    "pandas>=2.2,<3",
    "plotly>=6,<7",
    "scikit-learn>=1.6,<2",
    "sentence-transformers>=5,<6",
    "torch>=2.7,<3",
    "umap-learn>=0.5.7,<0.6",
]
```

Append to `.gitignore`:

```gitignore
data/topic-analysis/cache/
data/topic-analysis/results/
.ipynb_checkpoints/
```

- [ ] **Step 4: Implement the shared immutable types**

Create `types.py` with validated frozen dataclasses. Use these exact public fields:

```python
@dataclass(frozen=True, slots=True)
class Cue:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class CanonicalEpisode:
    episode_id: str
    duplicate_episode_ids: tuple[str, ...]
    title: str
    published_at: str
    audio_sha256: str
    duration_seconds: float
    audio_url: str
    cues: tuple[Cue, ...]
    source_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    min_seconds: float = 120.0
    target_seconds: float = 180.0
    max_seconds: float = 240.0
    min_words: int = 400
    max_words: int = 800
    overlap_seconds: float = 15.0
    merge_tail_seconds: float = 60.0
    max_merged_seconds: float = 300.0


@dataclass(frozen=True, slots=True)
class Passage:
    passage_id: str
    episode_id: str
    duplicate_episode_ids: tuple[str, ...]
    title: str
    start_seconds: float
    end_seconds: float
    text: str
    audio_sha256: str
    audio_url: str
    word_count: int
    cue_count: int

    @property
    def timestamped_audio_url(self) -> str:
        start = f"{self.start_seconds:g}"
        return f"{self.audio_url}#t={start}"


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    model_key: str
    model_id: str
    model_revision: str | None
    dimension: int
    passage_ids: tuple[str, ...]
    embeddings: np.ndarray
    cache_hits: int


@dataclass(frozen=True, slots=True)
class TopicRun:
    model_key: str
    topics: np.ndarray
    probabilities: np.ndarray | None
    reduced_embeddings: np.ndarray
    topic_info: pd.DataFrame
    representative_passages: Mapping[int, tuple[str, ...]]
    cluster_persistence: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ManualTopicReview:
    model_key: str
    topic_id: int
    verdict: Literal["coherent", "mixed", "duplicate", "unclear"]
    note: str = ""
```

Implement `__post_init__` checks for finite ordered intervals, nonempty text and IDs,
64-character lowercase hexadecimal SHA-256, positive counts, and compatible embedding
row counts.

- [ ] **Step 5: Run type tests and lint**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_types.py -q`

Expected: `2 passed`.

Run: `uv run --group dev ruff check src/arvamusfestivali_transcripts/topic_analysis/types.py tests/topic_analysis/test_types.py`

Expected: exit code 0.

- [ ] **Step 6: Commit the independently testable foundation**

```bash
git add pyproject.toml uv.lock .gitignore src/arvamusfestivali_transcripts/topic_analysis tests/topic_analysis/conftest.py tests/topic_analysis/test_types.py
git commit -m "feat: add topic analysis foundation"
```

---

### Task 2: Corpus loading, SHA deduplication, and diverse selection

**Files:**
- Create: `src/arvamusfestivali_transcripts/topic_analysis/corpus.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Modify: `tests/topic_analysis/conftest.py`
- Create: `tests/topic_analysis/test_corpus.py`

**Interfaces:**
- Consumes: `CanonicalEpisode` and `Cue` from Task 1; version-one archive JSON.
- Produces: `load_corpus(root: Path, year: int) -> tuple[CanonicalEpisode, ...]` and `select_diverse_episodes(episodes: Sequence[CanonicalEpisode], count: int = 6, explicit_ids: Sequence[str] | None = None) -> tuple[CanonicalEpisode, ...]`.

- [ ] **Step 1: Add archive-writing fixtures and failing corpus tests**

Add a fixture helper `write_archive(path, *, episode_id, sha256, title, published_at, cues)` to `conftest.py`. In `test_corpus.py`, cover canonical selection and diversity:

```python
def test_load_corpus_deduplicates_by_audio_hash_and_retains_provenance(
    tmp_path: Path, write_topic_archive: Callable[..., Path]
) -> None:
    write_topic_archive(tmp_path, episode_id="2", sha256="a" * 64, title="Same audio")
    write_topic_archive(tmp_path, episode_id="1", sha256="a" * 64, title="Same audio")
    write_topic_archive(tmp_path, episode_id="3", sha256="b" * 64, title="Other")

    episodes = load_corpus(tmp_path, 2026)

    assert [episode.episode_id for episode in episodes] == ["1", "3"]
    assert episodes[0].duplicate_episode_ids == ("1", "2")
    assert len(episodes[0].source_paths) == 2


def test_select_diverse_episodes_is_deterministic() -> None:
    episodes = make_episodes(
        "Teacher workload",
        "Early childhood education",
        "Preventive healthcare",
        "Artificial intelligence agents",
        "Climate policy",
        "Industrial investment",
        "Another teacher debate",
    )

    first = select_diverse_episodes(episodes, count=6)
    second = select_diverse_episodes(tuple(reversed(episodes)), count=6)

    assert tuple(item.episode_id for item in first) == tuple(item.episode_id for item in second)
    assert len({item.episode_id for item in first}) == 6


def test_explicit_selection_rejects_duplicate_audio_alias() -> None:
    episodes = make_episodes("Health", "Education")
    with pytest.raises(ValueError, match="unknown canonical episode IDs"):
        select_diverse_episodes(episodes, explicit_ids=["duplicate-alias"])
```

- [ ] **Step 2: Run corpus tests and confirm they fail**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_corpus.py -q`

Expected: FAIL because `load_corpus` and `select_diverse_episodes` do not exist.

- [ ] **Step 3: Implement archive parsing and deduplication**

In `corpus.py`, parse every `data/transcripts/<year>/*.json`, validate the exact fields
used by analysis, group by SHA, and choose the canonical record by
`(published_at, numeric episode ID, source path)`. Raise errors in this form:

```python
raise ValueError(f"invalid transcript archive {path}: {error}") from error
```

Sort canonical episodes by `(published_at, episode_id)` and sort provenance tuples.

- [ ] **Step 4: Implement deterministic diversity selection**

Use character n-gram TF-IDF over titles followed by deterministic farthest-first
selection:

```python
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
matrix = vectorizer.fit_transform(titles)
centroid_similarity = cosine_similarity(matrix, matrix.mean(axis=0)).ravel()
first_index = min(range(len(items)), key=lambda index: (centroid_similarity[index], items[index].episode_id))
selected = [first_index]
while len(selected) < count:
    remaining = [index for index in range(len(items)) if index not in selected]
    next_index = min(
        remaining,
        key=lambda index: (
            max(cosine_similarity(matrix[index], matrix[chosen])[0, 0] for chosen in selected),
            items[index].episode_id,
        ),
    )
    selected.append(next_index)
```

Explicit IDs preserve user order, require canonical IDs, and reject repetitions.

- [ ] **Step 5: Run corpus tests, existing schema tests, and lint**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_corpus.py tests/test_schema.py tests/test_archive.py -q`

Expected: all selected tests pass.

Run: `uv run --group dev ruff check src/arvamusfestivali_transcripts/topic_analysis/corpus.py tests/topic_analysis/test_corpus.py`

Expected: exit code 0.

- [ ] **Step 6: Commit corpus preparation**

```bash
git add src/arvamusfestivali_transcripts/topic_analysis tests/topic_analysis/conftest.py tests/topic_analysis/test_corpus.py
git commit -m "feat: prepare diverse transcript sample"
```

---

### Task 3: Timestamp-preserving passage construction

**Files:**
- Create: `src/arvamusfestivali_transcripts/topic_analysis/chunking.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Create: `tests/topic_analysis/test_chunking.py`

**Interfaces:**
- Consumes: `CanonicalEpisode`, `Cue`, and `ChunkingConfig`.
- Produces: `chunk_episode(episode: CanonicalEpisode, config: ChunkingConfig = ChunkingConfig()) -> tuple[Passage, ...]` and `chunk_episodes(episodes: Sequence[CanonicalEpisode], config: ChunkingConfig = ChunkingConfig()) -> tuple[Passage, ...]`.

- [ ] **Step 1: Write failing boundary, overlap, and tail tests**

```python
def test_chunk_episode_preserves_cue_boundaries_and_overlap() -> None:
    episode = episode_with_cues(count=12, seconds_per_cue=30, words_per_cue=100)
    config = ChunkingConfig(
        min_seconds=60,
        target_seconds=120,
        max_seconds=150,
        min_words=300,
        max_words=500,
        overlap_seconds=30,
        merge_tail_seconds=45,
        max_merged_seconds=180,
    )

    passages = chunk_episode(episode, config)

    assert passages[0].start_seconds == 0
    assert passages[0].end_seconds in {120, 150}
    assert passages[1].start_seconds == passages[0].end_seconds - 30
    assert all(item.start_seconds < item.end_seconds for item in passages)
    assert all(item.timestamped_audio_url.endswith(f"#t={item.start_seconds:g}") for item in passages)


def test_short_final_chunk_merges_without_exceeding_limit() -> None:
    episode = episode_with_cues(count=5, seconds_per_cue=30, words_per_cue=100)
    config = ChunkingConfig(
        min_seconds=60,
        target_seconds=90,
        max_seconds=120,
        min_words=200,
        max_words=500,
        overlap_seconds=0,
        merge_tail_seconds=60,
        max_merged_seconds=180,
    )

    passages = chunk_episode(episode, config)

    assert len(passages) == 1
    assert passages[0].end_seconds == 150
```

- [ ] **Step 2: Run chunking tests and confirm they fail**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_chunking.py -q`

Expected: FAIL because `chunk_episode` is missing.

- [ ] **Step 3: Implement deterministic cue accumulation**

Close a chunk when this predicate becomes true:

```python
should_close = (
    (duration >= config.target_seconds and word_count >= config.min_words)
    or duration >= config.max_seconds
    or word_count >= config.max_words
)
```

Build passage IDs using millisecond integers:

```python
passage_id = f"{episode.episode_id}:{round(start * 1000):012d}-{round(end * 1000):012d}"
```

After closing a chunk, reuse only complete trailing cues whose start is at or after
`end - overlap_seconds`. Merge a final passage shorter than `merge_tail_seconds` into
its predecessor only when the merged duration is at most `max_merged_seconds`.

- [ ] **Step 4: Run chunking tests and lint**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_chunking.py -q`

Expected: all tests pass.

Run: `uv run --group dev ruff check src/arvamusfestivali_transcripts/topic_analysis/chunking.py tests/topic_analysis/test_chunking.py`

Expected: exit code 0.

- [ ] **Step 5: Commit passage construction**

```bash
git add src/arvamusfestivali_transcripts/topic_analysis tests/topic_analysis/test_chunking.py
git commit -m "feat: chunk transcripts into timed passages"
```

---

### Task 4: Atomic per-passage embedding cache

**Files:**
- Create: `src/arvamusfestivali_transcripts/topic_analysis/cache.py`
- Create: `tests/topic_analysis/test_cache.py`

**Interfaces:**
- Consumes: model identity, instruction, dimension, chunking configuration, and `Passage`.
- Produces: `CacheIdentity`, `EmbeddingCache.get(identity, passage) -> np.ndarray | None`, `EmbeddingCache.put(identity, passage, vector) -> Path`, and `EmbeddingCache.assemble(identity, passages) -> tuple[np.ndarray | None, ...]`.

- [ ] **Step 1: Write failing identity and corruption tests**

```python
def test_cache_identity_changes_with_instruction_or_chunking() -> None:
    base = CacheIdentity(
        model_id="model",
        model_revision="rev",
        dimension=3,
        instruction="cluster Estonian debates",
        chunking=ChunkingConfig(),
        adapter_version=1,
    )
    changed = replace(base, instruction="search Estonian debates")
    assert base.digest != changed.digest


def test_cache_round_trip_and_corruption_rejection(tmp_path: Path, passage: Passage) -> None:
    cache = EmbeddingCache(tmp_path)
    identity = make_identity(dimension=3)
    vector = np.array([0.0, 0.6, 0.8], dtype=np.float32)

    path = cache.put(identity, passage, vector)
    assert np.allclose(cache.get(identity, passage), vector)

    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt embedding cache"):
        cache.get(identity, passage)
```

- [ ] **Step 2: Run cache tests and confirm they fail**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_cache.py -q`

Expected: FAIL because `EmbeddingCache` is missing.

- [ ] **Step 3: Implement stable identities and atomic writes**

Serialize identity data with sorted JSON keys. Hash normalized passage text together
with SHA, start, and end timestamps. Store each vector as:

```text
<cache-root>/<identity-digest>/<passage-digest>.npy
```

Write to a same-directory temporary path, flush and `os.fsync`, then replace the final
path. Validate loaded vectors for exact shape, finite values, and L2 norm within
`1e-4` of 1.0. Error messages must contain the cache path.

- [ ] **Step 4: Run cache tests and lint**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_cache.py -q`

Expected: all tests pass.

Run: `uv run --group dev ruff check src/arvamusfestivali_transcripts/topic_analysis/cache.py tests/topic_analysis/test_cache.py`

Expected: exit code 0.

- [ ] **Step 5: Commit cache support**

```bash
git add src/arvamusfestivali_transcripts/topic_analysis/cache.py tests/topic_analysis/test_cache.py
git commit -m "feat: cache passage embeddings safely"
```

---

### Task 5: Qwen, BGE, and optional Gemini embedding adapters

**Files:**
- Create: `src/arvamusfestivali_transcripts/topic_analysis/embedders.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Create: `tests/topic_analysis/test_embedders.py`

**Interfaces:**
- Consumes: `Passage`, `CacheIdentity`, and `EmbeddingCache`.
- Produces: `EmbeddingAdapter` protocol; `QwenEmbedder`, `BgeM3Embedder`, `GeminiEmbedder`, and test-only `DeterministicHashEmbedder`; `available_embedders(env: Mapping[str, str] = os.environ) -> Mapping[str, EmbeddingAdapter]`; and `embed_passages(adapter, passages, cache, batch_size) -> EmbeddingResult`.

- [ ] **Step 1: Write failing adapter orchestration tests with fakes**

```python
class FakeAdapter:
    key = "fake"
    model_id = "fake/model"
    model_revision = "revision-1"
    dimension = 3
    instruction = "cluster"

    def embed_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        rows = np.array([[len(text), 1.0, 2.0] for text in texts], dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_embed_passages_uses_cache_incrementally(tmp_path: Path, two_passages: tuple[Passage, ...]) -> None:
    cache = EmbeddingCache(tmp_path)
    adapter = FakeAdapter()

    first = embed_passages(adapter, two_passages, cache, batch_size=2)
    second = embed_passages(adapter, two_passages, cache, batch_size=2)

    assert first.cache_hits == 0
    assert second.cache_hits == 2
    assert np.allclose(first.embeddings, second.embeddings)


def test_available_embedders_skips_gemini_without_key() -> None:
    adapters = available_embedders(env={})
    assert set(adapters) == {"qwen", "bge"}


def test_gemini_prepares_clustering_instruction() -> None:
    assert GeminiEmbedder.prepare_text("Tere maailm") == (
        "task: clustering | query: Tere maailm"
    )


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    adapter = DeterministicHashEmbedder(dimension=32)
    first = adapter.embed_texts(["sama tekst"], batch_size=1)
    second = adapter.embed_texts(["sama tekst"], batch_size=1)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first[0]) == pytest.approx(1.0)
```

- [ ] **Step 2: Run adapter tests and confirm they fail**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_embedders.py -q`

Expected: FAIL because the adapter module is missing.

- [ ] **Step 3: Implement common cached orchestration**

Define the protocol:

```python
class EmbeddingAdapter(Protocol):
    key: str
    model_id: str
    model_revision: str | None
    dimension: int
    instruction: str

    def embed_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError
```

`embed_passages` must load cached rows, call the adapter only for misses, normalize
adapter output, persist each successful row immediately, and return rows in original
passage order. Reject incorrect shape or non-finite values with the model key in the
error.

Implement `DeterministicHashEmbedder` by deriving a fixed NumPy random seed from the
SHA-256 digest of each UTF-8 text, drawing `dimension` standard-normal values, and L2
normalizing the row. Its key and model ID are `fake` and `deterministic-hash`; it is
used only by automated notebook execution and is never returned by
`available_embedders`.

- [ ] **Step 4: Implement lazy local model adapters**

Do not import Torch or Sentence Transformers at module import time. Load them inside
the adapter on first use. Use these identities and defaults:

```python
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
BGE_MODEL_ID = "BAAI/bge-m3"
QWEN_INSTRUCTION = (
    "Represent this Estonian public-discussion passage for semantic topic clustering: "
)
```

`QwenEmbedder` calls `SentenceTransformer.encode` with the Qwen instruction, normalized
embeddings, configurable device, and configurable batch size. `BgeM3Embedder` calls the
same interface without a retrieval-query prefix and requests normalized dense output.
Wrap CUDA out-of-memory exceptions in a `RuntimeError` that recommends `device="cpu"`
or a smaller batch size.

- [ ] **Step 5: Implement the optional Gemini adapter**

Construct `google.genai.Client(api_key=key)` lazily. Embed one passage per API call so
every successful response can be cached immediately. Use model
`gemini-embedding-2`, `output_dimensionality=768`, and text prepared by
`GeminiEmbedder.prepare_text`. Validate that one finite vector is returned. Convert
quota and network exceptions into a `RuntimeError` naming Gemini and the passage ID;
previously cached passages remain valid.

- [ ] **Step 6: Run adapter tests and lint**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_embedders.py -q`

Expected: all tests pass without importing or downloading real model weights.

Run: `uv run --group dev ruff check src/arvamusfestivali_transcripts/topic_analysis/embedders.py tests/topic_analysis/test_embedders.py`

Expected: exit code 0.

- [ ] **Step 7: Commit embedding adapters**

```bash
git add src/arvamusfestivali_transcripts/topic_analysis tests/topic_analysis/test_embedders.py
git commit -m "feat: compare local and Gemini embeddings"
```

---

### Task 6: Controlled BERTopic fitting and evaluation metrics

**Files:**
- Create: `src/arvamusfestivali_transcripts/topic_analysis/modelling.py`
- Create: `src/arvamusfestivali_transcripts/topic_analysis/evaluation.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/types.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Create: `tests/topic_analysis/test_modelling.py`
- Create: `tests/topic_analysis/test_evaluation.py`

**Interfaces:**
- Consumes: ordered passages and `EmbeddingResult` objects.
- Produces: `TopicModelConfig`, `fit_topic_model`, `evaluate_run`, `compare_runs`, `build_review_rows`, and `export_experiment`.

- [ ] **Step 1: Write failing model-configuration tests**

```python
def test_fit_topic_model_preserves_passage_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBERTopic(topics=np.array([1, 0, -1]))
    monkeypatch.setattr(modelling, "_build_bertopic", lambda config: fake)
    passages = make_passages(3)
    result = make_embedding_result(3, dimension=4)

    run = fit_topic_model(passages, result, TopicModelConfig(random_state=42))

    assert run.model_key == result.model_key
    assert run.topics.tolist() == [1, 0, -1]
    assert fake.documents == [item.text for item in passages]


def test_controlled_builder_sets_seed_and_prediction_data(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(modelling, "_construct_models", recording_constructor(captured))

    modelling._build_bertopic(TopicModelConfig(random_state=17, min_cluster_size=8))

    assert captured["umap_random_state"] == 17
    assert captured["hdbscan_min_cluster_size"] == 8
    assert captured["hdbscan_prediction_data"] is True
```

- [ ] **Step 2: Write failing metric and export tests**

```python
def test_evaluate_run_handles_degenerate_assignments() -> None:
    run = make_topic_run(topics=np.array([-1, -1, -1]))
    metrics = evaluate_run(run, make_embedding_result(3, 4))

    assert metrics["topic_count"] == 0
    assert metrics["outlier_fraction"] == 1.0
    assert metrics["silhouette"] is None
    assert metrics["silhouette_unavailable_reason"] == "fewer than two non-outlier topics"


def test_compare_runs_reports_assignment_and_neighbour_agreement() -> None:
    qwen = make_run("qwen", topics=np.array([0, 0, 1, 1]))
    bge = make_run("bge", topics=np.array([2, 2, 3, 3]))
    comparison = compare_runs(
        {"qwen": qwen, "bge": bge},
        {"qwen": make_result("qwen"), "bge": make_result("bge")},
        neighbours=2,
    )
    assert comparison.loc[0, "adjusted_mutual_information"] == pytest.approx(1.0)
    assert 0.0 <= comparison.loc[0, "nearest_neighbour_overlap"] <= 1.0
```

- [ ] **Step 3: Run model and evaluation tests and confirm they fail**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_modelling.py tests/topic_analysis/test_evaluation.py -q`

Expected: FAIL because the modules are missing.

- [ ] **Step 4: Implement controlled BERTopic construction**

Define:

```python
@dataclass(frozen=True, slots=True)
class TopicModelConfig:
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    min_cluster_size: int = 10
    min_samples: int = 5
    top_n_words: int = 10
    random_state: int = 42
```

Build UMAP with cosine distance and HDBSCAN with Euclidean distance,
`cluster_selection_method="eom"`, and `prediction_data=True`. Build BERTopic with
precomputed embeddings, `calculate_probabilities=True`, and a `CountVectorizer` using
`ngram_range=(1, 3)` and `min_df=2`. Do not add a handcrafted stopword list.

Capture two reductions: five dimensions for clustering and a separate two-dimensional
projection for display, both with the same seed. Extract representative passage IDs
from BERTopic's representative documents by mapping text back to stable ordered IDs;
reject ambiguous duplicate text by using `(text, occurrence index)` keys.

- [ ] **Step 5: Implement per-run and cross-model evaluation**

Return dictionaries or DataFrames with exact named fields:

```text
model_key, passage_count, topic_count, outlier_count, outlier_fraction,
silhouette, silhouette_unavailable_reason, mean_cluster_persistence,
topic_diversity
```

Compute silhouette only when at least two non-outlier clusters remain. Compute topic
diversity as unique top terms divided by total top-term slots. `compare_runs` emits one
row per ordered model pair with adjusted mutual information and mean top-k nearest-
neighbour Jaccard overlap in original normalized embedding spaces.

- [ ] **Step 6: Implement review rows and reproducible export**

`build_review_rows` must include model, topic ID, size, representative passage IDs,
titles, timestamps, text, and blank verdict/note fields. `export_experiment` writes:

```text
<result-root>/<experiment-id>/manifest.json
<result-root>/<experiment-id>/passages.csv
<result-root>/<experiment-id>/topic-assignments.csv
<result-root>/<experiment-id>/metrics.csv
<result-root>/<experiment-id>/cross-model.csv
<result-root>/<experiment-id>/manual-review.csv
```

The manifest includes Git revision, UTC creation time, selected audio hashes, duplicate
mappings, chunk configuration, cache identities, model metadata, topic-model
configuration, random seeds, and installed package versions. Write each file through a
same-directory temporary path and replace it atomically.

- [ ] **Step 7: Run model/evaluation tests and lint**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_modelling.py tests/topic_analysis/test_evaluation.py -q`

Expected: all tests pass.

Run: `uv run --group dev ruff check src/arvamusfestivali_transcripts/topic_analysis/modelling.py src/arvamusfestivali_transcripts/topic_analysis/evaluation.py tests/topic_analysis/test_modelling.py tests/topic_analysis/test_evaluation.py`

Expected: exit code 0.

- [ ] **Step 8: Commit modelling and evaluation**

```bash
git add src/arvamusfestivali_transcripts/topic_analysis tests/topic_analysis/test_modelling.py tests/topic_analysis/test_evaluation.py
git commit -m "feat: evaluate controlled BERTopic runs"
```

---

### Task 7: Comparison visualizations

**Files:**
- Create: `src/arvamusfestivali_transcripts/topic_analysis/plotting.py`
- Modify: `src/arvamusfestivali_transcripts/topic_analysis/__init__.py`
- Create: `tests/topic_analysis/test_plotting.py`

**Interfaces:**
- Consumes: `Passage`, `TopicRun`, metrics, and cross-model correspondence DataFrames.
- Produces: `plot_semantic_map`, `plot_topic_sizes`, `plot_episode_topic_heatmap`, `plot_episode_timeline`, and `plot_topic_correspondence`, each returning `plotly.graph_objects.Figure`.

- [ ] **Step 1: Write failing figure-contract tests**

```python
@pytest.mark.parametrize(
    "builder",
    [
        plot_semantic_map,
        plot_topic_sizes,
        plot_episode_topic_heatmap,
        plot_episode_timeline,
    ],
)
def test_single_run_plot_contains_model_and_passage_provenance(builder) -> None:
    figure = builder(make_topic_run(), make_passages(6))
    payload = figure.to_plotly_json()
    assert "qwen" in payload["layout"]["title"]["text"].lower()
    serialized = json.dumps(payload)
    assert "#t=" in serialized
    assert "episode-" in serialized


def test_semantic_map_warns_that_separate_projections_are_not_aligned() -> None:
    figure = plot_semantic_map(make_topic_run(), make_passages(6))
    annotations = [item["text"] for item in figure.to_plotly_json()["layout"]["annotations"]]
    assert any("not geometrically aligned" in text for text in annotations)
```

- [ ] **Step 2: Run plotting tests and confirm they fail**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_plotting.py -q`

Expected: FAIL because plotting functions are missing.

- [ ] **Step 3: Implement accessible, provenance-rich Plotly figures**

Use topic IDs for stable categorical color mapping and include title, episode ID,
start/end time, excerpt, and timestamped audio link in `customdata`. Keep complete text
out of hover labels; truncate excerpts to 240 characters. Every figure must have a
descriptive title, axis labels where applicable, and a note when separate projections
cannot be compared by absolute position.

For timelines, draw passage intervals on a common minute axis. For the heatmap,
aggregate seconds per `(episode, topic)` and normalize each episode row only when the
notebook requests normalized display. For correspondence, use topic-overlap counts and
omit links with zero shared passages.

- [ ] **Step 4: Run plotting tests and lint**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_plotting.py -q`

Expected: all tests pass.

Run: `uv run --group dev ruff check src/arvamusfestivali_transcripts/topic_analysis/plotting.py tests/topic_analysis/test_plotting.py`

Expected: exit code 0.

- [ ] **Step 5: Commit visualizations**

```bash
git add src/arvamusfestivali_transcripts/topic_analysis tests/topic_analysis/test_plotting.py
git commit -m "feat: visualize embedding model comparisons"
```

---

### Task 8: Thin comparison notebook and user documentation

**Files:**
- Create: `notebooks/compare_embedding_models.ipynb`
- Create: `tests/topic_analysis/test_notebook.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: public functions exported by `topic_analysis` and optional
  `GEMINI_API_KEY`.
- Produces: an executable notebook that performs configuration, preparation,
  embedding, BERTopic fitting, evaluation, manual review, visualization, and export in
  order.

- [ ] **Step 1: Write the failing notebook-structure test**

```python
def test_comparison_notebook_is_thin_and_ordered() -> None:
    notebook = nbformat.read("notebooks/compare_embedding_models.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "load_corpus" in source
    assert "select_diverse_episodes" in source
    assert "chunk_episodes" in source
    assert "embed_passages" in source
    assert "fit_topic_model" in source
    assert "evaluate_run" in source
    assert "export_experiment" in source
    assert "GEMINI_API_KEY" in source
    assert "class QwenEmbedder" not in source
    assert "def chunk_episode" not in source


def test_notebook_has_no_saved_outputs_or_execution_counts() -> None:
    notebook = nbformat.read("notebooks/compare_embedding_models.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)
```

- [ ] **Step 2: Run the notebook tests and confirm they fail**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_notebook.py -q`

Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Create the notebook with explicit top-to-bottom sections**

Create markdown and code cells in this order:

```text
1. Purpose, privacy warning, and controlled-comparison explanation
2. Imports and display configuration
3. User configuration
4. Corpus loading, deduplication, and six-episode selection
5. Passage construction and diagnostics
6. Adapter availability, device, and cache configuration
7. Embedding generation with per-model exception isolation
8. Controlled BERTopic fitting
9. Automatic metric table
10. Semantic maps and topic-size charts
11. Representative and disputed passages
12. Episode heatmaps and timelines
13. Cross-model correspondence
14. Manual review DataFrame
15. Reproducible export
```

The configuration cell must expose these exact variables:

```python
YEAR = 2026
EPISODE_COUNT = 6
EXPLICIT_EPISODE_IDS: tuple[str, ...] = ()
MODEL_KEYS = ("qwen", "bge", "gemini")
DEVICE = "cpu"
BATCH_SIZES = {"qwen": 4, "bge": 4, "gemini": 1}
CACHE_ROOT = PROJECT_ROOT / "data" / "topic-analysis" / "cache"
RESULT_ROOT = PROJECT_ROOT / "data" / "topic-analysis" / "results"
CHUNKING = ChunkingConfig()
TOPIC_MODEL = TopicModelConfig(random_state=42)
```

If Gemini is unavailable, display `Gemini skipped: GEMINI_API_KEY is not set` and
continue. Catch failures per model, display the remediation, and retain successful
results from other models.

- [ ] **Step 4: Document installation and execution**

Add this workflow to `README.md`:

```powershell
uv sync --group dev --group topic-analysis
uv run --group topic-analysis jupyter lab notebooks/compare_embedding_models.ipynb
```

Document optional Gemini setup without printing the secret:

```powershell
$env:GEMINI_API_KEY = Read-Host -MaskInput "Gemini API key"
```

Explain generated cache/results paths, CPU-first defaults, how to choose explicit
episode IDs, and that enabling Gemini transmits selected passages to Google's API.

- [ ] **Step 5: Run notebook structure tests and static checks**

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_notebook.py -q`

Expected: `2 passed`.

Run: `uv run --group dev ruff check src tests`

Expected: exit code 0.

- [ ] **Step 6: Run the notebook with fake adapters in a test harness**

Add a notebook parameter `DRY_RUN_WITH_FAKE_EMBEDDINGS = False`. In the test, execute
the notebook through `nbclient` after replacing the configuration cell value with
`True`, using the six compact archives from `tests/topic_analysis/conftest.py` copied
to a temporary project root. Assert every cell completes and the export manifest is
created. This path must not import real local models or call Gemini.

Run: `uv run --group dev --group topic-analysis pytest tests/topic_analysis/test_notebook.py -q`

Expected: all notebook tests pass.

- [ ] **Step 7: Commit the notebook and documentation**

```bash
git add notebooks/compare_embedding_models.ipynb tests/topic_analysis/test_notebook.py README.md
git commit -m "feat: add embedding comparison notebook"
```

---

### Task 9: Full verification and opt-in real-model smoke test

**Files:**
- Modify only files implicated by verification failures.

**Interfaces:**
- Consumes: completed Tasks 1–8.
- Produces: verified analysis package, notebook, locked environment, and documented
  evidence from one small real-model run.

- [ ] **Step 1: Run the complete automated suite**

Run: `uv run --group dev --group topic-analysis pytest -q`

Expected: all tests pass and no test downloads a model or calls Gemini.

- [ ] **Step 2: Run lint and build**

Run: `uv run --group dev ruff check .`

Expected: exit code 0.

Run: `uv build`

Expected: source distribution and wheel build successfully.

- [ ] **Step 3: Run a bounded Qwen smoke test**

Use one canonical episode, construct at most eight passages, run Qwen on CPU with
batch size one, and verify:

```text
embedding rows = passage count
embedding dimension = 1024
all values finite
every row L2 norm within 1e-4 of 1.0
second run reports all rows as cache hits
```

Do not run BGE until the Qwen path succeeds. If the Qwen download requires network
approval, request it with the exact model-download command.

- [ ] **Step 4: Run a bounded BGE smoke test**

Use the same eight passages, CPU, and batch size one. Verify the same shape, finite,
normalization, and second-run cache assertions using BGE's reported dense dimension.

- [ ] **Step 5: Run Gemini only when the user has provided `GEMINI_API_KEY`**

If the variable is absent, verify the documented skip message and do not request the
secret. If present, embed exactly two passages, verify 768 normalized dimensions, and
verify the second run is served from cache without another API call.

- [ ] **Step 6: Inspect the controlled comparison output**

Run the notebook on six selected transcripts using cached real local embeddings.
Confirm that:

- every selected SHA is unique;
- all models receive identical passage IDs in identical order;
- metrics handle outliers without exceptions;
- representative passages link to timestamped audio;
- plots render without missing titles or provenance;
- export manifest records the current Git revision and configurations;
- separate semantic maps display the non-alignment warning.

- [ ] **Step 7: Commit verification-driven corrections**

If verification required changes, stage only those files and commit:

```bash
git add .gitignore README.md pyproject.toml uv.lock notebooks/compare_embedding_models.ipynb src/arvamusfestivali_transcripts/topic_analysis tests/topic_analysis
git commit -m "fix: complete topic comparison verification"
```

If no corrections were required, do not create an empty commit.
