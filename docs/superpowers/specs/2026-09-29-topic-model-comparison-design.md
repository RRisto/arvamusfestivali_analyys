# Topic-model embedding comparison design

## Objective

Build a reproducible notebook experiment that compares Qwen3, BGE-M3, and optional
Gemini embeddings as inputs to BERTopic on a small, thematically varied subset of the
2026 Arvamusfestival transcript archive.

The experiment must make it easy to inspect whether the resulting clusters are
substantively coherent in Estonian. Numerical metrics support that review but do not
select a winner automatically.

## Scope

The first version will:

- select six thematically diverse recordings automatically;
- deduplicate recordings by audio SHA-256 before selection;
- allow the notebook user to override the selection with episode IDs;
- split transcripts into timestamped passages;
- compare `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3` dense embeddings, and optional
  `gemini-embedding-2` embeddings;
- cache embeddings locally with enough provenance to prevent accidental cache reuse;
- fit and inspect one BERTopic model per embedding model;
- calculate automatic comparison metrics;
- provide interactive and tabular visual comparisons;
- export compact machine-readable results.

The first version will not:

- process the full archive by default;
- use BGE-M3 sparse or ColBERT-style multivector output;
- identify speakers;
- create a production search interface or vector database;
- require an LLM to run the core embedding comparison;
- claim that a single automatic metric establishes the best model.

## Architecture

Reusable code will live under:

```text
src/arvamusfestivali_transcripts/topic_analysis/
├── __init__.py
├── corpus.py
├── chunking.py
├── embedders.py
├── cache.py
├── modelling.py
├── evaluation.py
└── plotting.py
```

The experiment interface will be:

```text
notebooks/compare_embedding_models.ipynb
```

Generated data will use:

```text
data/topic-analysis/
├── cache/
└── results/
```

Both generated directories will be ignored by Git. Source code, tests, the notebook,
and documentation remain tracked.

## Corpus preparation

### Loading and deduplication

`corpus.py` will load validated transcript archives from
`data/transcripts/<year>/*.json`. It will group archives by
`episode.audio_sha256` and select one canonical archive for every unique recording.
The canonical record will retain the complete set of duplicate episode IDs and source
paths for provenance.

The loader will reject malformed records with an error naming the file and invalid
field. It will not silently omit malformed input.

### Automatic episode selection

The default selector will choose six unique recordings with varied content. Selection
will be deterministic for a fixed corpus and configuration. It will use episode-title
embeddings or another documented deterministic semantic-diversity method rather than
random sampling alone.

The notebook will display the selected titles, durations, IDs, hashes, and duplicate
counts before model execution. A configuration cell will accept explicit episode IDs
to replace the automatic selection.

## Passage construction

`chunking.py` will construct passages from timestamped transcript cues. The default
target is approximately two to four minutes or 400 to 800 words per passage, with
limited overlap between adjacent passages. Boundaries must coincide with cue
boundaries, and every passage must retain:

- canonical episode ID;
- duplicate episode IDs;
- episode title;
- passage ID;
- start and end seconds;
- text;
- source audio URL;
- timestamped audio URL;
- source audio SHA-256.

Chunking will be deterministic. Empty passages and passages containing only
whitespace are invalid. Very short final passages will be merged with their predecessor
when doing so does not exceed a documented upper bound.

Procedural speech such as introductions, microphone instructions, and applause will
not be deleted automatically in the first version. The notebook will make such
passages visible so their effect on clustering can be evaluated rather than hidden.

## Embedding adapters

`embedders.py` will expose one interface returning a two-dimensional, normalized
floating-point array with one row per passage, plus model metadata.

### Qwen3

- Model: `Qwen/Qwen3-Embedding-0.6B`
- Mode: dense, single-vector embedding
- Instruction: an English instruction describing semantic clustering of Estonian
  public-discussion passages
- Execution: local through Sentence Transformers

### BGE-M3

- Model: `BAAI/bge-m3`
- Mode: dense, single-vector embedding only
- Execution: local through a supported model interface
- Sparse and multivector outputs are explicitly outside the first comparison

### Gemini

- Model: `gemini-embedding-2`
- Mode: text embedding with the documented clustering instruction format
- Execution: Gemini API
- Authentication: `GEMINI_API_KEY` environment variable only
- Absence of the variable: mark Gemini unavailable and continue with local models

Gemini input will contain only the selected transcript passages and required task
instruction. The notebook will clearly indicate that enabling Gemini sends those
passages to Google's API.

### Device behavior

The target machine has an NVIDIA GeForce RTX 2050 with 4 GB VRAM. Local adapters must
support CPU execution and conservative batch sizes. Automatic device selection may use
CUDA when safe, but CUDA is not a requirement. An out-of-memory error must suggest a
smaller batch size or CPU execution rather than leave a cryptic stack trace as the only
guidance.

## Cache design

`cache.py` will cache embeddings separately for each model. A cache identity will
include:

- model identifier and relevant revision when available;
- embedding dimension;
- task instruction or task type;
- normalized passage text hashes in stable order;
- canonical audio hashes;
- chunking parameters;
- adapter version or schema version.

The cache will contain embeddings, passage identifiers, and metadata. Cache writes
must be atomic. A mismatch in any identity input creates a new cache entry instead of
reusing stale vectors.

Gemini responses will be cached incrementally so a partially completed API run can
resume without re-embedding successful passages.

## BERTopic modelling

`modelling.py` will accept passage text and precomputed embeddings. Each embedding
model will initially use identical BERTopic, dimensionality-reduction, and clustering
hyperparameters. Random seeds will be fixed wherever the underlying library supports
them.

The notebook will distinguish two experiments:

1. **Controlled comparison:** identical downstream parameters for every embedding
   model.
2. **Optional tuned comparison:** model-specific parameters chosen interactively after
   examining the controlled results.

Topic names in the core experiment will use deterministic BERTopic representations.
LLM-generated labels may be added as an optional later step, but the benchmark must run
without an LLM labelling service.

## Evaluation

`evaluation.py` will report, where mathematically applicable:

- topic count;
- outlier count and percentage;
- topic-size distribution;
- HDBSCAN cluster persistence;
- silhouette score calculated on non-outlier reduced embeddings;
- topic-word diversity;
- adjusted mutual information between model assignments;
- nearest-neighbour overlap between embedding spaces.

Metrics that are undefined because of too few clusters, too few non-outlier samples,
or another documented precondition will be reported as unavailable with a reason.

Manual inspection remains the primary evaluation. For every topic, the notebook will
show:

- representative passages;
- episode and timestamp provenance;
- passages near the cluster boundary;
- passages assigned differently by other embedding models;
- a review field with `coherent`, `mixed`, `duplicate`, or `unclear` choices.

Review annotations will be exportable but will not modify the source transcript
archives.

## Visualizations

`plotting.py` and the notebook will provide:

- comparable two-dimensional passage maps;
- topic-size bar charts;
- episode-by-topic heatmaps;
- timestamped topic timelines for each recording;
- cross-model topic correspondence;
- tabular representative-passage comparison.

Visualizations must state when coordinates come from separately fitted projections and
are therefore not directly geometrically comparable. Shared projections may be added
only when their interpretation is explicit.

## Notebook flow

The notebook will be executable from top to bottom:

1. Configure year, automatic or explicit episode selection, chunking, models, devices,
   and batch sizes.
2. Load and deduplicate the archive.
3. Display selected recordings and duplicate provenance.
4. Construct passages and inspect passage-length distributions and sample text.
5. Load or generate embeddings for each available model.
6. Fit the controlled BERTopic comparison.
7. Display automatic metrics.
8. Inspect representative and disputed passages.
9. Display maps, heatmaps, correspondence, and episode timelines.
10. Record manual review annotations.
11. Export results and experiment metadata.

The notebook must avoid hidden state: cells will either derive their inputs from prior
cells or load explicit cached artifacts.

## Dependencies

Heavy analysis dependencies will be placed in an optional `topic-analysis` dependency
group or extra. The base transcript-archiving installation will retain its existing
lightweight dependencies.

Expected analysis dependencies include BERTopic, Sentence Transformers, PyTorch,
UMAP, HDBSCAN, scikit-learn, pandas, Plotly, Jupyter, and Google's current Gen AI SDK.
Exact compatible version bounds will be selected and locked during implementation.

## Testing

Automated tests will not download models or call external APIs. They will cover:

- SHA-256 deduplication and canonical-record provenance;
- deterministic automatic selection;
- timestamp-preserving chunk construction and overlap;
- cache identity and stale-cache rejection;
- atomic and incremental cache behavior;
- adapter output-shape and normalization contracts using fakes;
- Gemini-unavailable behavior;
- BERTopic orchestration using synthetic embeddings or test doubles;
- metric behavior for ordinary and degenerate cluster assignments;
- exported result schemas.

A separate opt-in smoke check may exercise real local models or Gemini when explicitly
requested and configured.

## Error handling

User-facing failures will identify the failed stage, model, and remediation. Expected
cases include missing optional dependencies, missing API credentials, network or quota
errors, insufficient memory, invalid transcript files, empty episode selections,
invalid explicit episode IDs, and cache corruption.

Gemini failure will not invalidate completed Qwen or BGE results. Local-model failure
will likewise preserve other completed model outputs.

## Reproducibility and outputs

Every exported experiment will record:

- source Git revision;
- selected canonical episode IDs and hashes;
- duplicate mappings;
- passage-construction configuration;
- model identifiers and revisions when available;
- instructions and embedding dimensions;
- downstream hyperparameters and random seeds;
- package versions;
- automatic metrics;
- topic assignments and probabilities;
- manual review annotations when present.

Compact JSON and CSV outputs will be written under `data/topic-analysis/results/`.
Large embedding arrays remain in the ignored cache directory.

## Acceptance criteria

The design is implemented when a user can install the optional analysis dependencies,
open the notebook, run a controlled comparison on six automatically selected unique
recordings, compare Qwen and BGE results, optionally include Gemini through
`GEMINI_API_KEY`, inspect BERTopic topics and passages visually, record manual review,
and export reproducible results without changing any source transcript archive.
