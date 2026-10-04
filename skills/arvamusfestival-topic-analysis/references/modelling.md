# Fit or repeat the detailed topic models

## Existing implementation

`notebooks/compare_segmentation_modes.ipynb` owns the sweep and full exports.
Core corpus, cache, segmentation and baseline modelling operations live in
`src/arvamusfestivali_transcripts/topic_analysis/`. Read the notebook configuration,
`FINE_CANDIDATES`, `build_fine_model` and full-export cell before running it.

Install the locked environment from the project root:

```bash
uv sync --group dev --group topic-analysis
uv run --group topic-analysis jupyter lab notebooks/compare_segmentation_modes.ipynb
```

For unattended execution, use nbclient with a copied notebook whose configuration
cell has the requested settings. Save executed output separately; keep the source
notebook's outputs clear. Do not execute the whole notebook just to update charts.
`TOPIC_ANALYSIS_PROJECT_ROOT` can identify a project when the kernel's working
directory differs. `TOPIC_ANALYSIS_RESULT_ROOT` selects a fresh full-export directory;
`RESULT_ROOT` otherwise defaults to `results/topic-detail`.

## Corpus and configurations

Load validated `data/transcripts/<year>/` archives and deduplicate by audio SHA-256.
Original run: 2026, 219 canonical talks. Counts describe that corpus, not acceptance
criteria for a future dataset. Select all intended talks rather than silently accepting
`EPISODE_COUNT=219` on a larger corpus. Keep `EXPLICIT_EPISODE_IDS` consistent with
post-deduplication canonical IDs. Preserve duplicate recording IDs in exports.

Fixed defaults: minimum/target/maximum 120/180/240 seconds, 400–800 words,
15-second overlap, 60-second tail threshold and 300-second maximum merged window.
Semantic defaults: atomic target/max 30/45 seconds, 60-second context,
90/600-second min/max segments, boundary quantile 0.85. BGE embeds atomic blocks,
contextual changes select boundaries; constraints can force cuts. Inspect boundary
charts when changing segmentation. Actual durations also depend on cue/word boundaries.

Local embedders: BGE-M3 (`BAAI/bge-m3`, 1,024 dimensions) and Qwen3-Embedding-0.6B.
Immutable revisions are defined in `embedders.py` and saved manifests. Cache schema
includes revision, instruction, dimensions and NFC/whitespace normalization; don't
reuse vectors from a different encoder or preprocessing configuration. Keep original
text in exports. GPU and batch size are environment choices, not scientific parameters.

Fine sweep: fixed/semantic × Qwen/BGE × four candidates × seeds 42/7 = 32 fits.
Candidates: eom-10-3, leaf-10-3, leaf-6-2, leaf-local-6-2. Selected saved models are
**BGE, leaf-local-6-2, seed 42** for both segmentations. Set `SELECTED_CANDIDATE` to
`leaf-local-6-2` when saving their pickle snapshots: the source notebook still defaults
to `leaf-6-2`, although it fits/exports tables for all candidates.

Selected parameters: UMAP 10 neighbors, 5 components, min_dist 0, cosine metric,
random initialization; HDBSCAN min_cluster_size 6, min_samples 2, Euclidean distance,
leaf selection, prediction data enabled. Multilingual BERTopic, `nr_topics=None`,
CountVectorizer preserving accents with unigrams/bigrams and min_df 1,
BM25-weighted c-TF-IDF and frequent-word reduction, top 15 words.
`calculate_probabilities=False` retains assigned-cluster strength rather than a dense
all-topic probability matrix. Use the notebook's maintained Estonian/ASR/English
stopword list. Don't change it silently when reproducing the experiment.

## Review and exports

Compare original-space cosine silhouette, member/centroid cosine, topic-size
concentration, outlier fraction and seed stability. Do not optimize topic count alone.
Review representatives and weak members for detail and coherence before choosing a
candidate. The selected run had 236 semantic topics and 346 fixed topics, excluding -1;
these are overlapping vocabularies, not unique subjects across models.

The full run directory contains fine-metrics.csv, seed-stability.csv and
`<mode>-<encoder>/<candidate>/seed-<seed>/` with topics.csv,
passage-assignments.csv, topic-review.csv, talk-topics.csv, outliers.csv and manifest.json.
Optional selected snapshots/hierarchy proposals are aids for later merges. Only load
trusted local pickle files. Published compact experiment tables omit full assignments;
they are insufficient to regenerate segment datasets on their own.

Capture revisions, package versions, parameters, ordered passage IDs and run identity.
New fits can renumber clusters even under the same seed, especially after input changes.
Use isolated run outputs and fresh naming checkpoints; preserve old results when
comparing. For a new corpus choose a run-specific export root and namespace or directory,
since the current exporter builds model_key from mode/encoder/candidate/seed alone.

## Checks

Use the locked environment and an absolute package path for subprocess notebook tests:

```bash
uv run --group topic-analysis pytest tests/topic_analysis/test_segmentation_notebook.py -q
```

The fixture exercises both segmentation modes and all 32 small fits using fake
embeddings. It verifies execution and metadata, not production semantic quality.
Production checks include unique native passage IDs, valid timestamps, matching
embedding rows, expected model/candidate identities and explicit outlier counts.
Exact floating-point results can vary by package/hardware; record those differences.
