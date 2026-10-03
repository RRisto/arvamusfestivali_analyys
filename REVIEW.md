# Topic model review

The original notebook is preserved in the source project. The mounted project had no
`.git` metadata. A separate Git worktree was subsequently created from
`RRisto/arvamusfestivali_analyys` on branch `topic-detail-preserve-fine-topics`.
The working copy and experiment outputs remain under `/workspace/worktrees/topic-detail`.

## Findings from the recorded run

219 canonical recordings; fixed segmentation produced 6,545 passages and semantic segmentation
4,132. Semantic passages have a median of about four minutes and can reach ten minutes.
One cluster label cannot fully describe all subjects in such a passage.

| Segmentation | Embeddings | Topics | Outliers |
| --- | --- | ---: | ---: |
| fixed | Qwen | 137 | 27.1% |
| fixed | BGE-M3 | 178 | 18.1% |
| semantic | Qwen | 105 | 30.7% |
| semantic | BGE-M3 | 139 | 21.0% |

The later exploratory settings (`n_neighbors=30`, `min_cluster_size=15`, `min_samples=3`)
produced 97 topics, then `reduce_topics(nr_topics=40)` produced 39 non-outlier topics.
The largest reduced topic contained 2,235 of 4,132 passages. Automatic reduction loses the
fine information needed for this project. The stopword monkeypatch also changed subsequent
fits globally depending on notebook execution order.

## Changes

- Keep the baseline comparison. Replace the exploratory section with explicit BERTopic builders.
- Compare EOM and leaf selection with minimum clusters of 6 or 10, minimum samples of 2 or 3,
  and neighborhoods of 10 or 15. Compare two seeds on exactly the same cached passages.
- Keep all fine topics and noise labels. No fixed topic-count target or forced reassignment.
- Use curated Estonian/English stopwords and bigrams, preserving accents. Frequent-word
  reduction and BM25 weighting improve keyword extraction; they do not change embeddings.
- Report original-space cosine silhouette, member-to-centroid similarity, topic sizes,
  outlier coverage, seed agreement and the fraction used to measure that agreement.
- Export every candidate's assignments, talk profiles, review examples, weak members,
  outliers, topic tables and configuration manifest. Save selected model snapshots and
  hierarchy proposals for later controlled merging.
- Talk coverage uses unioned time intervals, so overlapping fixed windows do not inflate
  coverage within a topic. Different topics can overlap. Denominator is analyzed passage
  coverage, not necessarily the recording's complete duration.

`leaf-6-2` is a starting choice for detailed topics, not an automatically validated winner.
Review whether clusters distinguish concrete subjects, rather than speakers, language,
ASR errors or repeated fragments from adjacent windows. A topic from one talk can be useful.
Do not require multiple talks as a filter. Neither membership strength nor cosine scores
are calibrated confidence in the meaning of a topic.

## Run

Use the project's topic-analysis environment. Point the notebook at the existing corpus/cache
and direct exports into this working copy:

```bash
export TOPIC_ANALYSIS_PROJECT_ROOT=/workspace/vividface-storage/arvamusfestivali_analyys
export TOPIC_ANALYSIS_RESULT_ROOT=/workspace/worktrees/topic-detail/results/full
export PYTHONPATH=/workspace/worktrees/topic-detail/src
jupyter lab /workspace/worktrees/topic-detail/notebooks/compare_segmentation_modes.ipynb
```

The full sweep is 32 fits (4 passage/model pairs × 4 candidates × 2 seeds).
Change `FINE_PAIRS` to `(("semantic", "bge"),)` for a smaller first experiment.
The full archive setting remains 219; smaller fixtures automatically select the available talks.

After selecting clustering settings, run a separate shorter-segmentation comparison with
`SemanticSegmentationConfig(min_segment_seconds=60, max_segment_seconds=240,
boundary_quantile=0.70)`. This needs embeddings for the changed passages. Keeping these
experiments separate lets you see whether segmentation or clustering produced the improvement.

References: [HDBSCAN leaf selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering),
[BERTopic c-TF-IDF options](https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html).

## Validation

All 134 topic-analysis checks passed in the clean Git worktree, including full offline
notebook execution, exported inventories, and unioned interval coverage. Lint and
notebook schema/source compilation checks passed. An earlier run against the initial
mounted copy had three unrelated failures caused by the saved embedding notebook
and its archive setup; the clean Git worktree resolves those differences.

## Full-archive experiment

All 32 fits completed on 219 canonical recordings, using the existing cached embeddings.
No new model inference was needed. The following BGE results use seed 42:

| Segmentation | Candidate | Topics | Outliers | Median passages/topic | Original cosine silhouette |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed | eom-10-3 | 188 | 16.3% | 25 | 0.071 |
| fixed | leaf-10-3 | 207 | 21.9% | 23 | 0.072 |
| fixed | leaf-6-2 | 325 | 24.1% | 11 | 0.050 |
| fixed | leaf-local-6-2 | 346 | 23.6% | 11 | 0.050 |
| semantic | eom-10-3 | 152 | 20.5% | 18 | 0.076 |
| semantic | leaf-10-3 | 160 | 23.4% | 17 | 0.080 |
| semantic | leaf-6-2 | 219 | 23.8% | 14 | 0.073 |
| semantic | leaf-local-6-2 | 236 | 20.0% | 13 | 0.059 |

For fine detail, `leaf-local-6-2` is a promising alternative to the notebook's initial
`leaf-6-2` setting: semantic BGE produced 236 topics with 20.0% noise, against the
recorded baseline's 139 topics and 21.0% noise. Fixed BGE produced 346 topics with
23.6% noise, against 178 topics and 18.1% noise. This gains granularity with a coverage
tradeoff in fixed mode. Use semantic topics for contiguous passage coverage and fixed
topics to inspect finer fragments inside longer semantic passages.

Semantic BGE's local leaf setting has seed ARI 0.880 on shared inliers; the shared
inlier fraction is 71.4%, and outlier-status agreement is 84.0%. Seed agreement on
retained points does not prove that every topic is meaningful.

The original cosine silhouettes are around 0.05–0.11, substantially below the previously
reported UMAP-space silhouettes. These use different metrics/spaces; the gap cautions
against using the UMAP scores alone as evidence of semantic separation.

Inspecting top fixed-BGE keyword lists showed specific subjects such as health data,
workplace bullying, doctoral studies/knowledge transfer, biomethane/hydrogen filling
stations, minimum wages, museums and urban planning. Some topics still use speakers'
names; full manual coherence review remains necessary.

A concrete root cause of malformed words was BERTopic's default English preprocessing:
it deletes `ä`, `ö`, `õ`, `ü` before vectorization. Both builders now explicitly set
`language="multilingual"`. Descriptions and hierarchy proposals from this experiment
are refreshed under that setting without changing the fitted cluster assignments.
A regression check covers Estonian character preservation.

Compact metrics and seed comparisons are committed under `docs/topic-model-detail/`.
Full passage inventories, editable review sheets and talk profiles are stored at
`/workspace/worktrees/topic-detail/results/full/`. Model snapshots and hierarchy proposals
are retained for the initial `leaf-6-2` choice, seed 42, for each passage/model pair.
