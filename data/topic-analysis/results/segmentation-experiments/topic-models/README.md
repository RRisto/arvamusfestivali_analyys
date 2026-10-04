# Refitted topic models for segmentation experiments

All five segmentations were fitted independently on the same three recordings.
Both seeds 42 and 7 use the repository's production `leaf-local-6-2` builder:
BGE-M3 pinned revision `5617a9f61b028005a4858fdac845db406aefb181`, UMAP 10 neighbours,
5 components, cosine metric, random init, min_dist=0; HDBSCAN leaf, minimum cluster
size 6, min_samples 2, Euclidean metric, prediction data enabled. The maintained
Estonian/ASR/English stopword list, multilingual BERTopic, 1–2 grams, BM25 c-TF-IDF,
frequent-word reduction, 15 keywords and no topic merging are reused verbatim
from `notebooks/compare_segmentation_modes.ipynb`.

Only these three talks are fitted, not the 219-talk corpus. The original-segment
refit is the fair sample baseline. Its 3–4 topics must not be compared directly
to the 236 topics discovered from the full corpus.

## Counts and dynamics

| Variant | Segments | Topics, seed 42 | Topics, seed 7 | Unassigned time, seed 42 | Topic changes, seed 42 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 54 | 4 | 3 | 3.0% | 14 |
| short3 | 137 | 9 | 9 | 18.0% | 75 |
| cue5 | 99 | 7 | 7 | 18.7% | 54 |
| sentence3 | 137 | 9 | 10 | 33.6% | 89 |
| sentence5 | 99 | 6 | 10 | 19.6% | 57 |

Topic changes count adjacent native labels that differ, including transitions
to/from -1. They describe timeline detail and can also increase through noisy
switching; they are not an accuracy score. Per-talk rates are in talk-dynamics.csv.

Observations:
- The original seed-7 refit produces exactly one topic per recording and zero
  within-talk changes. It mostly separates whole talks, not detailed subjects.
- The 3-minute cue version produces 9 topics under both seeds and much more
  within-talk detail, with about 18–21% of time unassigned.
- Sentence3 produces 9–10 topics, but about 29–34% unassigned time. Its sentence
  readability benefit does not automatically improve embedding density/clustering.
- Cue5 produces 7 topics under both seeds, although unassigned time changes from
  18.7% to 7.3%, so equal topic counts do not mean equivalent assignments.
- Sentence5 changes from 6 to 10 topics across seeds, suggesting substantial
  sensitivity on this small sample. Do not choose it purely from one fit.

Seed stability:

| variant | shared_inlier_fraction | outlier_status_agreement | inlier_ARI |
| --- | --- | --- | --- |
| original | 0.944 | 0.944 | 0.856 |
| short3 | 0.693 | 0.796 | 0.695 |
| cue5 | 0.768 | 0.818 | 0.572 |
| sentence3 | 0.518 | 0.672 | 0.687 |
| sentence5 | 0.657 | 0.808 | 0.587 |

Inlier ARI compares seed partitions on segments assigned by both seeds, independent
of numeric topic IDs. Shared-inlier fraction and noise-status agreement show how
much evidence that restricted ARI excludes. These are stability, not accuracy.

## Review in the notebook

Restart the kernel and run all cells in notebooks/review_segment_topics.ipynb.
Choose Experiments, document 0/1/3, Left/Right variants and fit seed 42 or 7.
Labels toggles between Topic model, Jev or both. The compact timelines show all
five variants on the same recording axis; the shared time window can zoom them.
Hover for keyword labels, exact intervals and membership strength.

New topic IDs are scoped to their independent fitted model. Cluster labels are
fresh keyword labels (not generated Estonian titles); full evidence is retained.
Jev still uses the old full-corpus named topic vocabulary. Comparing its numeric
IDs with these new cluster IDs is meaningless; compare texts and label meanings.
HDBSCAN membership strength is separate from Jev confidence/probability.
Original full-corpus assignments remain visible only as the historical baseline.

## Standalone results

index.html links topic-count bars, six compact timelines, eight Sankey charts and
18 full-text comparison pages. Colors and topic IDs are not aligned across fits.
Sankey links and topic-overlap-seconds.csv measure native interval intersections
in minutes/seconds. They reveal possible splits/merges in time coverage, not
semantic equivalence or direct predictions on another model's boundaries.

Each variant has saved embedding rows and their fingerprint, then seed-42 and
seed-7 directories containing the trained model.pkl, topics.csv, native review
Parquet, complete passage/assignment exports, metrics, topic evidence with all
full texts, talk dynamics and configuration/environment manifests. Seed 42 also
has topic-size and talk/topic coverage charts. Model pickles require the matching
saved Python/package environment when loaded. The existing embedding/model runs,
segment texts, timestamps, Jev responses and old topic names remain unchanged.

## Reproduce

From the project root in the established analysis environment:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/refit_segmentation_topic_models.py
PYTHONPATH=src python scripts/export_refit_topic_dynamics.py
```

Set TOPIC_PILOT_LOCAL_CACHE to relocate new per-passage cache files. Old matching
workspace cache vectors are reused; new vectors are computed locally using
real BGE embeddings on CUDA. No API classification or naming was rerun. Saved
per-variant embedding files resume on the same fingerprint; changed inputs or
model definitions fail rather than silently reusing stale vectors.

Validation: all ten native datasets retain protected source columns unchanged,
unique segment keys, exact fitted topic scopes, null membership strength for -1,
matching source/embedding order and configuration manifests. Exact historical cue
counts were reconstructed from source archives in the original baseline export.

The review notebook executed successfully with all ten refits and the topic-model
view. All 40 focused tests and lint checks passed.
