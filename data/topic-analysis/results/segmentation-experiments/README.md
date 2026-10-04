# Small-sample segmentation experiments

Three recordings, kept separate from the full corpus: document 0 teacher support,
1 AI entrepreneurship, 3 military service. Open notebooks/review_segment_topics.ipynb,
restart the kernel and run all cells, choose Experiments and any Left/Right pair.
From min / To min selects a shared recording window (To min=0 means recording end).
All cards retain full native text, timestamps, audio links, Jev topic/probability
and top alternatives. The original model's assignments stay on its native segments.

## Variants and what they test

| Key | Cuts | Duration limit | Purpose / main tradeoff |
| --- | --- | --- | --- |
| original | Saved semantic run, q85 | 10 minutes | Existing reference; often mixes several subjects |
| short3 | Cue-aligned semantic, q70 | 3 minutes | More specific topics; frequent incomplete sentences |
| cue5 | Cue-aligned semantic, q70 | 5 minutes | Isolates length change; can still break sentences |
| sentence3 | Semantic, then nearby sentence-ending cue | 3 minutes + up to 20s | More complete sentences and fine topics; some context fragmentation |
| sentence5 | Semantic, then nearby sentence-ending cue | 5 minutes + up to 20s | More context and complete sentences; can combine multiple subjects |

The two sentence variants use a deterministic punctuation heuristic with common
abbreviation exclusions. For each original semantic cut, search ±20s for a nearby
sentence-ending cue, keep at least 45s between final cuts, and respect the soft
limit. If no eligible ending exists, retain the original cue cut and record that
fallback in boundaries.csv. No text is rewritten. This does not infer speakers,
restore punctuation, or guarantee that a question/answer or story stays together.

All new variants use identical cached BGE-M3 atomic blocks (30/45s), 60s context,
60s semantic minimum and q70. The original q85 run is a historical baseline, not
an otherwise controlled comparison. Sentence snapping moves cuts without changing
segment count. Identical source texts reuse saved Jev requests, while changed texts
are classified using the same model, question hash and 236-topic vocabulary + -1.
No embeddings were recomputed and no clustering/naming was rerun.

## Observed results

| variant | segments | median_seconds | max_seconds | sentence_ending_cut_pct | jev_unassigned_segments | jev_unassigned_time_pct | median_jev_choice_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 54 | 237.7 | 549.2 | 41.2 | 5 | 5.3 | 0.74 |
| short3 | 137 | 99.8 | 170.7 | 36.6 | 28 | 17.7 | 0.64 |
| cue5 | 99 | 133.8 | 294.8 | 39.6 | 21 | 15.5 | 0.65 |
| sentence3 | 137 | 102.2 | 188.9 | 95.5 | 26 | 16.6 | 0.66 |
| sentence5 | 99 | 136.5 | 300.1 | 97.9 | 19 | 14.4 | 0.67 |


Sentence-ending punctuation at internal cuts improves from 36.6%/39.6% in cue3/cue5
to 95.5%/97.9% in sentence3/sentence5. The sentence3 version has six fallback cuts;
sentence5 has two. These are punctuation metrics, not measured thought completeness.

Jev unassigned recording time falls from 17.7% to 16.6% for the 3-minute pair and
from 15.5% to 14.4% for the 5-minute pair. Median primary-choice probabilities also
rise slightly. Neither is validated topic accuracy; preserving context can help,
and a confident dominant-topic choice can also hide a secondary subject.

## Useful examples to inspect

- Document 1, 81–85 minutes: sentence snapping moves the cut from 1:23:06 to
  1:23:18, keeping the business-card anecdote's conclusion with its setup.
- Document 1, 78–85 minutes: sentence3 separates school use from a teenager's
  business story; sentence5 retains both in a roughly 4-minute segment. This
  directly illustrates fine topic resolution versus context retention.
- Document 1, 10–15 minutes: compare separate AI business/writing examples in the
  3-minute variants with their merged 5-minute equivalents.
- Document 3, 81–90 minutes: recruitment, training and school preparedness remain
  challenging because the topic vocabulary is narrow; better cuts alone do not
  fix missing or overly specific topic definitions.
- Document 0, 38–44 minutes: curriculum politics, teacher pay and local-government
  funding; inspect whether short segments express sufficient context.

A remaining failure is the brief AI-with-children/Estonian proverb passage around
1:23:18–1:24:13: Jev selects language-model Estonian quality at only 39%, although
that is not its substantive subject. A sentence-aligned fragment is not necessarily
an independently meaningful topic unit. Keep alternative probabilities visible.

My provisional preference is sentence5 as a comfortable review default, while
sentence3 is useful for more detailed timelines. No full-corpus replacement is
recommended solely from this small, selected sample.

## Files and review

index.html links to nine standalone side-by-side comparisons across the three talks.
comparison-summary.csv contains aggregate descriptive metrics. suggested-review.csv
lists useful time windows. manual-review.csv has blank boundary_quality, topic_fit,
lost_context and notes fields keyed to each native segment; repeated summaries do
not overwrite completed notes. Each new variant directory contains native segments,
full-text Jev review data, assignments, API responses, boundary diagnostics and
source/configuration manifests. Original outputs remain intact.

Compare short3 vs sentence3 to isolate boundary placement; cue5 vs sentence5 does
likewise at 5 minutes. Compare sentence3 vs sentence5 to inspect the length tradeoff.
Judge completeness, topic purity, preserved story context and appropriateness of
abstention. Do not choose the winner on probabilities alone.

## Reproduce

From the project root:

```bash
PYTHONPATH=src python scripts/short_semantic_pilot.py --variant cue5
PYTHONPATH=src python scripts/short_semantic_pilot.py --variant sentence3
PYTHONPATH=src python scripts/short_semantic_pilot.py --variant sentence5
python scripts/assign_topics_jev.py \
  --dataset data/topic-analysis/results/segmentation-experiments/sentence5/segments.parquet \
  --output data/topic-analysis/results/segmentation-experiments/sentence5 \
  --reuse-from data/topic-analysis/results/jev-semantic-sample/responses.jsonl \
    data/topic-analysis/results/jev-short-semantic-sample/responses.jsonl
PYTHONPATH=src python scripts/summarize_segmentation_experiments.py
```

Use the same assignment command with cue5/sentence3 paths for those variants.
API calls require TYPESAFE_API_KEY or the locally provided untitled.txt file.
Credentials are never included in output manifests or logs.

Validation completed: 37 focused tests and Ruff checks pass; the notebook
executes successfully with all five variants and the experiment view active.

## Independent topic-model refits

All five variants now also have independent BGE/BERTopic fits at seeds 42 and 7,
including a fair three-talk original-segmentation refit. See
[topic-model comparison](topic-models/README.md) and
[interactive charts](topic-models/index.html). In the review notebook use the
Fit seed and Labels controls; native topic timelines are shown above the cards.
The new cluster IDs/keyword labels are specific to each fit. Jev keeps the old
full-corpus vocabulary, so compare label meaning, not numeric ID equality.

## Broader 18-talk pilot

The [broader pilot](broader-pilot/index.html) compares the current native segmentation
with sentence-aware cuts around five minutes, using three seeds (42, 7, 19).
The review notebook includes an independent section with these six fits.
See broader-pilot/sample-talks.csv for purposeful selection and inputs.json for the
excluded sparse-timestamp English recording.
