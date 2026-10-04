# Assessment of the 18-talk pilot

Keep the current full-corpus segmentation for now. The sentence-aware alternative
improves punctuation-aligned boundaries and gives more localized passages, but
this pilot does not support replacing the current topic-discovery input.

| Measure | Current segmentation | Sentence-aware, around 5-minute cap |
|---|---:|---:|
| Native segments | 324 | 642 |
| Median segment duration | 260.7 seconds | 124.4 seconds |
| Topics across seeds 42, 7, 19 | 24, 23, 25 | 41, 38, 42 |
| Unassigned recording time | 4.2–5.2% | 14.2–18.1% |
| Pairwise seed ARI, jointly assigned segments | 0.883–0.918 | 0.795–0.853 |
| Fraction assigned in both paired seeds | 88.3–88.9% | 70.6–72.3% |
| Internal cuts after sentence-ending punctuation | 36.6% | 98.2% |

The new variant doubles the number of boundaries. Its 297–321 total topic
changes include 148–177 noise transitions, compared with 79–81 total changes
and 32–40 noise transitions in the baseline. Adjacent assigned-topic changes
increase from 41–47 to 136–166, but they need interpretation: a larger number of
segments and more specific clusters can produce changes without improved accuracy.
See transition-diagnostics.csv for counts and rates normalized by boundaries.

Six matched examples were read in full by Codex, with qualitative observations
recorded in qualitative-examples.csv. They are not human validation or an accuracy
estimate. The shorter food-label and patient-advice passages are easier to retrieve.
The parenting narrative loses the preceding family background, the workshop loses
a role heading, and the mathematics explanation is split while introducing sets.
The teacher exercise ends with a punctuated ASR fragment. Punctuation is therefore
an imperfect proxy for a complete explanation. New topic labels in these examples
are relevant keyword clusters, but that alone does not establish coherent boundaries.

There are 54 distributed eight-minute windows with comparisons for each of the three
seeds. manual-review.csv remains blank for independent human ratings. The notebook
supports arbitrary shared time windows and full native segment text as well.

This purposeful sample covers varied subjects, shorter recordings and workshop
formats, but excludes a sparse-timestamp English transcript with a source cue longer
than six minutes. It does not establish performance for English or every ASR format.
Topic counts describe independently fitted 18-talk models, not the full-corpus236.
The new setting combines shorter duration, q70 semantic cuts and sentence snapping;
this experiment cannot attribute any effect to snapping alone.

A useful follow-up is to preserve the current semantic boundaries and snap only
nearby eligible cuts to sentence endings. That would isolate readability changes
without doubling the number of segments. It has not been applied to the full corpus.

Verification: 40 focused tests and lint checks passed. All six real fits passed
source, embedding, namespace and export checks. A headless notebook execution
loaded all fits and rendered the seed-change callback successfully.
