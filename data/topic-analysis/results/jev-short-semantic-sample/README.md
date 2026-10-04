# Shorter semantic pilot: comparison and recommendation

Same three talks and original 236-topic vocabulary plus unassigned; same Jev
version and question/config hash as the first pilot. Changed native boundaries
only: semantic minimum 60s, maximum 180s, boundary quantile 0.70. Atomic BGE
configuration remains 30/45s blocks and 60s context. All 453 matching cached
atomic vectors were reused; no clustering or topic renaming was performed.

| Measure | Original | Shorter |
| --- | ---: | ---: |
| Native segments | 54 | 137 |
| Median duration | 3m 58s | 1m 40s |
| Maximum duration | 9m 9s | 2m 51s |
| Jev unassigned segments | 5 | 28 |
| Duration assigned unassigned | 5.3% | 17.7% |
| Median primary-choice probability | 0.74 | 0.64 |

Probability and abstention statistics are descriptive, not accuracy. Shorter
segments can reveal genuine gaps hidden by a dominant subject in a long segment,
but can also lose necessary context. Raw segment counts are not comparable
accuracy denominators across different segmentations.

Read five previously mixed original intervals and all 16 resulting shorter
segments in full, plus one newly specific parent-collaboration segment in the
teacher talk. See length-comparison-review.csv for exact intervals and judgments.

Useful changes:
- AI talk 10:27–14:27: solo-business automation and AI writing become separate labels.
- AI talk 1:15:04–1:24:13: adoption strategy separates from school use and youth business.
- Teacher talk 38:55–42:15: curriculum responsibility separates from salary support.

Remaining problems:
- The teenage business-card story is split at 1:23:06, separating setup from its conclusion.
- Several cuts preserve complete ASR cues but end or begin inside a spoken sentence
  or even a word already fragmented by ASR. Cue alignment does not guarantee
  complete thoughts. The duration cap forces some of these cuts.
- Military recruitment and personal development still lack suitable topic definitions.
- A primary Choice is still restrictive even for 1–3 minute passages.

Recommendation: shorter segments are useful for topic timelines and review, but
this configuration is not uniformly better and is not ready to replace the whole
corpus. Improve sentence/turn-aware cut placement and retain neighbour context
for ambiguous snippets, then test across more varied talks. If that check holds,
export a separately namespaced full-corpus short dataset while preserving the
original. Reusing this vocabulary is classification; rediscovering clusters after
resegmentation requires a fresh fit and fresh topic names, with new topic IDs.

Review: run notebooks/review_segment_topics.ipynb, choose Original vs shorter,
then document 0 (teachers), 1 (AI), or 3 (military). Static comparison previews
are review-document-{0,1,3}.html. The notebook makes no API requests.

Validation: exact source cue counts and text preserved, each cue included once,
unique segment keys, all durations <=180s, 137 actual Jev predictions, separate
original temporal overlap references, same question hash as original pilot.

Checks completed: 25 segmentation/review tests pass, Ruff checks pass, and the
review notebook executes successfully with the shorter comparison activated.
The executed copy is review-notebook-executed.ipynb.
