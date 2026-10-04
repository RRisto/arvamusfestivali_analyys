# Detailed Jev labels for the current semantic segmentation

All4,132 current native semantic segments from219 talks have Jev1.13.0 results.
Each request chooses one primary detailed topic from the existing236-topic
vocabulary plus -1 for unassigned. Full probability distributions are retained.
The20 high-level categories are added by the saved dictionary, not separately
predicted by Jev. Original clustering labels, memberships, text, audio provenance
and native boundaries are unchanged.

Open notebooks/review_segment_topics.ipynb, restart and run all cells. It now
loads this full export automatically and defaults to Original/Semantic. Cards
show original model topics, detailed Jev topics, choice probabilities, Jev
confidence, top3 alternatives and dictionary high-level categories.

Files: assignments.parquet / assignments.csv contain all semantic predictions;
review-segments.parquet contains original rows with Jev sidecars (fixed rows have
no Jev result). responses.jsonl is the complete request-hash checkpoint, including the54
reused pilot responses with provenance. manifest.json pins the
vocabulary, source hashes, model, rubric hash and mapping. fine-topic-summary.csv
and high-level-summary.csv summarize coverage; index.html and
high-level-coverage.html provide the offline report.

jev_high_level_probability_mass sums fine probabilities in the chosen parent.
It is not a new calibrated confidence or direct high-level classifier result.
The selected parent is the parent of the primary fine-topic choice; it need not
be the parent with the largest aggregate mass.

```json
{
  "semantic_segments": 4132,
  "talks": 219,
  "jev_unassigned_segments": 389,
  "jev_unassigned_time_pct": 6.245250212769612,
  "fine_label_agreement_pct": 58.37366892545982,
  "parent_label_agreement_pct": 69.33688286544046,
  "interpretation": "Agreement with the original clustering is descriptive, not accuracy. Jev uses detailed 236-topic choice; parents are dictionary rollups, not direct high-level predictions."
}
```

Validation: all semantic keys covered once, expected237-option distributions
checked, choices match maximum probability, source columns unchanged, and all
parent mappings exist. The original source hash is unchanged. Agreement with
clustering is descriptive and is not accuracy. Seven review tests and lint passed.

The full notebook executed successfully in headless verification, loaded all
4,132 detailed labels and parent mappings, and rendered a document change.

## Results by festival year

[2025](2025/README.md): 114 talks, 2,154 semantic segments.
[2026](2026/README.md): 105 talks, 1,978 semantic segments.
Each folder contains unchanged detailed predictions and response checkpoints,
original native segment datasets, review data enriched with organizers, talk metadata,
fine/high-level summaries, disagreements, treemaps and bubble maps.
Bubble maps retain combined-corpus coordinates and size scales; counts are year-specific.
No classifier requests or topic refits are made. Export from the project root:

```bash
uv run --group topic-analysis python scripts/export_jev_year_results.py
```
