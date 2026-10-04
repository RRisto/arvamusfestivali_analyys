# Jev semantic sample

54 native semantic segments across three talks; fitted topic vocabulary:
`semantic:bge:leaf-local-6-2:42`, 236 topics plus unassigned.
Model: `jev-1.13.0`. Each request uses full original segment text and topic names
with summaries capped at 160 characters to fit the API context budget.

Open `notebooks/review_segment_topics.ipynb`, run its cells, select Semantic,
and choose document 0, 1 or 3. The notebook loads assignments.parquet automatically.
Static previews: review-document-0.html, review-document-1.html, review-document-3.html.

assignments.parquet and assignments.csv hold predictions keyed by segment_key.
review-segments.parquet retains every source column and adds Jev fields.
responses.jsonl retains API responses; manifest.json records source hashes,
configuration, model and usage. One earlier successful request used longer
descriptions and remains in the response log, but is excluded from final assignments.

Five segments selected unassigned; 26 of 54 primary choices match the original
cluster topic ID. Agreement is descriptive, not accuracy. Choice probabilities
are mutually exclusive primary-topic alternatives, not topic presence probabilities.
Jev confidence and probabilities are separate from HDBSCAN membership strength.
Human review is still required to assess this Estonian pilot.
