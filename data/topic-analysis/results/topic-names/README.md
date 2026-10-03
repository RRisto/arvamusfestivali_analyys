# Evidence-grounded Estonian topic names

Open `topic-review.html` to inspect every topic, its keywords and 6–20 unique full-text segment examples with recording titles, timestamps and membership strengths. Topics smaller than ten segments use all available segments. No examples are fabricated.

`topic-names.csv` contains names, summaries, coherence assessments, naming confidence, subthemes, evidence references and API usage. `responses.jsonl` is a resumable checkpoint; `topic-evidence.jsonl.gz` retains the exact full-text prompts. The model is pinned; requests use structured outputs and `store=False`. Transcript text is treated as data.

Topic IDs, assignments, confidence, segment metadata, centroid embeddings and map coordinates are preserved. Keyword names remain in `topic_name_keywords`. Mixed/unclear topics warrant review before merging.
