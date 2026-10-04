---
name: arvamusfestival-topic-analysis
description: Repeat or extend this repository's detailed Arvamusfestival transcript topic modelling, including fixed and semantic segmentation, evidence-grounded topic names, reusable segment datasets and offline charts. Use for rerunning this analysis or regenerating its outputs.
---

# Arvamusfestival topic analysis

Locate the project by its `pyproject.toml`, `notebooks/compare_segmentation_modes.ipynb`
and `src/arvamusfestivali_transcripts/topic_analysis/`. Read its current README and
saved manifests before choosing parameters. Resolve commands from the project root,
not the skill directory; avoid depending on machine-specific temporary checkouts.

For fitting or changing the corpus, read [references/modelling.md](references/modelling.md).
For datasets, naming or charts, read [references/exports.md](references/exports.md).
Run only the stages needed by the user's request. Reuse embeddings when their cache
configuration matches; label or chart changes do not require reclustering.

Preserve these analysis invariants:

- Keep native fixed and semantic boundaries and all transcript provenance. Other-model
  overlap labels are assignments on other native segments, not direct predictions.
- Topic IDs belong to a fitted run. A new fit or new corpus requires a new naming
  checkpoint and namespace, even if numeric IDs repeat. Never transfer names by ID alone.
- Keep fine topics unless merging is requested. Topic `-1` stays unassigned.
- Preserve original keyword labels. Renaming must not change memberships, confidence,
  timestamps, text, audio metadata, centroids or existing plotted coordinates.
- Use genuine unique examples, up to 20 per topic; small topics use every member.
  Store full-text evidence. Don't fabricate examples to reach ten.
- HDBSCAN strength, embedding similarity, LLM naming self-assessment and any future
  Jev probability are different fields. The current all-high naming result is not an
  accuracy evaluation. Jev is not implemented in the current discovery workflow.

Use the repository's existing scripts rather than recreating exporters. Report output
locations, fitted-run identity, counts, checks and material limitations. API access,
commits and remote publication follow the user's current authorization; this skill
itself does not authorize external actions. Keep credentials out of outputs and Git.
