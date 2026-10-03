# Detailed topic results

These are the actual topic tables and talk profiles for all 32 fitted experiments.
Start with [semantic BGE: 236 topics](semantic-bge/leaf-local-6-2/seed-42/topics.csv)
and its [talk profiles](semantic-bge/leaf-local-6-2/seed-42/talk-topics.csv).
For finer fixed-window fragments, inspect
[fixed BGE: 346 topics](fixed-bge/leaf-local-6-2/seed-42/topics.csv).

Each experiment contains `topics.csv` (keywords and passage counts), `talk-topics.csv`
(coverage by episode), and `manifest.json` (model and configuration metadata).
Topic IDs are specific to each experiment. Topic -1 means unassigned content.
Estonian letters are preserved in all topic descriptions.

Talk coverage is computed from unioned passage intervals. Fixed-window coverage of
separate topics can overlap. Review these fine topics before merging or naming a taxonomy.

Comparison metrics are under `docs/topic-model-detail/` at the repository root.
Full passage text, timestamps, audio links, outliers and editable review sheets remain
in `/workspace/worktrees/topic-detail/results/full/` in the analysis workspace.
This Git version omits repeated full transcript excerpts and binary model snapshots.
