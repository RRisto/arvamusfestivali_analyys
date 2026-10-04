# Export datasets, name topics and build charts

Run from the project root in the locked topic-analysis environment. The commands below
use `results/topic-detail` as the full run root and `data/topic-analysis/results` as
presentation output. Substitute an isolated presentation root or checkout for a new
run; scripts must refer to that same run throughout. Check each script's `--help` and
current source rather than assuming older interfaces.

## Required order for a new fit

1. Complete full modelling exports described in modelling.md. The two selected raw
   directories must contain passage-assignments.csv, topics.csv and manifest.json.
2. Export native segment datasets from those directories.
3. Copy the selected raw topic tables into the presentation fine-topics layout.
4. Export centroid maps from the dataset and matching embedding cache.
5. Generate names using a **fresh checkpoint directory** for this fit.
6. Apply the complete naming table; then build named charts.

The copying step is important: export_segment_dataset.py creates datasets but doesn't
create the compact fine-topics tables expected by apply_topic_names.py. The apply
script also requires pre-existing topic map coordinates.

```bash
uv run --group topic-analysis python scripts/export_segment_dataset.py \
  --input-root results/topic-detail \
  --output-root data/topic-analysis/results/segment-dataset

uv run --group topic-analysis python - <<'PY'
from pathlib import Path
import shutil
for mode in ('semantic', 'fixed'):
    suffix = Path(f'{mode}-bge/leaf-local-6-2/seed-42/topics.csv')
    destination = Path('data/topic-analysis/results/fine-topics') / suffix
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path('results/topic-detail') / suffix, destination)
PY

uv run --group topic-analysis python scripts/export_topic_maps.py \
  --dataset-root data/topic-analysis/results/segment-dataset \
  --cache-root data/topic-analysis/cache \
  --output-root data/topic-analysis/results/topic-maps

# Reads OPENAI_API_KEY; never put the credential in the command or repository.
uv run --group topic-analysis python scripts/name_topics_openai.py \
  --dataset data/topic-analysis/results/segment-dataset/all-segments.parquet \
  --output data/topic-analysis/results/topic-names --workers 4

uv run --group topic-analysis python scripts/apply_topic_names.py --root .

uv run --group topic-analysis python scripts/export_named_topic_charts.py \
  --dataset data/topic-analysis/results/segment-dataset/all-segments.parquet \
  --output data/topic-analysis/results/named-topic-charts
```

API naming is a paid stage. Use current user authorization; don't rerun it when names
already exist for the same exact fitted dataset. The script pins GPT-5.4 mini,
uses structured outputs, `reasoning.effort=none` and `store=False`; verify API support
if changing the model. A `--key-file` alternative exists for a user-provided file;
prefer the environment for ongoing use. Never log file contents or credentials.
`--limit 3` permits a small acceptance run, followed by the same command without the
limit. Checkpointing resumes completed topics. Resume only on the identical dataset:
existing checkpoint keys alone do not detect new fits that reuse topic numbers.

## Naming evidence and propagation

The naming sampler uses all members for topics up to 20 segments; larger topics use
18 strong members spread across talks plus 2 weak members. Prompts use full original
texts, talk titles, timestamps and keyword labels. Retained evidence includes segment
keys; results include names, summaries, subthemes, supporting example numbers,
self-assessed coherence/naming confidence, model, response IDs, prompt hashes and usage.
Original run: 582 names, 7,513 examples, 223 topics below ten examples, estimated
standard API cost $6.82. All confidence/coherence responses were high/coherent;
this is not evidence of perfect quality. Human review remains separate.

apply_topic_names.py requires exact one-to-one topic coverage and updates the selected
presentation topic tables' LLM columns, dataset topic_name and nested overlap names,
retains topic_name_keywords, and regenerates CSV/JSONL/Parquet plus the review page.
It asserts protected segment columns are unchanged. Original clustering labels remain.
Noise stays unassigned. It changes map names while preserving x/y and marker counts.
It does not refit models or update every raw sweep export.

For existing outputs needing chart changes only, use export_named_topic_charts.py.
For map regeneration, export_topic_maps.py reads names from the dataset. For rerunning
export_segment_dataset.py after naming, source selected topics.csv needs its matching
LLM_Name columns; otherwise it reconstructs keyword names. Reapply the saved names
before regenerating charts. Avoid repeatedly applying onto already-named map coordinates
without preserving the original topic_name_keywords column; the current apply script
was designed primarily for first application.

## Data and display semantics

all-segments.parquet holds both native segmentations. Also export separate Parquet,
CSV.gz and JSONL.gz files. CSV/Parquet encode overlap arrays as JSON strings; JSONL
contains actual arrays. Every row retains text, talk/episode identifiers, timestamps,
audio hash, duplicates and links. topic_key scopes a topic to model_key; segment_key
scopes native passage_id. New runs need their own namespaces/directories.

`confidence` is cluster membership strength, null for -1. Naming confidence describes
an unvalidated LLM judgement about a title. Future category-assignment probabilities
must be stored separately, not substituted for either existing field.

Cross-model overlap records describe the other model's native assignments and their
interval intersections. Overlap fractions may sum above one. Coverage charts use
interval unions per talk/topic; different topic coverages can overlap.

Maps average assigned original BGE embeddings, normalize each centroid and project
all 582 original topics jointly using UMAP (cosine, 15 neighbors, min_dist .15, seed42).
They exclude -1, use shared axis ranges and scale dot area by segment count.
Plot helpers prefer LLM_Name then Name while retaining numeric IDs. Separate baseline
passage projections are not geometrically aligned; centroid maps share one projection.

## Validation and publication

Verify counts and unique segment keys; full topic-name coverage; correct names in
other_model_overlaps; unchanged metadata, memberships, centroid vectors and existing
map coordinates on renaming. Inspect labels and timestamps in both chart modes.
Use relevant tests in tests/topic_analysis for segment_assignments, topic_map,
plotting and topic_naming. Documentation-only changes don't require model/API reruns.

Most generated results/cache files are ignored by Git. When publishing is authorized,
explicitly allowlist intended result files (git add -f for ignored artifacts), code and
manifests. Keep raw credentials, caches and large scratch outputs out. The combined
Parquet, compact topic tables, evidence archive, review HTML and charts are the current
shared artifacts; alternate data formats can remain local. Preserve complete local
manifests but use compact copies when committing manifests with large passage lists.
Resolve namespace/provenance fields against the actual exported data: some older
manifest descriptive fields still mention keyword labels after names were applied.
