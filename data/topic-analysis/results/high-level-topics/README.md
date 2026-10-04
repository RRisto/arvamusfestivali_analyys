# High-level categories for the current semantic model

This reviewable editorial grouping places all 236 existing fine topics under 20
parent categories. It preserves the current segmentation, fine topic IDs and
memberships. All 4,132 semantic segments have a parent or explicit `unassigned`:
3,305 belong to substantive categories and 827 remain unassigned.

Open [index.html](index.html) to expand each category and review its fine topics,
original summaries and segment counts. Edit category-definitions.json and rerun
scripts/export_high_level_topics.py to regenerate the mapping and dataset.
The source names, taxonomy and full dataset hashes are recorded in manifest.json.

- category-definitions.json: 20 stable category IDs, Estonian names, definitions
  and member fine-topic IDs; the editable taxonomy source.
- topic-to-high-level.json: dictionary from string fine-topic ID to category ID,
  including `"-1": "unassigned"`.
- topic-key-to-high-level.json: the same lookup using full model-scoped topic keys.
- topic-mapping.csv: every fine topic, its original name and summary, and parent.
- categories.csv: category definitions and coverage counts.
- semantic-segments.parquet: separate copy of the current semantic dataset with
  high_level_topic_id and high_level_topic_name; original columns unchanged.
- jev-high-level-criteria.json: 20 categories plus explicit unassigned, ready for
  a future direct high-level Jev choice question. No requests have been sent.

The numeric dictionary applies only to `semantic:bge:leaf-local-6-2:42`. Other
fits can reuse the same numbers for different topics; use the scoped dictionary
and check the vocabulary manifest when integrating.

```python
import json
from pathlib import Path

base = Path("data/topic-analysis/results/high-level-topics")
fine_to_high_level = json.loads((base / "topic-to-high-level.json").read_text())
# Jev uses this exact existing fine-topic vocabulary:
high_level_id = fine_to_high_level[str(jev_topic_id)]
```

To aggregate a Jev fine-topic probability distribution, add the probability of
every fine topic belonging to each category. This produces parent probability
mass, not a newly calibrated classifier confidence. The largest parent mass can
differ from the parent of Jev's largest individual fine-topic probability.

```python
parent_mass = {}
for fine_id, probability in jev_probabilities.items():
    parent = fine_to_high_level[str(fine_id)]
    parent_mass[parent] = parent_mass.get(parent, 0.0) + probability
```

Mapping an existing fine result is different from asking Jev to classify directly
into these 20 categories. Both are now easy to support, but their outputs should
be distinguished when compared.

Cross-domain conventions: teaching and AI assessment stay in education; medical
AI and health data stay in healthcare; AI energy consumption stays in climate;
AI-agent authorization and creative AI copyright stay in AI; data surveillance
stays in privacy; women's IKT employment stays in work. Youth protection and
family/relationship issues have separate categories. Topic-level grouping cannot
resolve all within-topic mixtures, so the expandable member list remains the
review source. No human approval or quantitative accuracy is implied.

Validation: all236 fine topics have exactly one parent, all4,132 native semantic
segments are covered, and source columns are unchanged. All54 existing Jev pilot
choices map correctly; their aggregated probability masses retain the original
total. jev-pilot-high-level.parquet demonstrates these lookups. The notebook
section executed successfully, and lint checks passed. No new Jev calls were made.
