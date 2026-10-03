# Reusable segment dataset

Start with `all-segments.parquet` or `all-segments.csv.gz`. Separate `semantic-segments` and `fixed-segments` datasets are also included.

Every row preserves original text, recording title, timestamps, audio link, episode ID, duplicate recording IDs and audio hash. Added fields include `topic_id`, `topic_name`, `confidence`, `model_key`, `topic_key`, and `segment_key`.

`other_model_overlaps` preserves the other segmentation's native topic assignments and their exact time overlaps. JSONL stores these as arrays; CSV and Parquet store JSON strings. Read the manifest for confidence and coverage semantics.

```python
import pandas as pd
segments = pd.read_parquet('all-segments.parquet')
# Or: pd.read_csv('all-segments.csv.gz', dtype={'episode_id': str})
talk = segments[segments.episode_id == '2247188684']
```

`segment-topic-index.csv` is a compact version without text or overlap arrays. Full files live in the analysis workspace under `data/topic-analysis/results/segment-dataset/`. Install `pyarrow` to regenerate Parquet output.

The complete combined Parquet dataset is committed in this directory. Other full formats
(CSV and JSONL) are available in the analysis workspace. Use `segmentation` to select
`semantic` or `fixed` rows, and use `other_model_overlaps` to inspect the second topic view.

To reproduce the export from the saved experiment assignments:

```bash
PYTHONPATH=src python scripts/export_segment_dataset.py \
  --input-root /workspace/worktrees/topic-detail/results/full \
  --output-root data/topic-analysis/results/segment-dataset
```
