# Charts with Estonian topic names

Open index.html. Both selected BGE models have segment-count charts, talk/topic coverage heatmaps and talk timelines with a recording selector. All pages are standalone. Hover labels retain topic IDs, names and available segment timestamps and cluster membership strengths.

Regenerate without fitting models or API calls:

```bash
PYTHONPATH=src python scripts/export_named_topic_charts.py --dataset data/topic-analysis/results/segment-dataset/all-segments.parquet --output data/topic-analysis/results/named-topic-charts
```
