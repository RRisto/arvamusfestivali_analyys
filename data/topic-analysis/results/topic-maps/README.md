# Topic embedding maps

Open `semantic-bge-topic-map.html` (236 topics) or `fixed-bge-topic-map.html` (346 topics) in a browser. Both files are standalone and work offline. Hover shows topic names and counts; dot area is proportional to the number of assigned segments.

The maps share a UMAP projection of normalized mean BGE embeddings for all 582 topics. The two panels use the same coordinates and axis ranges. Topic -1 is excluded. Distances in 2D are approximate. Names are evidence-grounded Estonian LLM labels. Original keywords are retained.

Original 1,024-dimensional topic centroids are in `topic-centroids.npz`; labels, counts and plotted coordinates are in `topic-coordinates.csv`.

To regenerate from the segment dataset and cached BGE embeddings:

```bash
PYTHONPATH=src python scripts/export_topic_maps.py \
  --dataset-root data/topic-analysis/results/segment-dataset \
  --cache-root data/topic-analysis/cache \
  --output-root data/topic-analysis/results/topic-maps
```

Topic names were generated using GPT-5.4 mini from keywords and up to 20 full-text examples per topic. See `../topic-names/topic-review.html` for evidence and coherence assessments. Original keyword names are retained; membership strengths and segment boundaries are unchanged.
