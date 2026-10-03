# Topic embedding maps

Open `semantic-bge-topic-map.html` (236 topics) or `fixed-bge-topic-map.html` (346 topics) in a browser. Both files are standalone and work offline. Hover shows topic names and counts; dot area is proportional to the number of assigned segments.

The maps share a UMAP projection of normalized mean BGE embeddings for all 582 topics. The two panels use the same coordinates and axis ranges. Topic -1 is excluded. Distances in 2D are approximate. Names are automatic keyword labels.

Original 1,024-dimensional topic centroids are in `topic-centroids.npz`; labels, counts and plotted coordinates are in `topic-coordinates.csv`.

To regenerate from the segment dataset and cached BGE embeddings:

```bash
PYTHONPATH=src python scripts/export_topic_maps.py \
  --dataset-root data/topic-analysis/results/segment-dataset \
  --cache-root data/topic-analysis/cache \
  --output-root data/topic-analysis/results/topic-maps
```
