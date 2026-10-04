# Published analysis artifacts

Additional plots and embedding arrays are indexed in [published-additions.json](published-additions.json), including byte sizes and SHA-256 checksums.

- [Semantic BGE topic hierarchy](figures/semantic-bge-topic-hierarchy.html)
- [Semantic BGE treemap](figures/semantic-bge-treemap-40.html)
- [Fixed BGE topic sizes](segmentation-full/fixed-bge-topic-sizes.html)
- [Semantic BGE topic sizes](segmentation-full/semantic-bge-topic-sizes.html)
- [Fixed Qwen topic sizes](segmentation-full/fixed-qwen-topic-sizes.html)
- [Semantic Qwen topic sizes](segmentation-full/semantic-qwen-topic-sizes.html)
- [Fixed Qwen/BGE correspondence](segmentation-full/fixed-qwen-bge-correspondence.html)
- [Semantic Qwen/BGE correspondence](segmentation-full/semantic-qwen-bge-correspondence.html)

Download the repository and open HTML files locally to use the interactive plots. Keep the relative directory structure: plots load their shared JavaScript from `plotly-assets/`.

## Embedding arrays

Load `.npy` and `.npz` files with `numpy.load(path, allow_pickle=False)`.

- `jev-semantic-all/bubble-maps/segment-embeddings.npy`: 4,132 × 1,024 float32 embeddings. Row order is recorded in the adjacent `embedding-manifest.json` under `segment_keys`.
- `jev-semantic-all/bubble-maps/topic-centroids.npz`: arrays `embeddings`, `topic_ids`, and `topic_keys`.
- `jev-semantic-all/segment-maps/coordinates.npy`: 4,132 × 2 float32 map coordinates; these are projected positions, not full embeddings. See the adjacent manifest and segment map exports.
- `segmentation-experiments/topic-models/{original,cue5,sentence3,sentence5,short3}/embeddings.npy`: pilot embeddings, with row order and model configuration in each adjacent `embedding-manifest.json`.
- `segmentation-experiments/broader-pilot/topic-models/{original,sentence5}/embeddings.npy`: broader pilot embeddings, with adjacent embedding manifests.

The full content-addressed cache under `data/topic-analysis/cache/` is not included in this publication.

## Fitted models and reusable datasets

The 16 `model.pkl` files under `segmentation-experiments/topic-models/` and
`segmentation-experiments/broader-pilot/topic-models/` preserve the fitted pilot
models for their original segmentation and random seed. See each adjacent
`manifest.json` for its parameters and package versions. Pickle files require a
trusted source and compatible Python dependencies when loading.

The complete fixed/semantic BGE and Qwen topic tables are in `segmentation-full/`.
`segment-dataset/fixed-segments.parquet`, `semantic-segments.parquet` and
`segment-topic-index.csv` preserve reusable segments and their topic mappings.

File sizes and SHA-256 checksums are included in `published-additions.json`.
The full embedding cache is downloadable from the
[October 2026 release](https://github.com/RRisto/arvamusfestivali_analyys/releases/tag/analysis-cache-2026-10-04).
