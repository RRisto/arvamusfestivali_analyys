# Native semantic segment scatter maps

One equal-sized point per segment; color is its Jev high-level dictionary parent, including gray unassigned. All, 2025 and 2026 use a single UMAP projection of saved 1024-dimensional BGE segment embeddings, not topic centroids or labels as inputs. No topic refit or classifier API call.

Offline HTML/Plotly JSON, PNG/SVG, coordinates.npy, full coordinate Parquet and compact CSV are saved here. Projection cache verifies source/vector hashes, ordered segment keys and settings. Hover includes title, year, timestamps, detailed/main topics and a text excerpt.
