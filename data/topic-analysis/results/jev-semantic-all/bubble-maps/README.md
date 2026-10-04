# Jev topic bubble maps

Open index.html to switch between segments.html and talks.html. Each standalone
HTML file works offline. The236 detailed topics share one set of2D coordinates;
bubble area follows either segment count or distinct recording count. Color
identifies the20 dictionary high-level categories and matches the treemaps.
Hover displays topic names and both counts. Click the legend to filter categories,
and pan or zoom to explore.

Coordinates come from normalized mean original BGE-M3 embeddings of the segments
assigned to each primary topic by Jev. These are Jev member centroids, rather than
the old clustering centroids. Genuine cached1,024-dimensional vectors were reused;
no API calls, re-embedding or topic-model fitting was needed. UMAP uses cosine,
15 neighbors, min_dist.15, two components and seed42. Distances are approximate.
The389 unassigned segments are excluded because they do not define a coherent topic.

CSV coordinates, original-space centroids, figure JSON and source/cache manifests
are included. The review notebook contains a count selector for the two views.
All236 counts, exact shared positions, category colors, area sizing and centroids
were checked against the source data; both notebook views rendered in headless
verification. Three topic-map tests and lint checks passed.

Regenerate in the established analysis environment from the project root:
PYTHONPATH=src python scripts/export_jev_bubble_maps.py

All topics use circular bubbles with20 unique colors selected from the user-provided
named color list. The shared lookup is results/high-level-topics/category-style.json.
Treemaps use the same colors. Positions and count-sized areas are preserved.
