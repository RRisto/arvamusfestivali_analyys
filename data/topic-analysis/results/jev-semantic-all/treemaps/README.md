# Jev topic treemaps

Open index.html to switch between segments.html and talks.html. Both standalone
charts include Plotly and work offline. Click a high-level category to drill into
its detailed Jev topics. Gray is explicitly unassigned.

All4,132 native semantic segments and219 recording IDs are included. Parent
categories come from the saved dictionary. Segment areas and counts are additive.
For the talks view, leaves use distinct talk counts as area. Parent areas sum
child topic-talk memberships, while their displayed counts deduplicate the union
of talks. The root displays219 distinct talks. Hover shows both counts and weights.

Count tables, reusable figure JSON and a source-hash manifest are saved alongside
the charts. The review notebook has a metric selector for the same figures.
Every node count was checked against the actual Jev dataset, both notebook views
rendered in headless verification, and lint passed. No model or API was rerun.

Regenerate from the project root with:
PYTHONPATH=src python scripts/export_jev_treemaps.py
