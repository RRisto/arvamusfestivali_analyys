# Canonical talk metadata enriched with festival programmes

213/219 talks have explicit organizer labels. See manifest.json and ../jev-topic-analysis/programme-match-audit.csv for coverage.

talks.parquet / talks.csv join canonical catalog metadata, audio deduplication provenance and programme fields. segments-with-metadata.parquet attaches these fields to all native fixed/semantic rows. The original dataset remains unchanged.

organizer is the official label, including joint organizations. Moderator, stage curator and sponsor/support text are separate fields. Empty organizer fields are not inferred from these roles. Event IDs can list repeated sessions when titles and organizers agree but session timing is unresolved.
