# Broader segmentation pilot

Purposeful 18-talk sample; original full corpus remains unchanged. Compare original native segments with sentence-aware semantic cuts (q70, minimum60s, nominal maximum300s, sentence snap ±20s, minimum snapped45s). Three seeds:42,7,19. Same BGE-M3 encoder and leaf-local-6-2 BERTopic settings. Topic names are fresh c-TF-IDF keywords, not transferred original names.

See index.html for metrics, native timelines and 54 shared eight-minute windows per seed. Charts load the bundled local plotly.min.js; keep it beside the HTML files for offline review. manual-review.csv holds blank human ratings; no quality scores have been fabricated. Topic switches include -1 and increase mechanically with more segments. Seed ARI uses jointly assigned segments; inspect shared-inlier fraction too. Sample topic counts must not be compared directly to the full-corpus236.

This compares a package of changes: shorter maximum, lower semantic boundary threshold and sentence snapping. It does not isolate snapping alone. The English-language candidate2314163822 was excluded for sparse cue timing including an indivisible377.96-second cue; see inputs.json.

```csv
variant,seed,segments,topic_count,outlier_count,outlier_time_pct,outlier_segment_pct,median_topic_size,original_cosine_silhouette,topic_changes,mean_changes_per_hour,model_key
original,42,324,24,23,5.164862841117176,7.098765432098765,13.0,0.1344279050827026,81,3.067083462461304,pilot-refit:original:bge:leaf-local-6-2:42:45939440211b
original,7,324,23,24,4.505153556998849,7.4074074074074066,13.0,0.1365323513746261,81,3.0524652046664595,pilot-refit:original:bge:leaf-local-6-2:7:45939440211b
original,19,324,25,19,4.230769672124563,5.864197530864197,11.0,0.1222348436713218,79,3.076863831890781,pilot-refit:original:bge:leaf-local-6-2:19:45939440211b
sentence5,42,642,41,124,14.24413613938984,19.31464174454829,11.0,0.0857416465878486,321,12.100526755336452,pilot-refit:sentence5:bge:leaf-local-6-2:42:ea8c7bf4489f
sentence5,7,642,38,136,18.084638409492584,21.18380062305296,10.5,0.0975470244884491,297,11.402965549592764,pilot-refit:sentence5:bge:leaf-local-6-2:7:ea8c7bf4489f
sentence5,19,642,42,119,16.251294174682535,18.53582554517134,10.0,0.0741445273160934,314,11.754517592650858,pilot-refit:sentence5:bge:leaf-local-6-2:19:ea8c7bf4489f
```


```csv
variant,left_seed,right_seed,shared_inlier_fraction,outlier_status_agreement,inlier_ARI
original,42,7,0.8888888888888888,0.9228395061728396,0.9183482440192248
original,42,19,0.8888888888888888,0.9074074074074074,0.8827332154807318
original,7,19,0.8827160493827161,0.8981481481481481,0.9078921881887134
sentence5,42,7,0.7227414330218068,0.8504672897196262,0.8532686946915023
sentence5,42,19,0.719626168224299,0.8177570093457944,0.84086249333566
sentence5,7,19,0.705607476635514,0.8084112149532711,0.7946246999330328
```


## Interpretation and reproduction

See [ASSESSMENT.md](ASSESSMENT.md) for the recommendation and six matched readings.
From the project root in the established topic-analysis environment:

```bash
PYTHONPATH=src python scripts/prepare_broader_segmentation_pilot.py
PYTHONPATH=src python scripts/short_semantic_pilot.py --variant sentence5 \
  --original-dataset data/topic-analysis/results/segmentation-experiments/broader-pilot/original.parquet \
  --output data/topic-analysis/results/segmentation-experiments/broader-pilot/sentence5 \
  --episode-ids 2397177249 2400164916 2271676346 2247188693 2250797867 2314163816 2314163813 2387966373 2393585466 2394248499 2395857123 2395991463 2396670927 2396692746 2397177243 2397264585 2400184983 2400207765
```

For the segmentation command use the original dataset and output under
`data/topic-analysis/results/segmentation-experiments/broader-pilot/`, and supply
all episode IDs from inputs.json using --episode-ids. Then run:

```bash
PYTHONPATH=src python scripts/refit_segmentation_topic_models.py \
  --input-manifest data/topic-analysis/results/segmentation-experiments/broader-pilot/inputs.json \
  --output data/topic-analysis/results/segmentation-experiments/broader-pilot/topic-models \
  --seeds 42 7 19
PYTHONPATH=src python scripts/review_broader_pilot_examples.py
PYTHONPATH=src python scripts/export_broader_pilot_review.py
```

Validation: all six fits preserve every input source column exactly, use unique
segment keys and run-scoped topics, and store genuine normalized 1,024-dimensional
BGE vectors in source order. Saved models and manifests are present. Noise strength
is null. All three overlap totals agree. The 40 focused tests and lint checks pass.

The notebook executed successfully in headless verification, including loading all
six fits and switching to seed19. Only the executed copy bypasses nbclient
Output-widget capture bookkeeping; the source notebook keeps standard widgets.
The full-corpus hash is unchanged and baseline rows match the original dataset.
