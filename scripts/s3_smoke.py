"""Smoke-test Turftopic S³ against the saved semantic BGE embeddings.

Run from the repository root with:

    uv run --group topic-analysis python scripts/s3_smoke.py

This uses the complete saved semantic embedding matrix. Turftopic receives it through
``embeddings=`` and encodes only vocabulary terms with the matching BGE revision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import SemanticSignalSeparation

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/topic-analysis/results"
MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
N_COMPONENTS = 2
RANDOM_STATE = 42


def main() -> None:
    dataset_path = RESULTS / "segment-dataset/all-segments.parquet"
    embedding_path = RESULTS / "jev-semantic-all/bubble-maps/segment-embeddings.npy"
    embedding_manifest_path = (
        RESULTS / "jev-semantic-all/bubble-maps/embedding-manifest.json"
    )
    model_manifest_path = RESULTS / "segment-dataset/manifest.json"

    dataset = pd.read_parquet(
        dataset_path,
        columns=[
            "segment_key",
            "segmentation",
            "passage_id",
            "episode_id",
            "title",
            "start_seconds",
            "end_seconds",
            "text",
            "topic_id",
        ],
    )
    passages = dataset.loc[dataset.segmentation == "semantic"].reset_index(drop=True)
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_info = model_manifest["models"]["semantic"]
    expected_keys = embedding_manifest["segment_keys"]
    embeddings = np.load(embedding_path, allow_pickle=False)

    if model_info["embedding_model_id"] != MODEL_ID:
        raise ValueError("Saved dataset uses an unexpected embedding model")
    if model_info["embedding_revision"] != MODEL_REVISION:
        raise ValueError("Saved dataset uses an unexpected embedding revision")
    if int(model_info["embedding_dimension"]) != 1024:
        raise ValueError("Saved dataset has an unexpected embedding dimension")
    if passages.segment_key.tolist() != expected_keys:
        raise ValueError("Saved embedding rows do not align with semantic passage keys")
    if embeddings.shape != (len(passages), 1024):
        raise ValueError(f"Unexpected saved embedding shape: {embeddings.shape}")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError("Saved embeddings must be finite float32 values")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Saved embeddings are not L2-normalized")

    # Keep term embeddings in the same space. The corpus embeddings above are passed
    # directly to Turftopic and are not recomputed.
    encoder = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    model = SemanticSignalSeparation(
        n_components=N_COMPONENTS,
        encoder=encoder,
        vectorizer=CountVectorizer(max_features=500),
        random_state=RANDOM_STATE,
    )
    scores = model.fit_transform(passages.text.tolist(), embeddings=embeddings)
    if scores.shape != (len(passages), N_COMPONENTS) or not np.isfinite(scores).all():
        raise ValueError(f"Unexpected S³ score matrix: {scores.shape}")

    print(
        f"S³ fit passed: {len(passages):,} semantic passages, "
        f"{embeddings.shape[1]}-dimensional saved embeddings, "
        f"{N_COMPONENTS} components, seed {RANDOM_STATE}."
    )
    print("Passage embeddings were loaded from disk; no passage re-embedding was run.")
    for component_id in range(N_COMPONENTS):
        column = scores[:, component_id]
        print(f"\nComponent {component_id}: highest scoring passages")
        for index in np.argsort(column)[-3:][::-1]:
            row = passages.iloc[index]
            print(
                f"  score={column[index]:.4f} topic={row.topic_id} "
                f"{row.title} [{row.start_seconds:.1f}–{row.end_seconds:.1f}] "
                f"{row.text[:220]}"
            )
        print(f"Component {component_id}: lowest scoring passages")
        for index in np.argsort(column)[:3]:
            row = passages.iloc[index]
            print(
                f"  score={column[index]:.4f} topic={row.topic_id} "
                f"{row.title} [{row.start_seconds:.1f}–{row.end_seconds:.1f}] "
                f"{row.text[:220]}"
            )


if __name__ == "__main__":
    main()
