"""Prepare a purposeful, varied 18-talk pilot without changing the full corpus."""

import hashlib
import json
from pathlib import Path

import pandas as pd

SELECTION = {
    "2397177249": "AI entrepreneurship; previous pilot",
    "2400164916": "Teacher wellbeing; previous pilot; shorter recording",
    "2271676346": "Military/security; previous pilot",
    "2247188693": "Workplace bullying",
    "2250797867": "Environment, economy and security",
    "2314163816": "International politics",
    "2314163813": "International institutions; UN",
    "2387966373": "Electricity markets; shorter recording",
    "2393585466": "Mathematics education; shorter recording",
    "2394248499": "Cancer patient information",
    "2395857123": "Refugee identity",
    "2395991463": "Parenthood and gender roles",
    "2396670927": "Housing affordability",
    "2396692746": "Community agreements; longer recording",
    "2397177243": "Culture and wellbeing",
    "2397264585": "Food waste and safety",
    "2400184983": "Practical AI workshop; shorter recording",
    "2400207765": "Loneliness and healthcare responsibility",
}


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis/results"
    output = base / "segmentation-experiments/broader-pilot"
    output.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet(base / "segment-dataset/all-segments.parquet")
    original = original[
        (original.segmentation == "semantic") & original.episode_id.astype(str).isin(SELECTION)
    ].copy()
    assert set(original.episode_id.astype(str)) == set(SELECTION)
    assert original.audio_sha256.groupby(original.episode_id).first().is_unique
    original.to_parquet(output / "original.parquet", index=False)
    catalog = (
        original.groupby("episode_id")
        .agg(
            talk_name=("talk_name", "first"),
            segments=("text", "size"),
            duration_seconds=("end_seconds", "max"),
        )
        .reset_index()
    )
    catalog["selection_reason"] = catalog.episode_id.astype(str).map(SELECTION)
    catalog.to_csv(output / "sample-talks.csv", index=False)
    manifest = {
        "episode_ids": list(SELECTION),
        "original_full_dataset_sha256": hashlib.sha256(
            (base / "segment-dataset/all-segments.parquet").read_bytes()
        ).hexdigest(),
        "selection": "Purposeful coverage sample, not random or statistically representative",
        "excluded_candidates": [
            {
                "episode_id": "2314163822",
                "reason": "Sparse ASR timing including an indivisible 377.96-second cue",
                "replacement_episode_id": "2314163813",
            }
        ],
        "datasets": {
            "original": str(output / "original.parquet"),
            "sentence5": str(output / "sentence5/segments.parquet"),
        },
    }
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2))
    print(" ".join(SELECTION))


if __name__ == "__main__":
    main()
