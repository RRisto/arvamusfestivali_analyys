"""Sampling must preserve genuine evidence and cover both strong and weak members."""

import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "topic_naming", Path(__file__).parents[2] / "scripts/name_topics_openai.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_small_topic_uses_every_real_segment_once():
    frame = pd.DataFrame(
        {
            "segment_key": list("abcdef"),
            "episode_id": ["talk"] * 6,
            "confidence": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )
    sampled = module.sample_examples(frame)
    assert len(sampled) == 6
    assert set(sampled.segment_key) == set(frame.segment_key)


def test_large_topic_includes_other_talks_and_weak_members():
    frame = pd.DataFrame(
        {
            "segment_key": [str(i) for i in range(40)],
            "episode_id": ["dominant"] * 35 + ["other"] * 5,
            "confidence": [1 - i / 40 for i in range(40)],
        }
    )
    sampled = module.sample_examples(frame)
    assert len(sampled) == 20
    assert sampled.segment_key.is_unique
    assert sampled.episode_id.nunique() == 2
    assert {"38", "39"}.issubset(set(sampled.segment_key))
