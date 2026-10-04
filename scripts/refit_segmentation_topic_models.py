"""Refit controlled BGE/BERTopic models on native segmentation variants."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from arvamusfestivali_transcripts.topic_analysis.cache import CacheIdentity, EmbeddingCache
from arvamusfestivali_transcripts.topic_analysis.corpus import _load_archive
from arvamusfestivali_transcripts.topic_analysis.embedders import (
    ADAPTER_VERSION,
    BgeM3Embedder,
    embed_passages,
)
from arvamusfestivali_transcripts.topic_analysis.evaluation import export_experiment
from arvamusfestivali_transcripts.topic_analysis.modelling import (
    TopicModelConfig,
    _representative_ids,
)
from arvamusfestivali_transcripts.topic_analysis.plotting import (
    plot_episode_topic_heatmap,
    plot_topic_sizes,
)
from arvamusfestivali_transcripts.topic_analysis.types import ChunkingConfig, Passage, TopicRun


class ReusedCache(EmbeddingCache):
    """Read old shared vectors and persist new vectors in the local analysis cache."""

    def __init__(self, shared: Path, local: Path):
        super().__init__(local)
        self.shared = EmbeddingCache(shared)

    def get(self, identity, passage):
        vector = super().get(identity, passage)
        return vector if vector is not None else self.shared.get(identity, passage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    base = root / "data/topic-analysis/results"
    destination = base / "segmentation-experiments/topic-models"
    destination = args.output or destination
    destination.mkdir(parents=True, exist_ok=True)
    notebook_path = root / "notebooks/compare_segmentation_modes.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def build_fine_model(" in "".join(cell["source"])
    )
    scope = {
        "SEGMENTATION_MODES": ("semantic",),
        "MODEL_KEYS": ("bge",),
        "DRY_RUN_WITH_FAKE_EMBEDDINGS": False,
    }
    exec(compile(source, str(notebook_path) + ":fine-model-definition", "exec"), scope)
    options = scope["FINE_CANDIDATES"]["leaf-local-6-2"]
    adapter = BgeM3Embedder(device="cuda" if torch.cuda.is_available() else "cpu")
    cache_chunking = ChunkingConfig(
        min_seconds=90,
        target_seconds=300,
        max_seconds=600,
        min_words=1,
        max_words=1_000_000,
        overlap_seconds=0,
        merge_tail_seconds=0,
        max_merged_seconds=600,
    )
    identity = CacheIdentity(
        adapter.model_id,
        adapter.model_revision,
        adapter.dimension,
        adapter.instruction,
        cache_chunking,
        ADAPTER_VERSION,
    )
    cache = ReusedCache(
        root / "data/topic-analysis/cache",
        Path(os.environ.get("TOPIC_PILOT_LOCAL_CACHE", "/root/af-topic-pilot-cache")),
    )
    paths = {
        "original": base / "jev-semantic-sample/review-segments.parquet",
        "short3": base / "jev-short-semantic-sample/review-segments.parquet",
    }
    paths.update(
        {
            key: base / "segmentation-experiments" / key / "review-segments.parquet"
            for key in ["cue5", "sentence3", "sentence5"]
        }
    )
    sample_ids = {"2397177249", "2400164916", "2271676346"}
    if args.input_manifest:
        inputs = json.loads(args.input_manifest.read_text())
        paths = {key: Path(value) for key, value in inputs["datasets"].items()}
        sample_ids = set(inputs["episode_ids"])
    source_cues = {
        episode_id: _load_archive(root / "data/transcripts/2026" / f"{episode_id}.json", 2026).cues
        for episode_id in sample_ids
    }
    summaries, stability = [], []
    for variant, path in paths.items():
        data = pd.read_parquet(path)
        data = (
            data[(data.segmentation == "semantic") & data.episode_id.astype(str).isin(sample_ids)]
            .sort_values(["episode_id", "start_seconds"])
            .reset_index(drop=True)
        )
        passages = tuple(
            Passage(
                passage_id=r.passage_id,
                episode_id=str(r.episode_id),
                duplicate_episode_ids=tuple(json.loads(r.duplicate_episode_ids)),
                title=r.talk_name,
                start_seconds=r.start_seconds,
                end_seconds=r.end_seconds,
                text=r.text,
                audio_sha256=r.audio_sha256,
                audio_url=r.audio_link.split("#t=")[0],
                word_count=len(r.text.split()),
                cue_count=int(
                    getattr(
                        r,
                        "cue_count",
                        sum(
                            cue.start_seconds >= r.start_seconds - 1e-6
                            and cue.end_seconds <= r.end_seconds + 1e-6
                            for cue in source_cues[str(r.episode_id)]
                        ),
                    )
                ),
            )
            for r in data.itertuples()
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "segments": data[["segment_key", "text", "audio_sha256"]].to_dict("records"),
                    "cache": asdict(identity),
                    "fine_definition": source,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        run_root = destination / variant
        run_root.mkdir(parents=True, exist_ok=True)
        saved_vectors = run_root / "embeddings.npy"
        saved_metadata = run_root / "embedding-manifest.json"
        if saved_vectors.exists() and saved_metadata.exists():
            metadata = json.loads(saved_metadata.read_text())
            if metadata["fingerprint"] != fingerprint:
                raise ValueError("Saved embeddings belong to a different input/configuration")
            vectors = np.load(saved_vectors, allow_pickle=False)
            embedding = replace(
                embed_passages(adapter, passages[:1], cache, 2, chunking=cache_chunking),
                passage_ids=tuple(p.passage_id for p in passages),
                embeddings=vectors,
                cache_hits=len(passages),
            )
        else:
            parts, hits = [], 0
            for offset in range(0, len(passages), 8):
                part = embed_passages(
                    adapter, passages[offset : offset + 8], cache, 2, chunking=cache_chunking
                )
                parts.append(part.embeddings)
                hits += part.cache_hits
                print(
                    f"{variant}: embedded {min(offset + 8, len(passages))}/{len(passages)} "
                    f"(cache hits so far {hits})",
                    flush=True,
                )
            vectors = np.concatenate(parts)
            embedding = replace(
                part,
                passage_ids=tuple(p.passage_id for p in passages),
                embeddings=vectors,
                cache_hits=hits,
            )
            with saved_vectors.open("wb") as stream:
                np.save(stream, vectors, allow_pickle=False)
            saved_metadata.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "cache_identity": asdict(identity),
                        "shared_cache_hits": hits,
                        "passage_ids": list(embedding.passage_ids),
                    },
                    indent=2,
                )
            )
        seed_labels = {}
        for seed in args.seeds:
            namespace = f"pilot-refit:{variant}:bge:leaf-local-6-2:{seed}:{fingerprint[:12]}"
            model = scope["build_fine_model"](options, seed, len(passages))
            labels, strength = model.fit_transform(data.text.tolist(), embeddings=vectors)
            labels = np.asarray(labels, dtype=int)
            seed_labels[seed] = labels
            strength = np.asarray(strength, dtype=float)
            info = model.get_topic_info().copy()
            names = {
                int(row.Topic): (
                    "Unassigned" if row.Topic == -1 else " · ".join(row.Representation[:5])
                )
                for row in info.itertuples()
            }
            info["Name"] = [names[int(t)] for t in info.Topic]
            emb = replace(embedding, model_key=namespace)
            config = TopicModelConfig(
                n_neighbors=10,
                n_components=5,
                min_cluster_size=6,
                min_samples=2,
                top_n_words=15,
                random_state=seed,
            )
            run = TopicRun(
                model_key=namespace,
                topics=labels,
                probabilities=strength,
                reduced_embeddings=np.zeros((len(labels), 2)),
                topic_info=info,
                representative_passages=_representative_ids(passages, labels, model),
                cluster_persistence=tuple(
                    float(p) for p in model.hdbscan_model.cluster_persistence_
                ),
                passage_ids=emb.passage_ids,
                topic_model_config=asdict(config),
                clustering_embeddings=np.nan_to_num(model.umap_model.embedding_),
            )
            output = export_experiment(
                run_root,
                f"seed-{seed}",
                passages,
                {namespace: run},
                {namespace: emb},
                chunking=cache_chunking,
                topic_model=config,
                cache_identities={namespace: identity},
                git_revision="unavailable-workspace-no-git",
            )
            assignments = data.copy()
            assignments["cluster_model_key"] = namespace
            assignments["cluster_topic_id"] = labels
            assignments["cluster_topic_key"] = [f"{namespace}:{topic}" for topic in labels]
            assignments["cluster_topic_name"] = [names[int(topic)] for topic in labels]
            assignments["cluster_membership_strength"] = np.where(labels == -1, np.nan, strength)
            assignments.to_parquet(output / "review-segments.parquet", index=False)
            info.to_csv(output / "topics.csv", index=False)
            evidence = []
            for topic, group in assignments.groupby("cluster_topic_id"):
                if topic == -1:
                    continue
                evidence.append(
                    {
                        "topic_key": f"{namespace}:{topic}",
                        "topic_id": int(topic),
                        "keyword_label": names[int(topic)],
                        "segment_count": len(group),
                        "full_text_members": group[
                            [
                                "segment_key",
                                "episode_id",
                                "start_seconds",
                                "end_seconds",
                                "text",
                                "audio_link",
                            ]
                        ].to_dict("records"),
                    }
                )
            (output / "topic-evidence.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evidence)
            )
            model.save(str(output / "model.pkl"), serialization="pickle")
            manifest = json.loads((output / "manifest.json").read_text())
            manifest.update(
                {
                    "variant": variant,
                    "fitted_model_key": namespace,
                    "fingerprint": fingerprint,
                    "input_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "input_path": str(path.relative_to(root)),
                    "fine_model_definition_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "cluster_selection_method": "leaf",
                    "stopwords": scope["ALL_STOPWORDS"],
                    "bm25_weighting": True,
                    "reduce_frequent_words": True,
                    "ngram_range": [1, 2],
                    "topic_name_source": "c-TF-IDF keywords from this fit; "
                    "no old names transferred",
                    "display_coordinates": "placeholder zeros; not a semantic map",
                }
            )
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False)
            )
            valid = labels != -1
            sizes = pd.Series(labels[valid]).value_counts()
            cosine_silhouette = (
                float(silhouette_score(vectors[valid], labels[valid], metric="cosine"))
                if 1 < len(sizes) < valid.sum()
                else None
            )
            duration = (data.end_seconds - data.start_seconds).to_numpy()
            transitions = []
            for _, group in assignments.groupby("episode_id"):
                seq = group.cluster_topic_id.to_numpy()
                changes = int(np.sum(seq[1:] != seq[:-1]))
                transitions.append(
                    {
                        "episode_id": str(group.episode_id.iloc[0]),
                        "topic_count": len(set(seq) - {-1}),
                        "topic_changes": changes,
                        "changes_per_hour": changes
                        / (float((group.end_seconds - group.start_seconds).sum()) / 3600),
                    }
                )
            pd.DataFrame(transitions).to_csv(output / "talk-dynamics.csv", index=False)
            summaries.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "segments": len(data),
                    "topic_count": len(sizes),
                    "outlier_count": int((~valid).sum()),
                    "outlier_time_pct": float(duration[~valid].sum() / duration.sum() * 100),
                    "outlier_segment_pct": float((~valid).mean() * 100),
                    "median_topic_size": float(sizes.median()) if len(sizes) else None,
                    "original_cosine_silhouette": cosine_silhouette,
                    "topic_changes": sum(item["topic_changes"] for item in transitions),
                    "mean_changes_per_hour": float(
                        np.mean([item["changes_per_hour"] for item in transitions])
                    ),
                    "model_key": namespace,
                }
            )
            if seed == 42:
                plot_topic_sizes(run, passages).write_html(
                    output / "topic-sizes.html", include_plotlyjs=True
                )
                plot_episode_topic_heatmap(run, passages).write_html(
                    output / "talk-topic-heatmap.html", include_plotlyjs=True
                )
            print(
                f"{variant} seed {seed}: {len(sizes)} topics; "
                f"{int((~valid).sum())}/{len(data)} unassigned",
                flush=True,
            )
        for left_seed, right_seed in itertools.combinations(args.seeds, 2):
            shared = (seed_labels[left_seed] != -1) & (seed_labels[right_seed] != -1)
            stability.append(
                {
                    "variant": variant,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "shared_inlier_fraction": float(shared.mean()),
                    "outlier_status_agreement": float(
                        np.mean((seed_labels[left_seed] == -1) == (seed_labels[right_seed] == -1))
                    ),
                    "inlier_ARI": float(
                        adjusted_rand_score(
                            seed_labels[left_seed][shared], seed_labels[right_seed][shared]
                        )
                    )
                    if shared.sum() > 1
                    else None,
                }
            )
    pd.DataFrame(summaries).to_csv(destination / "comparison-summary.csv", index=False)
    pd.DataFrame(stability).to_csv(destination / "seed-stability.csv", index=False)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "sample_episode_ids": sorted(sample_ids),
                "variants": list(paths),
                "seeds": args.seeds,
                "candidate": "leaf-local-6-2",
                "device": adapter.device,
                "embedding_model": adapter.model_id,
                "embedding_revision": adapter.model_revision,
                "versions": {
                    name: version(name)
                    for name in [
                        "bertopic",
                        "hdbscan",
                        "umap-learn",
                        "numpy",
                        "scikit-learn",
                        "torch",
                    ]
                },
                "scope": f"Independent {len(sample_ids)}-talk fits; do not compare counts directly "
                "to full-corpus 236 topics",
            },
            indent=2,
        )
    )
    print(
        pd.DataFrame(summaries)[
            ["variant", "seed", "segments", "topic_count", "outlier_time_pct", "topic_changes"]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
