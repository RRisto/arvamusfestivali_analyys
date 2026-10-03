"""Apply reviewed LLM names without changing segment assignments or map coordinates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path

import pandas as pd

from arvamusfestivali_transcripts.topic_analysis.topic_map import plot_topic_embedding_map


def apply(root: Path):
    results = root / "data/topic-analysis/results"
    naming = results / "topic-names"
    rows = [json.loads(line) for line in (naming / "responses.jsonl").read_text().splitlines()]
    names = {r["topic_key"]: r for r in rows}
    dataset = results / "segment-dataset"
    data = pd.read_parquet(dataset / "all-segments.parquet")
    expected = set(data.loc[data.topic_id >= 0, "topic_key"])
    if set(names) != expected or len(rows) != len(expected):
        raise ValueError("Naming must cover every assigned topic exactly once")
    original = data.copy(deep=True)
    if "topic_name_keywords" not in data:
        data["topic_name_keywords"] = data.topic_name
    data["topic_name"] = [
        names[k]["topic_name_et"] if k in names else "Unassigned" for k in data.topic_key
    ]
    data["topic_name_source"] = [
        "openai:" + names[k]["model"] if k in names else "unassigned" for k in data.topic_key
    ]
    data["topic_naming_confidence"] = [
        names[k]["naming_confidence"] if k in names else None for k in data.topic_key
    ]
    data["topic_coherence"] = [
        names[k]["coherence"] if k in names else None for k in data.topic_key
    ]

    def relabel(encoded):
        links = json.loads(encoded)
        for link in links:
            key = link["topic_key"]
            if key in names:
                link.setdefault("topic_name_keywords", link["topic_name"])
                link["topic_name"] = names[key]["topic_name_et"]
        return json.dumps(links, ensure_ascii=False)

    data["other_model_overlaps"] = data.other_model_overlaps.map(relabel)
    protected = [c for c in original if c not in {"topic_name", "other_model_overlaps"}]
    pd.testing.assert_frame_equal(original[protected], data[protected])
    manifest = json.loads((dataset / "manifest.json").read_text())
    for stem, frame in {
        "all-segments": data,
        "semantic-segments": data[data.segmentation == "semantic"],
        "fixed-segments": data[data.segmentation == "fixed"],
    }.items():
        frame.to_parquet(dataset / f"{stem}.parquet", index=False, compression="zstd")
        frame.to_csv(
            dataset / f"{stem}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
        )
        with gzip.open(dataset / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as stream:
            for record in json.loads(frame.to_json(orient="records", force_ascii=False)):
                for col in ["other_model_overlaps", "duplicate_episode_ids"]:
                    record[col] = json.loads(record[col])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        for suffix in ["parquet", "csv.gz", "jsonl.gz"]:
            path = dataset / f"{stem}.{suffix}"
            manifest["files"][path.name] = dict(
                rows=len(frame),
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    manifest["schema_version"] = 2
    manifest["topic_naming"] = dict(
        model=rows[0]["model"],
        topic_count=len(rows),
        evidence="../topic-names/topic-evidence.jsonl.gz",
        preserved_keyword_field="topic_name_keywords",
    )
    (dataset / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    data.drop(columns=["text", "other_model_overlaps"]).to_csv(
        dataset / "segment-topic-index.csv", index=False
    )
    pd.concat(
        [data[data.segmentation == "semantic"].head(3), data[data.segmentation == "fixed"].head(2)]
    ).to_csv(dataset / "preview.csv", index=False)
    for mode in ["semantic", "fixed"]:
        path = results / f"fine-topics/{mode}-bge/leaf-local-6-2/seed-42/topics.csv"
        topics = pd.read_csv(path)
        keys = [f"{mode}:bge:leaf-local-6-2:42:{i}" for i in topics.Topic]
        for source, target in [
            ("topic_name_et", "LLM_Name"),
            ("summary_et", "LLM_Summary"),
            ("coherence", "LLM_Coherence"),
            ("naming_confidence", "LLM_Naming_Confidence"),
        ]:
            topics[target] = [names[k][source] if k in names else None for k in keys]
        topics.to_csv(path, index=False)
    maps = results / "topic-maps"
    coords = pd.read_csv(maps / "topic-coordinates.csv")
    coords["topic_name_keywords"] = coords.topic_name
    coords["topic_name"] = [
        names[f"{k}:{i}"]["topic_name_et"]
        for k, i in zip(coords.model_key, coords.topic_id, strict=True)
    ]
    coords.to_csv(maps / "topic-coordinates.csv", index=False)
    low, high = coords[["x", "y"]].min(), coords[["x", "y"]].max()
    pad = (high - low) * 0.07
    for mode, color in [("semantic", "#2563eb"), ("fixed", "#d97706")]:
        frame = coords[coords.segmentation == mode]
        fig = plot_topic_embedding_map(
            frame,
            title=f"{mode.title()} BGE topics — Estonian LLM names",
            color=color,
            max_segment_count=int(coords.segment_count.max()),
        )
        fig.update_xaxes(range=[low.x - pad.x, high.x + pad.x])
        fig.update_yaxes(range=[low.y - pad.y, high.y + pad.y])
        fig.write_html(
            maps / f"{mode}-bge-topic-map.html",
            include_plotlyjs=True,
            config={"scrollZoom": True, "displaylogo": False},
        )
    packets = [
        json.loads(line) for line in (naming / "topic-evidence.jsonl").read_text().splitlines()
    ]
    body = [
        '<!doctype html><meta charset="utf-8"><title>Topic naming review</title>',
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}"
        "details{border:1px solid #ddd;padding:12px;margin:12px 0}summary{cursor:pointer}"
        "pre{white-space:pre-wrap}input{padding:12px;width:95%}</style>",
        "<h1>582 detailed topics</h1><p>Keywords and up to 20 genuine segment examples. "
        "Small topics include every available segment. Naming confidence is separate "
        'from cluster membership strength.</p><input placeholder="Search topics" '
        "oninput=\"document.querySelectorAll('details.topic').forEach(e=>"
        'e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))">',
    ]
    for packet in packets:
        row = names[packet["topic_key"]]

        def esc(value):
            return html.escape(str(value))

        body.append(
            f'<details class="topic"><summary>{esc(row["topic_name_et"])} '
            f"— {esc(packet['topic_key'])} · {packet['segment_count']} segments</summary>"
            f"<p>{esc(row['summary_et'])}</p><p>Keywords: {esc(packet['keywords'])}</p>"
            f"<p>Coherence: {esc(row['coherence'])}; naming confidence: "
            f"{esc(row['naming_confidence'])}</p>"
        )
        for example in packet["examples"]:
            body.append(
                f"<h4>Example {example['number']}: {esc(example['talk_name'])} "
                f"[{example['start_seconds']}–{example['end_seconds']}s]</h4>"
                f"<p>Membership strength: {example['confidence']:.3f}</p>"
                f"<pre>{esc(example['text'])}</pre>"
            )
        body.append("</details>")
    (naming / "topic-review.html").write_text("\n".join(body))
    with gzip.open(naming / "topic-evidence.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write((naming / "topic-evidence.jsonl").read_text())
    input_tokens = sum(r["usage"]["input_tokens"] for r in rows)
    output_tokens = sum(r["usage"]["output_tokens"] for r in rows)
    stats = dict(
        topics=len(rows),
        model=rows[0]["model"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_standard_cost_usd=(input_tokens * 0.75 + output_tokens * 4.5) / 1e6,
        topics_with_fewer_than_10_examples=sum(len(p["examples"]) < 10 for p in packets),
        example_count=sum(len(p["examples"]) for p in packets),
    )
    (naming / "manifest.json").write_text(json.dumps(stats, indent=2))
    (naming / "README.md").write_text(
        "# Evidence-grounded Estonian topic names\n\n"
        "Open `topic-review.html` to inspect every topic, its keywords and 6–20 unique "
        "full-text segment examples with recording titles, timestamps and membership strengths. "
        "Topics smaller than ten segments use all available segments. "
        "No examples are fabricated.\n\n"
        "`topic-names.csv` contains names, summaries, coherence assessments, naming confidence, "
        "subthemes, evidence references and API usage. "
        "`responses.jsonl` is a resumable checkpoint; "
        "`topic-evidence.jsonl.gz` retains the exact full-text prompts. The model is pinned; "
        "requests use structured outputs and `store=False`. Transcript text is treated as data.\n\n"
        "Topic IDs, assignments, confidence, segment metadata, centroid embeddings and map "
        "coordinates are preserved. Keyword names remain in `topic_name_keywords`. "
        "Mixed/unclear topics warrant review before merging.\n"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    apply(parser.parse_args().root)
