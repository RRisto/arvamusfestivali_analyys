"""Save six recorded qualitative readings; these are not human accuracy scores."""

from pathlib import Path

import pandas as pd

base = (
    Path(__file__).resolve().parents[1]
    / "data/topic-analysis/results/segmentation-experiments/broader-pilot"
)
examples = [
    (
        "2397264585",
        1235,
        "sentence5 for focused lookup",
        (
            "The shorter segment starts at the explicit question contrasting best-before"
            " and use-by labels. The baseline starts mid-sentence and spans the poll, "
            "definitions, product examples and regulation. The new end still separates a"
            " product question from its answer, so punctuation is not complete question-"
            "and-answer context."
        ),
    ),
    (
        "2394248499",
        1245,
        "sentence5 for practical advice; baseline for explanation",
        (
            "The 54-second segment isolates taking a trusted person to the appointment "
            "and asking again. The baseline mixes audience setup, diagnosis shock, "
            "staged information and practical advice. Short text is useful for lookup, "
            "but the opening says everything has already been said and relies on "
            "preceding context."
        ),
    ),
    (
        "2395991463",
        1346,
        "baseline for standalone context",
        (
            "The new segment begins by saying the speaker was lucky and inherited the "
            "mother's traits, but the preceding segment contains the explanation about "
            "the father and mother working alone. The baseline preserves that story and "
            "comparison. A sentence-ending cut splits one narrative despite "
            "grammatically intact boundaries."
        ),
    ),
    (
        "2400164916",
        1100,
        "mixed",
        (
            "The new segment isolates sociodrama role-selection instructions from "
            "participant feedback and ministry discussion. This is a useful activity "
            "boundary, but ends with the ASR fragment Nii ja. The apparent sentence-"
            "ending punctuation does not ensure a complete thought. Activity "
            "instructions may also be less useful as a substantive topic."
        ),
    ),
    (
        "2400184983",
        1500,
        "baseline for standalone role context",
        (
            "The shorter workshop segment combines strategic decision support and "
            "project management. It starts after the explicit heading introducing the "
            "CEO example, so the role anchor is in the previous segment. It is shorter "
            "but neither fully self-contained nor a single use case."
        ),
    ),
    (
        "2393585466",
        1500,
        "sentence5 for focused lookup, with boundary caveat",
        (
            "The new segment centers on definitions of mathematics and removes the "
            "handover and later phone-poll logistics present in the baseline. However, "
            "it ends just after introducing sets, while structures and relations "
            "continue next. It is more focused without preserving the whole conceptual "
            "explanation."
        ),
    ),
]
rows = []
for eid, t, pref, note in examples:
    row = {
        "episode_id": eid,
        "comparison_seconds": t,
        "reviewer": "Codex qualitative reading; not human validation",
        "preference": pref,
        "assessment": note.replace("sociodrama", "sociodrama"),
    }
    for variant in ("original", "sentence5"):
        path = base / "topic-models" / variant / "seed-42/review-segments.parquet"
        data = pd.read_parquet(path)
        r = data[
            (data.episode_id.astype(str) == eid)
            & (data.start_seconds <= t)
            & (data.end_seconds > t)
        ].iloc[0]
        row[variant + "_segment_key"] = r.segment_key
        row[variant + "_start_seconds"] = r.start_seconds
        row[variant + "_end_seconds"] = r.end_seconds
        row[variant + "_topic_key"] = r.cluster_topic_key
        row[variant + "_keyword_label"] = r.cluster_topic_name
        row[variant + "_membership_strength"] = r.cluster_membership_strength
    rows.append(row)
pd.DataFrame(rows).to_csv(base / "qualitative-examples.csv", index=False)
print(
    pd.DataFrame(rows)[
        ["episode_id", "original_keyword_label", "sentence5_keyword_label", "preference"]
    ].to_string(index=False)
)
