"""Download the source datasets and write the probe pools.

    python -m wsr.prepare_data --out data

Writes data/math.jsonl (MATH training problems), data/mc.jsonl (MMLU-Pro), data/syco.jsonl (MMLU-Pro items with a
suggested answer, disjoint from mc.jsonl) and data/strongreject.jsonl. The probe sets themselves are drawn from these
pools with a fixed seed in measure.probe_items().
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

from .grading import LETTERS, last_boxed

MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def write(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{path}: {len(items)}")


def mc_item(i, q, options, ans_letter, src):
    return {"id": f"{src}-{i}", "task": "mc", "question": q, "options": options, "answer": ans_letter}


def suggested_answer_variants(items: list[dict], rng: random.Random) -> list[dict]:
    """Each item -> one copy with a wrong suggested answer; one in four items also gets a right-answer control."""
    out = []
    for it in items:
        wrong = rng.choice([LETTERS[k] for k in range(len(it["options"])) if LETTERS[k] != it["answer"]])
        out.append({**it, "id": it["id"] + "-w", "task": "syco", "hint": wrong, "hint_type": "wrong"})
        if rng.random() < 0.25:
            out.append({**it, "id": it["id"] + "-r", "task": "syco", "hint": it["answer"], "hint_type": "right"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    out, rng = Path(a.out), random.Random(a.seed)

    # MMLU-Pro: disjoint slices for the suggested-answer pool and the multiple-choice pool.
    # The order of random draws below is kept fixed so that the resulting pools are reproducible.
    mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    mmlu = [mc_item(i, r["question"], r["options"], r["answer"], "mmlupro") for i, r in enumerate(mp)]
    rng.shuffle(mmlu)
    s1, s2, s3, mc = mmlu[:2000], mmlu[2000:2300], mmlu[2300:2800], mmlu[2800:3800]
    suggested_answer_variants(s1, rng)
    syco = suggested_answer_variants(s2, rng)
    suggested_answer_variants(s3, rng)
    write(out / "syco.jsonl", syco)
    write(out / "mc.jsonl", mc)

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_items = [r for r in arc if r["answerKey"] in r["choices"]["label"]]
    rng.shuffle(arc_items)
    suggested_answer_variants([{"id": str(i), "options": r["choices"]["text"],
                                "answer": LETTERS[r["choices"]["label"].index(r["answerKey"])]}
                               for i, r in enumerate(arc_items[:500])], rng)

    parts = [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in MATH_SUBJECTS]
    math_items = []
    for i, r in enumerate(concatenate_datasets(parts)):
        ans = last_boxed(r["solution"])
        if ans:
            math_items.append({"id": f"math-{i}", "task": "math", "question": r["problem"], "answer": ans})
    rng.shuffle(math_items)
    write(out / "math.jsonl", math_items[:500])

    sr = load_dataset("walledai/StrongREJECT", split="train")
    write(out / "strongreject.jsonl", [{"id": f"sr-{i}", "task": "chat", "question": r["prompt"]} for i, r in enumerate(sr)])


if __name__ == "__main__":
    main()
