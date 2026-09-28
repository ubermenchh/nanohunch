"""Phase 1 step 1: 1,000-row gold train and eval sets for the walking skeleton.

BoolQ (Noul, train + validation pooled) and ARC Easy + Challenge (Choice, 3 to 5 options, all splits
pooled), split by group key with a throwaway salt (not the Phase 3 frozen salt). Per split: 500 BoolQ,
250 ARC-Easy, 250 ARC-Challenge. Row schema (plan Phase 1 "Produces"):
  {"id", "group_key", "type": "noul"|"choice", "state", "question", "options", "gold_index"}

usage: uv run python -m skeleton.prep_gold [--show N]
"""

import argparse
import hashlib
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "skeleton"
SALT = "nanohunch-skeleton-v0"
ARC_STATE = "Science exam question. There is no passage; answer from general knowledge.\n\n"
QUOTA = {"boolq": 500, "arc-easy": 250, "arc-challenge": 250}


def split_of(source: str, group_key: str, salt: str) -> str:
    u = int(hashlib.sha256(f"{salt}|{source}|{group_key}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if u < 0.5 else "eval"


def boolq_rows() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", revision="35b264d0")
    print("boolq columns:", ds["train"].column_names, "(no title column: group_key = passage hash)")
    rows = []
    for split in ("train", "validation"):
        for r in ds[split]:
            q = r["question"].strip()
            rows.append(
                {
                    "id": f"boolq:{len(rows)}",
                    "group_key": "passage:" + hashlib.sha256(r["passage"].encode()).hexdigest()[:16],
                    "type": "noul",
                    "state": r["passage"],
                    "question": q[0].upper() + q[1:] + ("" if q.endswith("?") else "?"),
                    "options": ["yes", "no"],
                    "gold_index": 0 if r["answer"] else 1,
                }
            )
    return rows


def arc_rows(config: str, source: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", config, revision="210d026f")
    rows = []
    for split in ds:
        for r in ds[split]:
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            if not 3 <= len(texts) <= 5 or r["answerKey"] not in labels:
                continue
            rows.append(
                {
                    "id": f"{source}:{r['id']}",
                    "group_key": r["id"],
                    "type": "choice",
                    "state": ARC_STATE + r["question"],
                    "question": "Which option correctly answers the question in the STATE?",
                    "options": list(texts),
                    "gold_index": labels.index(r["answerKey"]),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    pools = {"boolq": boolq_rows(), "arc-easy": arc_rows("ARC-Easy", "arc-easy")}
    pools["arc-challenge"] = arc_rows("ARC-Challenge", "arc-challenge")
    out = {"train": [], "eval": []}
    for source, rows in pools.items():
        for split in out:
            cand = [r for r in rows if split_of(source, r["group_key"], SALT) == split]
            out[split] += random.Random(0).sample(cand, QUOTA[source])
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        (OUT / f"gold_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shared = {r["group_key"] for r in out["train"]} & {r["group_key"] for r in out["eval"]}
    print(f"train={len(out['train'])} eval={len(out['eval'])} shared_group_keys={len(shared)}")
    for r in random.Random(1).sample(out["eval"], a.show) if a.show else []:  # R6: read these
        print(f"--- {r['id']} [{r['type']}] {r['state'][:110]!r}")
        print(f"    Q: {r['question']}  options={r['options']}  GOLD -> {r['options'][r['gold_index']]!r}")


if __name__ == "__main__":
    main()
