# Score a model on OpenDecisions (huggingface.co/datasets/jgeuter/OpenDecisions), a held-out set of 2,352
# decision questions. Never train or fit a temperature on it.
#   uv run python od.py                                     base model, no training
#   uv run python od.py --adapter adapter_n900.npz          base model + a LoRA saved by run.py
#   uv run python od.py --model Qwen/Qwen3-4B-Instruct-2507
# One forward pass per question with the options in their given order, as on the OpenDecisions leaderboard.
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from data import encode, from_jev
from model import add_lora, answer_logits, load_model


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ADAPTER = arg("--adapter", None)
# An adapter from run_decider.py has a .json next to it with its LoRA shape and temperature. Older ones used these.
CFG = dict(rank=8, alpha=160, layers=8, targets=["self_attn.q_proj", "self_attn.v_proj"], temp=1.0)
if ADAPTER and Path(ADAPTER).with_suffix(".json").exists():
    CFG = json.load(open(Path(ADAPTER).with_suffix(".json")))
LIMIT = int(arg("--limit", 0)) or None  # only the first N questions, for a quick check
MODEL = arg("--model", CFG.get("model", "Qwen/Qwen3-4B-Base"))
OUT = Path("runs") / f"od_{Path(ADAPTER).stem if ADAPTER else MODEL.split('/')[-1]}.jsonl"


def get(name):
    return hf_hub_download("jgeuter/OpenDecisions", name, repo_type="dataset", revision="main")


def to_probs(row, ids, p):
    by_id = {k: float(x) for k, x in zip(ids, p, strict=True)}
    return {"yes": by_id["true"], "no": by_id["false"]} if row["question"]["type"] == "noul" else by_id


rows = [json.loads(line) for line in open(get("opendecisions.jsonl"), encoding="utf-8")][:LIMIT]
model, tok = load_model(MODEL)
if ADAPTER:
    add_lora(model, CFG["rank"], CFG["alpha"], CFG["layers"], tuple(CFG["targets"]))
    model.load_weights(list(mx.load(ADAPTER).items()), strict=False)

OUT.parent.mkdir(exist_ok=True)
hits = {"choice": [], "noul": [], "score": []}
with open(OUT, "w") as f:
    for i, row in enumerate(rows):
        item = from_jev(row["state"], row["question"], row["expected"])
        ids = [o.split(":")[0] for o in item.options]
        z = np.array(answer_logits(model, *encode(tok, item)), dtype=np.float64)
        z = z / CFG.get("temp_by_type", {}).get(row["question"]["type"], CFG["temp"])
        p = np.exp(z - z.max())
        f.write(json.dumps({"id": row["id"], "probs": to_probs(row, ids, p / p.sum())}) + "\n")
        if i % 200 == 0:
            print(f"{i}/{len(rows)}", flush=True)
        hits[row["question"]["type"]].append(int(np.argmax(p)) == item.gold)

# Plain accuracy per question type. Capability (the leaderboard number) comes from the dataset's own score.py.
print(f"wrote {OUT}")
for t, h in ((t, h) for t, h in hits.items() if h):
    print(f"  {t:7s} accuracy {100 * np.mean(h):5.1f}%  ({len(h)} questions)")
