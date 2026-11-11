# Refit the temperature of a run_decider.py adapter on HARDER held-out decider questions (teacher_p < 0.8, not trained on).
# The first fit used random held-out questions, which are mostly easy (median teacher_p .95), and OpenDecisions is not.
#   uv run python refit_temp.py adapter_decider_n4000.npz [--n 4000] [--m 300]
import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from data import encode, from_jev
from eval import collect, fit_temperature
from model import add_lora, answer_logits, load_model

ADAPTER = Path(sys.argv[1])
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4000  # questions the adapter trained on
M = int(sys.argv[sys.argv.index("--m") + 1]) if "--m" in sys.argv else 300  # hard held-out questions to fit each temperature on
CFG = json.load(open(ADAPTER.with_suffix(".json")))

# the same filter and shuffle as data.load_decider, keeping teacher_p
rows = [
    (from_jev(r["state"], q, q["answer"]), q["teacher_p"], q["type"])
    for line in open("data_cache/custom_questions.jsonl")
    for r in [json.loads(line)]
    for q in r["questions"]
    if q.get("teacher_p", 1.0) >= 0.5
]
random.Random(0).shuffle(rows)
held = rows[N:]  # never trained on
hard = [it for it, p, _ in held if p < 0.8][:M]
easy = [it for it, _, _ in held][:300]  # what run_decider.py fitted on (same first 300 after N)

model, tok = load_model(CFG["model"])
add_lora(model, CFG["rank"], CFG["alpha"], CFG["layers"], tuple(CFG["targets"]))
model.load_weights(list(mx.load(str(ADAPTER)).items()), strict=False)
one = lambda n: [tuple(range(n))]  # noqa: E731
fit = lambda items: fit_temperature(collect(lambda it: answer_logits(model, *encode(tok, it)), items, one))  # noqa: E731
T_easy, T_hard = fit(easy), fit(hard)
print(f"temperature on {len(easy)} random held-out {T_easy:.3f}, on {len(hard)} hard held-out {T_hard:.3f}")
# one temperature per question type, each fitted on up to M hard held-out questions of that type
by_type = {t: fit([it for it, p, ty in held if p < 0.8 and ty == t][:M]) for t in ("choice", "noul", "score")}
print("per type", {t: round(T, 3) for t, T in by_type.items()})
CFG |= dict(temp_easy=CFG.get("temp_easy", CFG["temp"]), temp=T_hard, temp_by_type=by_type)
json.dump(CFG, open(ADAPTER.with_suffix(".json"), "w"))
