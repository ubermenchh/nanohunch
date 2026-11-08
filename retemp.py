# Change the temperature of saved OpenDecisions predictions without running the model.
# log p = z / T_old - const, so softmax(log p * T_old / T_new) is the prediction at T_new.
#   uv run python retemp.py adapter_decider_n4000.npz     reads runs/od_<stem>.jsonl, writes runs/od_<stem>_pertype.jsonl
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

stem = Path(sys.argv[1]).stem
cfg = json.load(open(Path(sys.argv[1]).with_suffix(".json")))
kind = {}  # id -> question type
for line in open(hf_hub_download("jgeuter/OpenDecisions", "opendecisions.jsonl", repo_type="dataset", revision="main")):
    r = json.loads(line)
    kind[r["id"]] = r["question"]["type"]
with open(f"runs/od_{stem}_pertype.jsonl", "w") as out:
    for line in open(f"runs/od_{stem}.jsonl"):
        r = json.loads(line)
        z = np.log(np.array(list(r["probs"].values()))) * cfg["temp"] / cfg["temp_by_type"][kind[r["id"]]]
        p = np.exp(z - z.max())
        out.write(json.dumps({"id": r["id"], "probs": dict(zip(r["probs"], (p / p.sum()).tolist(), strict=True))}) + "\n")
