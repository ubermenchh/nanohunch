# Ask the adapter on Hugging Face a typed question: a state and a question in, one probability per answer out.
#   uv run python ask.py        runs three examples; import ask() to use your own
import json

import mlx.core as mx
from huggingface_hub import hf_hub_download

from data import encode, from_jev
from model import add_lora, answer_logits, load_model

REPO = "ubermenchh/nanohunch-decider-4b"
cfg = json.load(open(hf_hub_download(REPO, "config.json")))
model, tok = load_model(cfg["model"])
add_lora(model, cfg["rank"], cfg["alpha"], cfg["layers"], tuple(cfg["targets"]))
model.load_weights(list(mx.load(hf_hub_download(REPO, "adapter.npz")).items()), strict=False)


def ask(state, q: dict) -> dict:
    # q: {"type": "choice" | "noul" | "score", "instructions": "...", "criteria": ...}
    #   choice: criteria = {"name": "description", ...}   noul: yes/no, criteria None (or {"true": ..., "false": ...})
    #   score:  criteria = ["description of level 0", "level 1", ...]
    first = {"choice": next(iter(q.get("criteria") or {}), None), "noul": "yes", "score": 0}[q["type"]]
    item = from_jev(state, q, first)  # the answer is only needed to build the item; the model never sees it
    ids, labels = encode(tok, item)
    z = answer_logits(model, ids, labels) / cfg["temp_by_type"][q["type"]]
    p = mx.softmax(z).tolist()
    return {o.split(":")[0]: round(x, 3) for o, x in zip(item.options, p, strict=True)}


if __name__ == "__main__":
    print(ask("Here are three expense lines from my trip. Turn them into a JSON list with fields date, amount, category.",
              {"type": "choice", "instructions": "Which model should handle this request?",
               "criteria": {"chitchat": "casual conversation", "extraction_json": "pull structured fields out of text",
                            "data_analysis": "compute statistics over data"}}))
    print(ask("Customer: my parcel arrived broken and I want my money back today.",
              {"type": "noul", "instructions": "Should this ticket be escalated to a human agent?"}))
    print(ask("Review: the battery lasts a day, but the screen scratches easily and support never answered.",
              {"type": "score", "instructions": "How satisfied is the reviewer?",
               "criteria": ["very unhappy", "unhappy", "neutral", "happy", "very happy"]}))
