import json
import random
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from datasets import load_dataset

LETTERS = "ABCDEFGH"

@dataclass(frozen=True)
class Item:
    state: str # text to read
    question: str
    options: tuple[str, ...]
    gold: int # index of the correct option in `options`

def shuffled(item: Item, perm: tuple[int, ...]) -> Item:
    return replace(item, options=tuple(item.options[i] for i in perm), gold=perm.index(item.gold))

def prompt(item: Item) -> str:
    options = [f"{LETTERS[j]}. {o}" for j, o in enumerate(item.options)]
    return "\n".join([item.state, "", f"Question: {item.question}", *options, "Answer:"])

def encode(tok, item: Item) -> tuple[list[int], list[int]]:
    ids = tok.encode(prompt(item))
    label_ids = [tok.encode(f" {LETTERS[j]}") for j in range(len(item.options))]
    assert all(len(t) == 1 for t in label_ids), label_ids
    return ids, [t[0] for t in label_ids]

def from_boolq(row: dict) -> Item:
    return Item(row["passage"], row["question"].capitalize() + "?", ("Yes", "No"), 0 if row["answer"] else 1)

def from_arc(row: dict) -> Item:
    c = row["choices"]
    return Item(row["question"], "Which option is correct?", tuple(c["text"]), c["label"].index(row["answerKey"]))

def from_jev(state, q: dict, answer) -> Item:
    # One Jev-style question. 
    # choice: criteria = {id: text}. noul: yes/no.
    # score: criteria = levels, lowest first.
    crit = q.get("criteria") or {}
    if q["type"] == "noul":
        ids = ["true", "false"]
        texts = [crit.get(k) or f"The proposition is {k}." for k in ids] 
        answer = "true" if answer in (True, "yes", "true") else "false"
    elif q["type"] == "choice":
        ids, texts = list(crit), [t or k for k, t in crit.items()]
    else:
        ids, texts = [str(i) for i in range(len(crit))], list(crit)
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    options = tuple(f"{k}: {t}" for k, t in zip(ids, texts, strict=True))
    return Item(state, q["instructions"], options, ids.index(str(answer)))

# Teacher-written decision questions from github.com/Mapika/decider (Apache-2.0).
DECIDER = "https://raw.githubusercontent.com/Mapika/decider/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/teacher_data/"

def load_decider(name: str, n: int, seed: int = 0, skip: int = 0) -> list[Item]:
    path = Path("data_cache") / name
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DECIDER + name, path)
    items = [
        from_jev(row["state"], q, q["answer"])
        for line in open(path)
        for row in [json.loads(line)]
        for q in row["questions"]
        if q.get("teacher_ok", True) and q.get("teacher_p", 1.0) >= 0.5
    ]
    random.Random(seed).shuffle(items)
    return items[skip:skip+n]

SOURCES = {
    "boolq": ("google/boolq", None, from_boolq),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", from_arc),
    "csqa": ("tau/commonsense_qa", None, from_arc),
}

def load_items(name: str, split: str, n: int, seed: int = 0, skip: int = 0) -> list[Item]:
    path, config, convert = SOURCES[name]
    rows = load_dataset(path, config, split=split).shuffle(seed=seed).select(range(skip, skip + n))
    return [convert(r) for r in rows]