from dataclasses import dataclass, replace

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

SOURCES = {
    "boolq": ("google/boolq", None, from_boolq),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", from_arc),
    "csqa": ("tau/commonsense_qa", None, from_arc),
}

def load_items(name: str, split: str, n: int, seed: int = 0, skip: int = 0) -> list[Item]:
    path, config, convert = SOURCES[name]
    rows = load_dataset(path, config, split=split).shuffle(seed=seed).select(range(skip, skip + n))
    return [convert(r) for r in rows]