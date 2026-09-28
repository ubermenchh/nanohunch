"""Deterministic public-gold evaluation set adapters and builder (Phase 3, step 5)."""

import hashlib
import json
import random
from pathlib import Path

import yaml

from dataset import assign_split
from fmt import Question, render


def _question_text(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith("?") else "?")


def convert_boolq(row: dict) -> dict:
    passage = row["passage"]
    return {
        "id": "boolq:" + hashlib.sha256((passage + "|" + row["question"]).encode()).hexdigest()[:20],
        "group_key": "passage:" + hashlib.sha256(passage.encode()).hexdigest()[:16],
        "qtype": "noul",
        "state": passage,
        "question": _question_text(row["question"]),
        "options": ["yes", "no"],
        "gold": 0 if row["answer"] else 1,
        "source": "boolq",
        "meta": {},
    }


def convert_arc(row: dict) -> dict:
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    if not 2 <= len(texts) <= 26 or row["answerKey"] not in labels:
        raise ValueError(f"invalid ARC row {row.get('id')}")
    return {
        "id": "arc:" + row["id"],
        "group_key": row["id"],
        "qtype": "choice",
        "state": row["question"],
        "question": "Which option answers the question in the STATE?",
        "options": list(texts),
        "gold": labels.index(row["answerKey"]),
        "source": "arc",
        "meta": {},
    }


def convert_csqa(row: dict) -> dict:
    question = row["question"]
    if isinstance(question, dict):
        # Older cached CSQA revisions nested both the stem and choices under question.
        stem = question["stem"]
        choices = question["choices"]
        labels, texts = [o["label"] for o in choices], [o["text"] for o in choices]
    else:
        # Current Hub rows store question text and the parallel choice columns separately.
        stem = question
        labels, texts = row["choices"]["label"], row["choices"]["text"]
    if len(labels) != len(texts) or row["answerKey"] not in labels:
        raise ValueError(f"invalid CommonsenseQA row {row.get('id')}")
    return {
        "id": "csqa:" + row["id"],
        "group_key": row["id"],
        "qtype": "choice",
        "state": stem,
        "question": "Which option answers the question in the STATE?",
        "options": list(texts),
        "gold": labels.index(row["answerKey"]),
        "source": "csqa",
        "meta": {},
    }


def convert_hotpot(row: dict) -> dict | None:
    if row["answer"].strip().lower() not in {"yes", "no"}:
        return None
    context = row["context"]
    state = "".join(
        f"{title}\n{' '.join(sentences)}\n\n"
        for title, sentences in zip(context["title"], context["sentences"], strict=True)
    )
    row_id = row["_id"] if "_id" in row else row["id"]
    return {
        "id": "hotpot:" + row_id,
        "group_key": row_id,
        "qtype": "noul",
        "state": state,
        "question": _question_text(row["question"]),
        "options": ["yes", "no"],
        "gold": 0 if row["answer"].strip().lower() == "yes" else 1,
        "source": "hotpot",
        "meta": {},
    }


def _row_hash(row: dict, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}|{row['source']}|{row['group_key']}".encode()).digest()


def _length_bucket(n: int) -> str:
    return "le512" if n <= 512 else "512_2k" if n <= 2048 else "2k_4k" if n <= 4096 else "gt4k"


def _n_tokens(tokenizer, row: dict) -> int:
    q = Question(row["id"], row["qtype"], row["question"], tuple(row["options"]))
    rendered = render(tokenizer, row["state"], [q], n_perms=1, max_context=8192)
    return len(rendered.prefix_ids) + len(rendered.branches[0].token_ids)


def _permutations(row: dict) -> None:
    n = len(row["options"])
    if row["qtype"] != "choice" or n < 3:
        return
    for k in (1, 2, 3):
        rng = random.Random(f"{k}|{row['id']}")
        perm = list(range(n))
        while perm == list(range(n)) or perm == list(reversed(range(n))):
            perm = rng.sample(range(n), n)
        row["meta"][f"perm_seed_{k}"] = ",".join(map(str, perm))


def _pad_hotpot(rows: list[dict], tokenizer, targets: tuple[int, ...]) -> list[dict]:
    source_rows = [r for r in rows if r["source"] == "hotpot"]
    # Reuse bases only if each can fit the shortest target without dropping evidence.
    bases = [r for r in source_rows if _n_tokens(tokenizer, r) <= min(targets)]
    if len(bases) < 150:
        raise ValueError(f"need 150 short Hotpot bases, found {len(bases)}")
    variants = []
    for base in bases[:150]:
        for target in targets:
            candidates = [r for r in source_rows if r["group_key"] != base["group_key"]]
            random.Random(f"pad|{base['group_key']}|{target}").shuffle(candidates)
            padded = dict(base, id=f"{base['id']}:pad{target}", meta={**base["meta"], "template": f"pad{target}"})
            state, count = base["state"], _n_tokens(tokenizer, base)
            lower = {1900: 512, 3900: 2048, 7600: 4096}[target]
            if not lower < count <= target:
                for candidate in candidates:
                    trial = state + candidate["state"]
                    padded["state"] = trial
                    trial_count = _n_tokens(tokenizer, padded)
                    if trial_count > target:
                        continue
                    state, count = trial, trial_count
                    if count > lower:
                        break
            if not lower < count <= target:
                raise ValueError(f"could not pad {base['id']} into ({lower}, {target}]")
            padded["state"] = state
            variants.append(padded)
    return variants


def _select(rows: list[dict], split: str, source: str, quota: int, *, salt: str, fractions: dict) -> list[dict]:
    candidates = [r for r in rows if assign_split(r["group_key"], r["source"], fractions, salt) == split]
    candidates.sort(key=lambda r: (_row_hash(r, salt), r["id"]))
    if len(candidates) < quota:
        raise ValueError(f"not enough {source} rows for {split}: need {quota}, found {len(candidates)}")
    return candidates[:quota]


def build_eval_v1(cfg: dict) -> dict[str, int]:
    """Download, convert, split, pad and write the deterministic cal/test public-gold suite."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    split_cfg = yaml.safe_load((root / cfg["split_config"]).read_text())
    fractions, salt = split_cfg["fractions"], split_cfg["salt"]
    pools = {"boolq": [], "arc": [], "csqa": [], "hotpot": []}
    boolq_ds = load_dataset("google/boolq")
    csqa_ds = load_dataset("tau/commonsense_qa")
    hotpot_ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    for split in ("train", "validation"):
        pools["boolq"].extend(convert_boolq(r) for r in boolq_ds[split])
        pools["csqa"].extend(convert_csqa(r) for r in csqa_ds[split])
        pools["hotpot"].extend(r for row in hotpot_ds[split] if (r := convert_hotpot(row)) is not None)
    for config in ("ARC-Easy", "ARC-Challenge"):
        arc = load_dataset("allenai/ai2_arc", config)
        for split in arc:
            pools["arc"].extend(convert_arc(row) for row in arc[split])
    tokenizer = AutoTokenizer.from_pretrained(cfg["reference_tokenizer"])
    selected = {"cal": [], "test": []}
    for split in selected:
        for source, quota in cfg["quotas"][split].items():
            if source == "hotpot_pad":
                continue
            selected[split].extend(_select(pools[source], split, source, quota, salt=salt, fractions=fractions))
    pads = _pad_hotpot(selected["test"], tokenizer, (1900, 3900, 7600))
    selected["test"].extend(pads)
    if len(pads) != cfg["quotas"]["test"].get("hotpot_pad", len(pads)):
        raise ValueError(f"hotpot padding produced {len(pads)} variants")
    out = root / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    counts, digests = {}, {}
    for split, rows in selected.items():
        for row in rows:
            row["meta"]["n_tokens"] = _n_tokens(tokenizer, row)
            row["meta"]["length_bucket"] = _length_bucket(row["meta"]["n_tokens"])
            _permutations(row)
            row["state_id"] = hashlib.sha256(row["state"].encode()).hexdigest()
            row["split"] = split
        rows.sort(key=lambda r: (r["source"], r["id"]))
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        path = out / f"{split}.jsonl"
        path.write_text(payload)
        digests[split] = hashlib.sha256(payload.encode()).hexdigest()
        counts[split] = {
            source: sum(r["source"] == source for r in rows) for source in sorted({r["source"] for r in rows})
        }
    (out / "manifest.json").write_text(
        json.dumps({"split_salt": salt, "counts": counts, "sha256": digests}, indent=2, sort_keys=True) + "\n"
    )
    return {split: len(rows) for split, rows in selected.items()}
