"""Phase 1 step 4: reference metrics for the walking skeleton.

These are the numbers Phase 3 `calibrate.accuracy`, `calibrate.ece(..., bins=15, scheme="width")`
and `calibrate.flip_rate` must reproduce exactly on the same inputs.

usage: uv run python -m skeleton.tiny_eval --name b0|lora|... --fwd FWD.json --rev REV.json [--json reports/skeleton.json]
"""

import argparse
import json
import pathlib

import numpy as np


def accuracy(probs, gold) -> float:
    return float(np.mean([int(np.argmax(p)) == g for p, g in zip(probs, gold, strict=True)]))


def ece15(conf, correct) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * 15).astype(int), 14)  # bin 14 includes conf == 1.0
    return float(
        sum(
            abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
            for b in range(15)
            if (idx == b).any()
        )
    )


def flip_rate(fwd_top1, rev_top1) -> float:
    return float(np.mean([a != b for a, b in zip(fwd_top1, rev_top1, strict=True)]))


def block(fwd: list[dict], rev: list[dict]) -> dict:
    by_id = {r["id"]: r for r in rev}
    out = {}
    for name, rows in [
        ("noul", [r for r in fwd if r["type"] == "noul"]),
        ("choice", [r for r in fwd if r["type"] == "choice"]),
        ("all", fwd),
    ]:
        probs, gold = [r["probs"] for r in rows], [r["gold_index"] for r in rows]
        conf = [max(p) for p in probs]
        correct = [int(np.argmax(p)) == g for p, g in zip(probs, gold, strict=True)]
        m = {"n": len(rows), "acc": accuracy(probs, gold), "ece15": ece15(conf, correct)}
        if name == "choice":
            m["flip"] = flip_rate(
                [int(np.argmax(p)) for p in probs], [int(np.argmax(by_id[r["id"]]["probs"])) for r in rows]
            )
        out[name] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fwd", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fwd, rev = json.load(open(a.fwd)), json.load(open(a.rev))
    assert not any(r["reversed"] for r in fwd) and all(r["reversed"] for r in rev), "fwd/rev files swapped"
    res = block(fwd, rev)
    print(f"{'name':6s} {'type':6s} {'n':>5s} {'acc':>6s} {'ece15':>6s} {'flip':>6s}")
    for t, m in res.items():
        flip = f"{m['flip']:6.3f}" if "flip" in m else "     -"
        print(f"{a.name:6s} {t:6s} {m['n']:5d} {m['acc']:6.3f} {m['ece15']:6.3f} {flip}")
    if a.json:
        p = pathlib.Path(a.json)
        data = json.loads(p.read_text()) if p.exists() else {}
        data[a.name] = res
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2) + "\n")
        print(f"merged '{a.name}' into {p}")


if __name__ == "__main__":
    main()
