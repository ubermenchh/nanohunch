# Run cell by cell in the editor ("# %%" cells), or all at once: uv run python run.py
# %% Settings and the base model (B0 = the model before any training)
import pickle
import random
import sys
import time
from math import comb
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import wandb
from data import encode, load_items
from eval import collect, ece, fit_temperature, item_stats, pool
from model import add_lora, answer_logits, load_model
from train import train

SMALL = "--small" in sys.argv  # a quick end-to-end check with a handful of questions


def arg(name, default):  # e.g. --seed 1
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


HARD = "--hard" in sys.argv  # train on the questions the base model finds hardest, out of a pool
SEED = arg("--seed", 0)  # changes the LoRA start, the order of the training questions and the option shuffles
MODEL = "Qwen/Qwen3-4B-Base"
N_TRAIN, N_EVAL, N_CAL = (6, 4, 30) if SMALL else (arg("--n", 900), 250, 200)  # questions per dataset
LR, RANK, ALPHA, LAYERS = 5e-5, 8, 160, 8
# TAG names the saved files, so runs with different settings do not overwrite each other
POOL = 20 if SMALL else 900  # with --hard: the hardest N_TRAIN of the first POOL train questions per dataset
TAG = f"n{N_TRAIN}" + ("_hard" if HARD else "") + (f"_s{SEED}" if SEED else "")
DATASETS = ("boolq", "arc", "csqa")

wandb.init(
    project="nanohunch",
    mode="disabled" if SMALL else "online",
    config=dict(
        model=MODEL,
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        n_cal=N_CAL,
        lr=LR,
        seed=SEED,
        rank=RANK,
        alpha=ALPHA,
        layers=LAYERS,
        hard=HARD,
    ),
)
model, tok = load_model(MODEL)


def logits_fn(item):
    return answer_logits(model, *encode(tok, item))


def metrics(cases, temp=1.0):
    # Accuracy, loss (NLL), calibration error (ECE) and flip rate from saved logits at one temperature.
    ps = [pool(perms, zs, temp) for perms, zs, _ in cases]
    hit = [p.argmax() == g for p, (_, _, g) in zip(ps, cases, strict=True)]
    nll = [-np.log(max(p[g], 1e-12)) for p, (_, _, g) in zip(ps, cases, strict=True)]
    flip = [
        len({int(pool([pm], [z], temp).argmax()) for pm, z in zip(perms, zs, strict=True)}) > 1
        for perms, zs, _ in cases
    ]
    return dict(acc=np.mean(hit), nll=np.mean(nll), ece=ece([p.max() for p in ps], hit), flip=np.mean(flip))


# %% Data. Calibration questions come from the train split AFTER the ones used for training,
# so the temperature is never fitted on a training question or on a validation question.
val = {name: load_items(name, "validation", N_EVAL) for name in DATASETS}
cal = [it for name in DATASETS for it in load_items(name, "train", N_CAL, skip=POOL if HARD else N_TRAIN)]
train_items = [it for name in DATASETS for it in load_items(name, "train", N_TRAIN)]
random.Random(SEED).shuffle(train_items)

# %% B0: logits on the validation and calibration questions, then fit its temperature
b0_val = {name: collect(logits_fn, items) for name, items in val.items()}
T_b0 = fit_temperature(collect(logits_fn, cal))
print(f"B0 temperature {T_b0:.3f}")

# %% With --hard: keep the questions where B0 has the highest loss (the pool is disjoint from the calibration questions)
if HARD:
    train_items = []
    for name in DATASETS:
        items = load_items(name, "train", POOL)
        nll = item_stats(collect(logits_fn, items))["nll"]
        train_items += [items[i] for i in np.argsort(-nll)[:N_TRAIN]]
        print(
            f"{name}: B0 loss on the pool {nll.mean():.3f}, on the {N_TRAIN} kept {np.sort(nll)[-N_TRAIN:].mean():.3f}"
        )
    random.Random(SEED).shuffle(train_items)

# %% Train the LoRA on public train sets, option order shuffled each time
mx.random.seed(SEED)  # the LoRA starts from the same random weights on every run
add_lora(model, rank=RANK, alpha=ALPHA, num_layers=LAYERS)
t = time.time()


def on_step(step, loss):
    # Each step's loss is the mean of only 8 questions, so it is very noisy. Also log the mean of the last 30 steps.
    recent.append(loss)
    wandb.log({"train/loss": loss, "train/loss_avg30": sum(recent[-30:]) / len(recent[-30:])}, step=step)


recent = []
log = train(model, tok, train_items, lr=LR, seed=SEED, on_step=on_step)
print(f"trained on {len(train_items)} questions in {time.time() - t:.0f}s, loss {log[0]:.3f} -> {log[-1]:.3f}")
mx.savez(f"adapter_{TAG}.npz", **dict(tree_flatten(model.trainable_parameters())))

# %% Trained model: the same steps
tr_val = {name: collect(logits_fn, items) for name, items in val.items()}
T_tr = fit_temperature(collect(logits_fn, cal))
# Save the logits, so confidence intervals (analyze.py) can be computed without running the model again.
Path("runs").mkdir(exist_ok=True)
with open(f"runs/cases_{TAG}.pkl", "wb") as f:
    pickle.dump(dict(b0=b0_val, trained=tr_val, T_b0=T_b0, T_trained=T_tr), f)
print(f"trained temperature {T_tr:.3f}")

# %% Report. Keep rule, written before looking: keep a temperature if ECE on validation falls and NLL does not rise.
rows, final = [], {"T_b0": T_b0, "T_trained": T_tr}
for name in DATASETS:
    for label, cases, T in (("b0", b0_val[name], T_b0), ("trained", tr_val[name], T_tr)):
        raw, scaled = metrics(cases), metrics(cases, T)
        keep = scaled["ece"] < raw["ece"] and scaled["nll"] <= raw["nll"]
        print(
            f"{name:6} {label:8} acc {raw['acc']:.3f}  flip {raw['flip']:.3f}  "
            f"ECE {raw['ece']:.3f} -> {scaled['ece']:.3f}  NLL {raw['nll']:.3f} -> {scaled['nll']:.3f}  keep={keep}"
        )
        rows.append([name, label, raw["acc"], raw["flip"], raw["ece"], scaled["ece"], raw["nll"], scaled["nll"]])
        final |= {f"{name}/{label}/{k}": v for k, v in raw.items()} | {f"{name}/{label}/ece_T": scaled["ece"]}
    a, b = b0_val[name], tr_val[name]
    hit = lambda cs: [pool(p, z, 1.0).argmax() == g for p, z, g in cs]  # noqa: E731
    fixed = sum(y and not x for x, y in zip(hit(a), hit(b), strict=True))
    broken = sum(x and not y for x, y in zip(hit(a), hit(b), strict=True))
    k = fixed + broken  # two-sided sign test: are fixed and broken equally likely?
    p = min(1.0, 2 * sum(comb(k, i) for i in range(min(fixed, broken) + 1)) / 2**k) if k else 1.0
    print(f"{name:6} paired: fixed {fixed} broken {broken} p={p:.3f}")
    final |= {f"{name}/fixed": fixed, f"{name}/broken": broken}
# wandb.log (not summary) puts the numbers in the run history, so the line panels have a point to draw.
cols = ["dataset", "model", "acc", "flip", "ece", "ece_T", "nll", "nll_T"]
wandb.log(final | {"results": wandb.Table(columns=cols, data=rows)})
wandb.finish()
