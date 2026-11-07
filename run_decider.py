# Train on decision-shaped questions (decider's teacher data) plus a little public replay, then score on OpenDecisions
# with od.py. Run: uv run python run_decider.py [--small] [--n 4000] [--seed S] [--lr 1e-4]
# Start: the instruct checkpoint. LoRA on all 7 projections of all layers, alpha/rank = 2, warmup + cosine decay.
import json
import random
import sys
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

import wandb
from data import encode, load_decider, load_items
from eval import collect, fit_temperature
from model import add_lora, answer_logits, load_model
from train import train


def arg(name, default):  # e.g. --seed 1
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SMALL = "--small" in sys.argv
SEED, N, LR = arg("--seed", 0), arg("--n", 4000), arg("--lr", 1e-4)
MODEL = arg("--model", "Qwen/Qwen3-4B-Instruct-2507")
RANK, ALPHA, ACCUM, REPLAY, N_CAL = 16, 32, 32, 270, 300  # REPLAY: public questions per dataset
TARGETS = [f"self_attn.{p}_proj" for p in "qkvo"] + [f"mlp.{p}_proj" for p in ("gate", "up", "down")]
if SMALL:
    N, REPLAY, N_CAL, ACCUM = 64, 4, 20, 8
TAG = f"decider_n{N}" + (f"_s{SEED}" if SEED else "")
FILE = "custom_questions.jsonl"

steps = (N + 3 * REPLAY) // ACCUM
warmup = max(1, steps // 10)
schedule = optim.join_schedules(
    [optim.linear_schedule(0.0, LR, warmup), optim.cosine_decay(LR, steps - warmup)], [warmup]
)
wandb.init(
    project="nanohunch",
    mode="disabled" if SMALL else "online",
    name=TAG,
    config=dict(model=MODEL, n=N, replay=REPLAY, lr=LR, seed=SEED, rank=RANK, alpha=ALPHA, accum=ACCUM, steps=steps),
)
model, tok = load_model(MODEL)

# The temperature is fitted on decider questions after the training ones, never on OpenDecisions.
cal = load_decider(FILE, N_CAL, skip=N)
items = load_decider(FILE, N) + [it for d in ("boolq", "arc", "csqa") for it in load_items(d, "train", REPLAY)]
random.Random(SEED).shuffle(items)

mx.random.seed(SEED)
add_lora(model, rank=RANK, alpha=ALPHA, num_layers=len(model.layers), targets=tuple(TARGETS))
recent = []


def on_step(step, loss):
    recent.append(loss)
    wandb.log({"train/loss": loss, "train/loss_avg10": sum(recent[-10:]) / len(recent[-10:]), "lr": float(schedule(step))}, step=step)


t = time.time()
log = train(model, tok, items, lr=schedule, accum=ACCUM, seed=SEED, on_step=on_step)
print(f"trained on {len(items)} questions, {len(log)} steps, {time.time() - t:.0f}s, loss {log[0]:.3f} -> {log[-1]:.3f}")
mx.savez(f"adapter_{TAG}.npz", **dict(tree_flatten(model.trainable_parameters())))

# One pass per question, as od.py scores, so the temperature fitted here is the one od.py applies.
one_order = lambda n: [tuple(range(n))]  # noqa: E731
T = fit_temperature(collect(lambda it: answer_logits(model, *encode(tok, it)), cal, one_order))
print(f"temperature {T:.3f}")
with open(f"adapter_{TAG}.json", "w") as f:
    json.dump(dict(model=MODEL, rank=RANK, alpha=ALPHA, layers=len(model.layers), targets=TARGETS, temp=T), f)
wandb.finish()
