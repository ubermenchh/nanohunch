import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from data import Item, encode, shuffled
from model import answer_logits

def loss_fn(model, ids: list[int], label_ids: list[int], gold: int, brier: float = 0.0) -> mx.array:
    logits = answer_logits(model, ids, label_ids)
    ce = nn.losses.cross_entropy(logits[None], mx.array([gold])).mean()
    onehot = (mx.arange(len(label_ids)) == gold).astype(logits.dtype)
    return ce + brier * ((mx.softmax(logits) - onehot) ** 2).sum()

def train(
    model, 
    tok, 
    items: list[Item], 
    lr: float, 
    accum: int = 8, 
    seed: int = 0, 
    shuffle: bool = True,
    on_step = None,
    brier: float = 0.0,
) -> list[float]:
    opt = optim.AdamW(learning_rate=lr)
    grad_fn = nn.value_and_grad(model, loss_fn)
    rng = random.Random(seed)
    grads_sum, losses, log = None, [], []
    for item in items:
        n = len(item.options)
        perm = tuple(rng.sample(range(n), n)) if shuffle else tuple(range(n))
        s = shuffled(item, perm)
        ids, label_ids = encode(tok, s)
        loss, grads = grad_fn(model, ids, label_ids, s.gold, brier)
        grads_sum = grads if grads_sum is None else tree_map(mx.add, grads_sum, grads)
        mx.eval(loss, grads_sum)
        losses.append(loss.item())
        if len(losses) == accum:
            opt.update(model, tree_map(lambda g: g / accum, grads_sum))
            mx.eval(model.parameters(), opt.state)
            log.append(sum(losses) / accum)
            if on_step:
                on_step(len(log), log[-1])
            grads_sum, losses = None, []
    return log