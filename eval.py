import numpy as np

from data import Item, shuffled

def default_perms(n: int) -> list[tuple[int, ...]]:
    perms = [tuple(range(n)), tuple(reversed(range(n))), tuple((i + 1) % n for i in range(n))]
    return list(dict.fromkeys(perms))

def evaluate(logits_fn, items: list[Item], perms_fn=default_perms) -> dict:
    hits = first = flips = 0
    for item in items:
        ps = [probs(logits_fn, item, [perm]) for perm in perms_fn(len(item.options))]
        hits += np.mean(ps, axis=0).argmax() == item.gold
        first += ps[0].argmax() == item.gold
        flips += len({int(p.argmax()) for p in ps}) > 1
    n = len(items)
    return {"acc": hits / n, "acc_one_order": first / n, "flip": flips / n}

def pool(perms, zs, temp: float = 1.0) -> np.ndarray:
    # Avg the answer probs over severl option orders, in the original option order
    total = np.zeros(len(perms[0]))
    for perm, z in zip(perms, zs, strict=True):
        # temp > 1, flattens the probs
        # temp < 1, sharpens
        z = np.array(z, dtype=np.float64) / temp
        p = np.exp(z - z.max())
        p /= p.sum()
        total[list(perm)] += p
    return total / len(perms)

def probs(logits_fn, item: Item, perms, temp: float = 1.0) -> np.ndarray:
    return pool(perms, [logits_fn(shuffled(item, perm)) for perm in perms], temp)

def ece(conf, hit, bins: int = 10) -> float:
    # expected calibration error: in each confidence bin, the gap between mean confidence and accuracy
    conf, hit = np.asarray(conf, dtype=np.float64), np.asarray(hit, dtype=np.float64)
    b = np.minimum((conf * bins).astype(int), bins - 1) # confidence 1.0 gies in the last bin
    return sum((b == k).mean() * abs(hit[b == k].mean() - conf[b == k].mean()) for k in range(bins) if (b == k).any())

def collect(logits_fn, items: list[Item], perms_fn=default_perms) -> list:
    # run model once and keep the logits, so many temps can be tried without running it again
    cases = []
    for item in items:
        perms = perms_fn(len(item.options))
        cases.append((perms, [np.array(logits_fn(shuffled(item, p))) for p in perms], item.gold))
    return cases

def fit_temperature(cases, grid=np.geomspace(0.1, 10, 200)) -> float:
    # pick the T with the lowest average loss: minus the log probability of the gold option
    nll = [np.mean([-np.log(max(pool(p, z, t)[g], 1e-12)) for p, z, g in cases]) for t in grid]
    best = int(np.argmin(nll))
    if best in (0, len(grid) - 1): # best value on the edge means the search range is wrong, or the data is
        raise ValueError(f"temperature on the search edge: {grid[best]:.3f}")
    return float(grid[best])