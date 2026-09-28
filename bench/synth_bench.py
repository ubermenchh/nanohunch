"""Synthetic-weight speed benchmark on the M5. Builds the text model from the real HF
config.json with RANDOM bf16 weights (speed does not depend on weight values), then
times prefill, branched tails, and a LoRA fwd+bwd step with a candidate-rows-only loss.

usage: python synth_bench.py <config.json> <arch: qwen35|llama> <mode: prefill|branch|train> <S> [B] [ckpt]
"""
import json, sys, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
from mlx_lm.models.cache import KVCache, ArraysCache
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint

cfg_path, arch, mode, S = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
B = int(sys.argv[5]) if len(sys.argv) > 5 else 1
ckpt = len(sys.argv) > 6 and sys.argv[6] == "ckpt"
cfg = json.load(open(cfg_path))
if arch == "qwen35":
    from mlx_lm.models import qwen3_5 as M
    tc = cfg["text_config"]
    model = M.TextModel(M.TextModelArgs.from_dict(tc))
    backbone, V, H = model.model, tc["vocab_size"], tc["hidden_size"]
    head_w = lambda: backbone.embed_tokens.weight
else:
    from mlx_lm.models import llama as M
    model = M.Model(M.ModelArgs.from_dict(cfg))
    backbone, V, H = model.model, cfg["vocab_size"], cfg["hidden_size"]
    head_w = lambda: (model.lm_head.weight if hasattr(model, "lm_head") else backbone.embed_tokens.weight)
model.update(tree_map(lambda p: (p * 0.02).astype(mx.bfloat16) if p.dtype == mx.float32 else p, model.parameters()))
mx.eval(model.parameters())
nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
mx.random.seed(0)
state = mx.random.randint(1000, 30000, (1, S))
letters = mx.array([10, 20, 30, 40])

def med(fn, n):
    fn(); ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    ts.sort(); return ts[len(ts) // 2]

import os
CH = int(os.environ.get("CHUNK", "0")) or S
def prefill():
    c = model.make_cache() if hasattr(model, "make_cache") else [KVCache() for _ in backbone.layers]
    for i in range(0, S, CH):
        h = backbone(state[:, i:i+CH], cache=c)
        mx.eval(h, [x.state for x in c])
    return c

if mode == "prefill":
    model.eval()
    mx.reset_peak_memory()
    t = med(prefill, 3)
    print(f"{cfg_path.split('/')[-1]} params={nparams/1e9:.2f}B prefill S={S}: {t*1e3:.0f} ms = {S/t:.0f} tok/s peak={mx.get_peak_memory()/1e9:.2f} GB")

elif mode == "branch":
    model.eval()
    T = 32
    tails = mx.random.randint(1000, 30000, (B, T))
    c0 = prefill()
    def branch_and_score():
        nc = []
        for x in c0:
            if isinstance(x, KVCache):
                k, v = x.state; y = KVCache(); y.state = (mx.repeat(k, B, 0), mx.repeat(v, B, 0))
            else:
                y = ArraysCache(size=2); y[0] = mx.repeat(x[0], B, 0); y[1] = mx.repeat(x[1], B, 0)
            nc.append(y)
        h = backbone(tails, cache=nc)[:, -1, :]
        p = mx.softmax((h @ head_w()[letters].T).astype(mx.float32), -1)
        mx.eval(p); return nc
    mx.reset_peak_memory()
    t_pre = med(prefill, 3)
    t_br = med(branch_and_score, 3)
    kv = sum(x.nbytes for x in c0 if isinstance(x, KVCache))
    rec = sum(x[0].nbytes + x[1].nbytes for x in c0 if not isinstance(x, KVCache))
    print(f"{cfg_path.split('/')[-1]} S={S} B={B} T={T}: prefill {t_pre*1e3:.0f} ms, branch+tails {t_br*1e3:.0f} ms, "
          f"total {(t_pre+t_br)*1e3:.0f} ms; per-branch prefix KV {kv/1e6:.1f} MB, per-branch recurrent+conv {rec/1e6:.1f} MB; peak {mx.get_peak_memory()/1e9:.2f} GB")

elif mode == "train":
    model.freeze()
    layers = backbone.layers
    linear_to_lora_layers(backbone, len(layers), {"rank": 16, "scale": 2.0, "dropout": 0.0})
    if ckpt:
        grad_checkpoint(layers[0])
    model.train()
    ntrain = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    target = mx.array([1])
    def loss_fn(m):
        h = backbone(state)[:, -1, :]                       # gather answer position first
        logits = (h @ head_w()[letters].T).astype(mx.float32)  # candidate rows only
        return nn.losses.cross_entropy(logits, target).mean()
    opt = optim.Adam(learning_rate=1e-5)
    lg = nn.value_and_grad(model, loss_fn)
    def step():
        l, g = lg(model); opt.update(model, g); mx.eval(l, model.parameters(), opt.state)
    mx.reset_peak_memory()
    t = med(step, 2)
    print(f"{cfg_path.split('/')[-1]} TRAIN S={S} ckpt={ckpt} lora_r16_params={ntrain/1e6:.1f}M: {t:.2f} s/step = {S/t:.0f} tok/s peak={mx.get_peak_memory()/1e9:.2f} GB")
