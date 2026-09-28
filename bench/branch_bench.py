"""Branch microbenchmark: prefill a state once, then score B question tails batched
against a copied KV cache. Compares against naive re-encode and checks equivalence."""
import sys, time, glob, os
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

path = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--Qwen--{sys.argv[1]}/snapshots/*"))[0]
S = int(sys.argv[2]); B = int(sys.argv[3]); T = 32
model, tok = load(path)
mx.random.seed(0)
state = mx.random.randint(1000, 30000, (1, S))
tails = mx.random.randint(1000, 30000, (B, T))
letters = mx.array([tok.encode(x)[0] for x in [" A", " B", " C", " D"]])

def t(fn, n=5):
    fn(); ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter() - a)
    ts.sort(); return ts[len(ts)//2]

def prefill():
    c = make_prompt_cache(model)
    out = model(state, cache=c)
    mx.eval(out, [x.state for x in c])
    return c

def branch(c):
    nc = []
    for x in c:
        k, v = x.state
        y = KVCache(); y.state = (mx.repeat(k, B, axis=0), mx.repeat(v, B, axis=0))
        nc.append(y)
    mx.eval([y.state for y in nc])
    return nc

def score(nc):
    logits = model(tails, cache=nc)[:, -1, :]
    p = mx.softmax(logits[:, letters].astype(mx.float32), axis=-1)
    mx.eval(p); return p

c0 = prefill()
t_pre = t(prefill)
t_branch = t(lambda: branch(c0))
t_score = t(lambda: score(branch(c0)))
mx.reset_peak_memory()
p_branch = score(branch(prefill()))
peak = mx.get_peak_memory() / 1e9
kv_bytes = sum(x.nbytes for x in c0) / 1e6

def naive():
    full = mx.concatenate([mx.repeat(state, B, axis=0), tails], axis=1)
    lg = model(full)[:, -1, :]
    p = mx.softmax(lg[:, letters].astype(mx.float32), axis=-1); mx.eval(p); return p
t_naive = t(naive, n=2)
p_naive = naive()
diff = mx.max(mx.abs(p_branch - p_naive)).item()
print(f"model={sys.argv[1]} S={S} B={B} T={T}")
print(f"prefill_ms={t_pre*1e3:.1f} branch_copy_ms={t_branch*1e3:.1f} branch_copy_plus_tails_ms={t_score*1e3:.1f}")
print(f"shared_total_ms={(t_pre+t_score)*1e3:.1f} naive_reencode_ms={t_naive*1e3:.1f} speedup={t_naive/(t_pre+t_score):.1f}x")
print(f"prefix_kv_MB={kv_bytes:.1f} peak_GB_shared_path={peak:.2f} max_abs_prob_diff_branch_vs_naive={diff:.2e}")
