import sys, glob, os
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache
path = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--Qwen--{sys.argv[1]}/snapshots/*"))[0]
S=int(sys.argv[2]); B=int(sys.argv[3]); dt=sys.argv[4]; T=32
model, tok = load(path)
if dt=="fp32": model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
mx.random.seed(0)
state = mx.random.randint(1000, 30000, (1, S)); tails = mx.random.randint(1000, 30000, (B, T))
letters = mx.array([tok.encode(x)[0] for x in [" A"," B"," C"," D"]])
def probs(lg): return mx.softmax(lg[:, letters].astype(mx.float32), axis=-1)
c = make_prompt_cache(model); model(state, cache=c)
nc=[]
for x in c:
    k,v=x.state; y=KVCache(); y.state=(mx.repeat(k,B,axis=0),mx.repeat(v,B,axis=0)); nc.append(y)
pb = probs(model(tails, cache=nc)[:, -1, :])
ref = mx.concatenate([probs(model(mx.concatenate([state, tails[i:i+1]], axis=1))[:, -1, :]) for i in range(B)])
full = probs(model(mx.concatenate([mx.repeat(state,B,axis=0), tails],axis=1))[:, -1, :])
print(f"{sys.argv[1]} S={S} B={B} {dt}: branch_vs_isolated={mx.max(mx.abs(pb-ref)).item():.2e} batched_naive_vs_isolated={mx.max(mx.abs(full-ref)).item():.2e} maxprob_mean={mx.mean(mx.max(ref,axis=-1)).item():.2f}")
def branched(tl):
    nc=[]
    for x in c:
        k,v=x.state; y=KVCache(); y.state=(mx.repeat(k,tl.shape[0],axis=0),mx.repeat(v,tl.shape[0],axis=0)); nc.append(y)
    return probs(model(tl, cache=nc)[:, -1, :])
alone = mx.concatenate([branched(tails[i:i+1]) for i in range(B)])
print(f"  isolation (branched alone vs branched in batch of {B}) = {mx.max(mx.abs(alone-pb)).item():.2e}")
