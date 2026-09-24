# Capacity model: nanohunch

Author: sd-capacity. Mode GREENFIELD, depth standard. Date 2026-09-23.

Every number is labelled `MEASURED` (run by us today on this M5, command
recorded in section 2, or read from a file/API today), `DERIVED` (arithmetic on
labelled inputs, shown), or `ASSUMPTION` (a guess with a range). Published
third-party numbers are cited to `_brief/research-notes.md` (`RN:<line>`) or
`_brief/requirements.md` (`REQ:<line>`) and count as "measured by them".
Rounded aggressively on purpose: we want the order of magnitude and the thing
that breaks first.

This is an ML project, so "load" means three things: (a) training compute,
(b) teacher-labelling throughput, (c) local inference on the M5. There is no
RPS, no cross-AZ traffic and no fleet.

---

## 0. Summary

**First bottlenecks, one per subsystem:**

| Subsystem | First bottleneck | Where it saturates | Label |
|---|---|---|---|
| Training | **Qwen3.5 Gated DeltaNet training path.** On the M5 it trains at 8 tok/s (4B) and 56 tok/s (0.8B), 12x to 22x slower than a dense model of the same size. On CUDA, without `flash-linear-attention` kernels, the target-size 4B run is 13 A100-h (10 to 32). That is over the 8 A100-h per-run cap. | Already over the cap at 1x plan if the kernels do not load. At about 1.5x plan if they do. | Mac MEASURED; CUDA DERIVED from ASSUMPTION |
| Labelling | **Cost and terms of service, not throughput.** Throughput is fine: a low API tier (60 RPM) labels the MVP in about 2.5 h, and an open 27B on one H100 labels it in about 11 min of prefill. For the open-weight teacher, startup (weights plus engine warmup, about 15 to 25 min) costs more time than the work itself. | n/a | DERIVED from ASSUMPTION |
| Inference (M5) | **Qwen3.5-4B long-state prefill.** W2 takes 11.5 s against a 10 s target, and peak memory is 14.6 GB, 77% of the 19.07 GB Metal working set. | Already over at 1x (W2) | MEASURED (synthetic weights) |

**Second and third** are ranked in section 9. **Headline for the orchestrator:**
the base-model choice is also a capacity choice. MiniCPM5-2B (plain Llama) meets
every Mac latency target with 2x margin and trains on the Mac overnight.
Qwen3.5-4B misses the W0 and W2 targets on the Mac, cannot train on the Mac,
and on CUDA needs kernels we have not yet verified.

---

## 1. Load model

Why this matters: every later number is one of these inputs multiplied by a
hardware rate, so a wrong input here scales every answer.

### 1.1 Training load

| Input | Value | Label | Source |
|---|---|---|---|
| Train decisions, MVP / target | 15k / 40k (30k to 50k) | ASSUMPTION | REQ:280-281 |
| Decisions per state `d` | 2 (1.5 to 4; Jev-style decomposition up to 16) | ASSUMPTION | REQ:283, pngwn 45,932 / 31,109 = 1.48 (RN:96-97) |
| Mean state length | 1,000 tokens (400 to 2,000) | ASSUMPTION | REQ:163, A18 |
| State length if p50 1k / p99 12k is lognormal | mean = 1,000 x exp(sigma^2 / 2), sigma = ln(12) / 2.326 = 1.07, so mean = 1,000 x exp(0.57) = **1,770** | DERIVED | a right-skewed distribution has mean well above median; truncation at 8k brings it to about 1,600 |
| Question tail (question text + options + answer slot) | 32 tokens | ASSUMPTION | REQ:161 |
| Epochs | 2 | ASSUMPTION | pngwn used 2 (RN:105-108) |
| Trained max length | 4,096 MVP, 8,192 target | requirement | REQ:270 |
| Share of decisions teacher-labelled | 60% | ASSUMPTION | REQ:282 |

### 1.2 Labelling load

```
MVP teacher decisions    = 15k x 0.6 = 9k decisions        DERIVED
MVP teacher states       = 9k / d(2) = 4.5k states          DERIVED
labels to buy            = 9k x 2 teachers = 18k decision-labels
API calls (1 per state)  = 4.5k x 2 teachers = 9k calls     DERIVED
Target                   = 40k x 0.6 = 24k decisions, 12k states, 24k calls
```

### 1.3 Inference load (M5, one caller)

| Workload | State tokens | Questions | Tail tokens each | Label |
|---|---|---|---|---|
| W0 short | 256 | 8 | 32 | ASSUMPTION (REQ:160) |
| W1 headline | 1,024 | 16 | 32 | ASSUMPTION (REQ:161) |
| W2 long | 8,192 | 16 | 32 | ASSUMPTION (REQ:162) |
| p99 questions per request | 64 | | | ASSUMPTION (REQ:164) |
| Concurrency | 1 (1 to 2) | | | ASSUMPTION (REQ:159) |

Little's law for the demo: one request every 10 s (ASSUMPTION) x 1.8 s latency
= 0.18 requests in flight (DERIVED). Nothing queues at one user. A second user
arriving mid-request waits the full request, so their latency doubles. That is
acceptable for a demo.

---

## 2. What was measured today (commands and outputs)

Why this matters: every figure in sections 3, 4 and 6 either comes from these
runs or is labelled as a guess. This section tells you which is which.

Machine: Apple M5 (`Mac17,2`), 24 GiB, mlx 0.31.2, mlx-lm 0.31.3, from the
`/Users/umangkaushik/fun/gemma-mlx/.venv` venv. Nothing was installed and no
model weights were downloaded. Scripts are copied to
`capacity-bench/` next to this file.

**M2.0 Metal working set.**
`python -c "import mlx.core as mx; print(mx.device_info())"` ->
`max_recommended_working_set_size: 19069665280` = **19.07 GB** MEASURED.
This settles REQ A15 (it guessed 16 to 18 GB). The 70% headroom line is
0.7 x 19.07 = **13.3 GB** DERIVED.

**M2.1 HF configs** (MEASURED, `curl https://huggingface.co/<id>/raw/main/config.json`):

| Model | layers | hidden | inter | attn heads / kv / head_dim | DeltaNet (k heads / v heads / dk / dv / conv) | vocab | tied |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-Base | 24 (18 DeltaNet + 6 attn) | 1024 | 3584 | 8 / 2 / 256, output gate | 16 / 16 / 128 / 128 / 4 | 248,320 | yes |
| Qwen3.5-4B-Base | 32 (24 DeltaNet + 8 attn) | 2560 | 9216 | 16 / 4 / 256, output gate | 16 / 32 / 128 / 128 / 4 | 248,320 | yes |
| MiniCPM5-2B-Base | 42 | 2048 | 6144 | 16 / 2 / 128 | none | **130,560** (not 73k) | no |
| Qwen3-1.7B-Base | 28 | 2048 | 6144 | 16 / 8 / 128 | none | 151,936 | yes |
| Qwen3-4B-Base | 36 | 2560 | 9728 | 32 / 8 / 128 | none | 151,936 | yes |

mlx-lm stores the DeltaNet recurrent state as fp32 `(B, Hv, Dv, Dk)`
(`mlx_lm/models/gated_delta.py:240,279`, MEASURED code read). The conv state has
the activation dtype (`qwen3_5.py:151-154`).

**M2.2 Synthetic-weight benchmarks.** `capacity-bench/synth_bench.py` builds each
text model from its real `config.json` with random bf16 weights. Speed does not
depend on weight values. Validation: synthetic Qwen3-1.7B prefill measured
2,661 tok/s at 1k, against 2,447 tok/s for real weights in `mlx_lm.benchmark`
(REQ:140), so synthetic runs are within about 10%. Run-to-run spread on this
laptop was up to about 15% (ASSUMPTION: thermal state). Command form:
`CHUNK=<n> python synth_bench.py <config.json> <qwen35|llama> <prefill|branch|train> <S> [B] [ckpt]`.

Prefill, single forward (MEASURED):

| Model | 256 tok | 1,024 tok | 4,096 tok | 8,192 tok (one shot) | 8,192 tok (chunks of 1,024) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 4,048 tok/s | 4,667 | 3,832 | 2,793 (2.93 s, 5.2 GB) | 3,672 (512-chunk, 2.23 s, 2.3 GB) |
| Qwen3.5-4B | 840 | 897 (9.4 GB) | | **468 (17.5 s, 14.9 GB)** | **785 (10.4 s, 9.8 GB)** |
| MiniCPM5-2B | 1,390 | 1,867 (5.5 GB) | | 1,227 (6.7 s) | 1,296 (2k chunks, 6.3 s, 6.0 GB) |
| Qwen3-4B | | 878 (8.6 GB) | | 697 (11.8 s, 10.3 GB) | 685 (2k chunks) |

Finding: one-shot 8k prefill on Qwen3.5-4B is 1.7x slower and uses 5 GB more
than chunked prefill. **The engine must prefill in chunks of about 1k tokens.**
Chunk sizes 256 / 512 / 1,024 / 2,048 gave 666 / 719 / 785 / 592 tok/s MEASURED.

Branch benchmark: prefill the state once, then score B tails of 32 tokens
batched against a copied cache (KV for attention layers, conv + recurrent state
for DeltaNet layers), then gather 4 letter rows and softmax (MEASURED, chunked
prefill 1,024):

| Model | Workload | prefill ms | branch + tails ms | total ms | per-branch prefix KV | per-branch DeltaNet state | peak GB |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | W0 256 x 8 | 58 | 49 | **107** | 3.1 MB | 19.5 MB | 2.1 |
| Qwen3.5-0.8B | W1 1k x 16 | 224 | 134 | **358** | 12.6 MB | 19.5 MB | 2.7 |
| Qwen3.5-0.8B | W2 8k x 16 | 2,163 | 273 | **2,436** | 100.7 MB | 19.5 MB | 4.1 |
| Qwen3.5-4B | W0 | 312 | 329 | **642** | 8.4 MB | 51.5 MB | 9.4 |
| Qwen3.5-4B | W1 | 1,165 | 641 | **1,806** | 33.6 MB | 51.5 MB | 10.6 |
| Qwen3.5-4B | W1, k=1 | 1,165 | 153 | 1,319 | | | 9.5 |
| Qwen3.5-4B | W2 | 10,437 | 1,043 | **11,479** | 268.4 MB | 51.5 MB | **14.6** |
| Qwen3.5-4B | 1k x **64** (p99 questions) | 1,165 | 2,618 | 3,784 | 33.6 MB | 51.5 MB | **15.9** |
| MiniCPM5-2B | W0 | 154 | 165 | **318** | 11.0 MB | 0 | 5.4 |
| MiniCPM5-2B | W1 | 552 | 341 | **893** | 44.0 MB | 0 | 6.2 |
| MiniCPM5-2B | W1, k=1 | 550 | 87 | 637 | | | 5.5 |
| MiniCPM5-2B | W2 | 5,817 | 765 | **6,581** | 352.3 MB | 0 | 11.3 |
| MiniCPM5-2B | 1k x 64 | 547 | 1,348 | 1,895 | 44.0 MB | 0 | 9.0 |
| Qwen3-4B | W0 | 282 | 307 | **589** | 37.7 MB | 0 | 8.9 |
| Qwen3-4B | W1 | 1,017 | 648 | **1,664** | 151.0 MB | 0 | 11.4 |

Real-weights branch run, `capacity-bench/branch_bench.py` (Qwen3-1.7B from HF
cache, MEASURED): W1 prefill 479 ms + tails 338 ms = **817 ms**. Naive
re-encode of 16 x 1,056 tokens took 8,353 ms, so sharing gives **10.2x**
(REQ:171 predicted an 11x ceiling). W0 = 290 ms (3.8x over naive). The same run
on real Qwen3-0.6B: 382 ms, 8.5x.

**M2.3 Mac LoRA training.**

(a) Stock `mlx_lm lora` (full-vocab loss, rank 8 on all linears, batch 1, 12 iters; MEASURED):

```
python -m mlx_lm lora --model <Qwen3-1.7B snapshot> --train --data data --num-layers -1 \
  --batch-size 1 --iters 12 --steps-per-report 4 --steps-per-eval 1000 --val-batches 1 \
  --max-seq-length <L> --adapter-path <dir> [--grad-checkpoint]
```

| Model | seq | grad ckpt | tok/s (steady) | peak GB |
|---|---|---|---|---|
| Qwen3-1.7B | 1,024 | yes | 473 | 5.45 |
| Qwen3-1.7B | 2,048 | yes | 381 to 403 | 7.14 |
| Qwen3-1.7B | 2,048 | no | 481 to 503 | **17.39** (91% of working set) |
| Qwen3-1.7B | 4,096 | yes | 316 | 10.54 |
| Qwen3-0.6B | 2,048 | yes | 685 | 4.37 |

(b) `synth_bench.py train`: LoRA r16 on every linear, positions-first loss
(hidden state at the answer position only, 4 candidate rows of the LM head),
Adam, grad checkpointing, batch 1 (MEASURED):

| Model | seq | LoRA params | s/step | tok/s | peak GB |
|---|---|---|---|---|---|
| Qwen3-1.7B | 2,048 | 17.4M | 4.75 | 431 | 5.09 |
| MiniCPM5-2B | 1,024 | 25.1M | 2.95 | 347 | 6.27 |
| MiniCPM5-2B | 2,048 | 25.1M | 6.90 | 297 | 6.86 |
| MiniCPM5-2B | 4,096 | 25.1M | 17.66 | 232 | 9.17 |
| Qwen3-4B | 2,048 | 33.0M | 11.52 | 178 | 10.68 |
| **Qwen3.5-0.8B** | 512 | 10.8M | 7.00 | **73** | 5.14 |
| **Qwen3.5-0.8B** | 1,024 | 10.8M | 18.23 | **56** | 9.39 |
| **Qwen3.5-4B** | 512 | 32.5M | 66.27 | **8** | **15.84** |

Positions-first loss vs the stock full-vocab loss on Qwen3-1.7B at 2k with
checkpointing: 431 vs 390 tok/s and **5.09 vs 7.14 GB** (MEASURED; the rank
differs, 16 vs 8, which only makes the comparison conservative). Between 1k and
4k, stock memory grows 1.7 GB per 1k tokens: (10.54 - 5.45) / 3 (DERIVED).
Most of that growth is full-vocab logits (section 4).

**M2.4 Branch equivalence and isolation** (`capacity-bench/equiv.py`, real
Qwen3-0.6B, S=1,024, B=16, random-token state, MEASURED):

| Comparison | bf16 | fp32 |
|---|---|---|
| Isolation: question scored alone (branched, B=1) vs inside a 16-branch batch | **0.0** | **0.0** |
| Branched (prefill 1,024 then tail 32) vs isolated full re-encode of 1,056 tokens | **1.3e-2** | 4.0e-4 |
| Batched naive re-encode (16 x 1,056 in one call) vs isolated re-encode | 1.2e-1 | 1.2e-1 |

Reading: isolation (REQ S10) holds exactly. Splitting prefill from the tail
changes bf16 probabilities by about 1e-2 on low-confidence distributions (mean
max prob 0.4). REQ:260-261 targets (1e-3 bf16) are therefore too tight for
"branch vs re-encode". **Train and serve with the same prefill/tail split.**
The 0.12 gap on the batched naive path also appears in fp32, which is too large
to be rounding. It looks like an MLX batched-attention issue at B=16 x 1,056
that we did not investigate: `> TODO: unverified`, see open questions.

**M2.5 Post-processing** (real Qwen3 tokenizer, MEASURED): tokenizing a 1,024-token
state plus 16 questions of 32 tokens takes **2.2 ms**. Temperature scaling, softmax
and `json.dumps` for 16 x 26 probabilities take **0.14 ms** and produce 5.2 KB.

---

## 3. Training compute

Why this matters: GPU hours are the largest line of a 200 USD budget, and they
equal tokens x FLOPs per token / achieved FLOP/s. Each of those three factors
can be cut on purpose.

### 3.1 Parameters from config (DERIVED; every total cross-checked against the synthetic build count, MEASURED)

```
Qwen3.5-4B
  MLP/layer      3 x 2560 x 9216                                   = 70.8M  x 32 = 2.27B
  DeltaNet/layer qkv 2560 x 8192 + z 2560 x 4096 + out 4096 x 2560
                 + a,b 2 x 2560 x 32                               = 42.1M  x 24 = 1.01B
  Attn/layer     q+gate 2560 x 8192 + k,v 2 x 2560 x 1024 + o 4096 x 2560 = 36.7M x 8 = 0.29B
  N (non-embedding) = 3.57B ; embedding 248,320 x 2560 = 0.64B (tied) ; total 4.21B (MEASURED 4.21B)
Qwen3.5-0.8B
  MLP 3 x 1024 x 3584 = 11.0M x 24 = 0.26B ; DeltaNet 10.5M x 18 = 0.19B ; attn 7.3M x 6 = 0.04B
  N = 0.50B ; embedding 0.25B ; total 0.75B (MEASURED 0.75B)
MiniCPM5-2B   (attn 9.4M + MLP 37.7M) x 42 = 1.98B = N ; embed + head 2 x 130,560 x 2048 = 0.53B ; total 2.52B
Qwen3-1.7B    (12.6M + 37.7M) x 28 = 1.41B = N ; embedding 0.31B tied ; total 1.72B
Qwen3-4B      (26.2M + 74.7M) x 36 = 3.63B = N ; embedding 0.39B tied ; total 4.02B
```

### 3.2 Tokens per epoch: naive vs shared state

Naive training makes one sequence per decision and re-encodes the state every
time. Shared-state training encodes each state once and attaches every
question tail to it. On dense models (MiniCPM5, Qwen3) that is **packing with a
tree mask**: one sequence `[state | q1 | q2 | ...]` in which each tail attends
to the state and to itself only (FlexAttention, or an SDPA block mask). On
Qwen3.5 the mask trick **does not work**, because the DeltaNet recurrence is
sequential: q2's tokens would read q1's tokens through the recurrent state.
Qwen3.5 must use **cache branching inside the autograd graph**. Run the state
once, expand its KV and DeltaNet state to B branches without detaching, run the
B tails as a batch, and let gradients from all branches add up in the shared
prefix. This needs the chunked DeltaNet kernel to accept an `initial_state` and
return its gradient (ASSUMPTION for fla, Phase 0 item P0-4).

```
naive   tokens = decisions x (state + tail)
shared  tokens = states x state + decisions x tail

MVP, d = 2, state 1,000, tail 32:
  naive  = 15k x 1,032                  = 15.5M tokens/epoch   DERIVED
  shared = 7.5k x 1,000 + 15k x 32      =  8.0M tokens/epoch   DERIVED   saving 1.9x
  d = 4:  3.75k x 1,000 + 0.48M         =  4.2M                          saving 3.7x
  d = 16: 0.94k x 1,000 + 0.48M         =  1.4M                          saving 11x
Target, d = 2: naive 41M, shared 21M tokens/epoch                       DERIVED
2 epochs: MVP 16M, target 43M tokens (shared)                           DERIVED
If mean state is 1,770 (lognormal case): multiply everything by ~1.7
```

The saving grows with questions per state, which is the same lever that makes
inference cheap. Designing datasets with more questions per state (Jev-style
decomposition) is **free training compute**.

### 3.3 FLOPs per token

The convention used here: **6N hardware FLOPs per token = 4N for LoRA
(forward 2N + backward-to-activations 2N; the frozen-weight gradients are
skipped) + 2N for the recomputed forward under gradient checkpointing.** It
happens to equal the full fine-tune figure. Without checkpointing (possible at
seq up to 2k on 80 GB) it drops to 4N and the run gets 1.5x faster. N is
non-embedding, because the positions-first head computes the LM head only at
answer positions. If you compute it at every position you add 3 x 2 x h x V
per token: 3.8e9 on Qwen3.5-4B (+18%) and 1.5e9 on Qwen3.5-0.8B (+51%).

| Model | N | 6N FLOP/token | attention add at 1k (8 x L_attn x s x d_attn) | at 8k | Label |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.50B | 3.0e9 | 8 x 6 x 1k x 2048 = 0.1e9 (+3%) | +26% | DERIVED |
| Qwen3.5-4B | 3.57B | 21.4e9 | 8 x 8 x 1k x 4096 = 0.26e9 (+1%) | +10% | DERIVED |
| MiniCPM5-2B | 1.98B | 11.9e9 | 8 x 42 x 1k x 2048 = 0.69e9 (+6%) | **+47%** | DERIVED |
| Qwen3-1.7B | 1.41B | 8.5e9 | 8 x 28 x 1k x 2048 = 0.46e9 (+5%) | +43% | DERIVED |
| Qwen3-4B | 3.63B | 21.8e9 | 8 x 36 x 1k x 4096 = 1.2e9 (+5%) | +43% | DERIVED |

At 8k tokens and beyond, the hybrid's 1-in-4 attention layers are a real
advantage. DeltaNet chunk FLOPs are linear in s and small. At a 1k mean
length, the attention term can be ignored.

### 3.4 GPU hours

Achieved rate = peak bf16 dense x MFU. Peaks are spec-sheet figures recalled
from memory (ASSUMPTION): A100 40/80 GB 312 TFLOP/s, H100 SXM 989, L40S 362,
L4 121, A10G about 70 (A10 is 125). MFU for LoRA on a 1B to 4B model with FA2
and 4k to 8k-token packed micro-batches: **25%** (15 to 40%, ASSUMPTION). On
H100 we use 20%, because small models underfill it. Achieved rates: A100
**78 TFLOP/s**, H100 **200**, L40S 90, L4/A10G **~25**.

Cross-check against pngwn arm B: 0.6B, 46k decisions, about 350 tokens,
2 epochs, 68 min on one A100 (RN:106-108). That is 2 x 46k x 350 / 4,080 s =
7.9k tok/s (DERIVED, REQ:326-329). At 6 x 0.44e9 = 2.6e9 FLOP/token it is
21 TFLOP/s, **7% MFU**. Tiny models at short lengths are overhead-bound. For
sub-1B models, the estimate below takes the larger of the FLOP-based time and
tokens / 8k tok/s.

Qwen3.5 slow-kernel factor P (multiplies time, applies to Qwen3.5 only):
- With `flash-linear-attention` chunked kernels and `causal-conv1d` loaded:
  P = 1.4 (1.2 to 2). ASSUMPTION: chunked DeltaNet kernels run at lower MFU
  than GEMMs, but most FLOPs are still projections and MLP.
- Torch reference fallback (`modeling_qwen3_5.py:249-301,437` falls back when
  hub kernels are missing, system-map 0.1): P = **4 (3 to 10)**. ASSUMPTION. The
  torch fallback is chunked (it loops over s/64 chunks), not per-token, so it
  should hurt less than MLX's per-token loop. The MEASURED MLX penalty is
  **12x to 22x**. Derivation: dense Qwen3-4B trains at 178 tok/s against 8 for
  Qwen3.5-4B, so 22x. Qwen3.5-0.8B trains at 56 tok/s against an estimated
  ~700 for a dense 0.8B (Qwen3-0.6B stock measured 685), so 12x.

```
A100 hours = tokens x 6N / 78e12 / 3600 x P

MVP (16M tokens, shared):
  Qwen3.5-4B  16e6 x 21.4e9 = 3.4e17 / 7.8e13 = 4,400 s = 1.2 h x 1.4 = 1.7 h  (fla)
                                                        x 4 = 4.9 h (3.7 to 12)  (fallback)
  MiniCPM5-2B 16e6 x 11.9e9 = 1.9e17 / 7.8e13 = 2,450 s = 0.7 h
  Qwen3-4B    16e6 x 21.8e9 = 3.5e17 / 7.8e13           = 1.25 h
  Qwen3-1.7B  16e6 x 8.5e9  = 1.4e17 / 7.8e13           = 0.5 h
  Qwen3.5-0.8B 16e6 x 3.0e9 = 4.8e16 / 7.8e13 = 0.17 h x 1.4 = 0.25 h,
               but overhead floor 16e6 / 8k tok/s = 0.56 h -> ~0.5 h (fla), ~1 h (fallback)
Target (43M tokens) = MVP x 2.7
```

| Model | MVP A100-h | Target A100-h | Target H100-h (/2.6) | Target L4/A10G-h (x3.1) | Target USD at A100 2.50/h (Modal, ASSUMPTION) | Label |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (fla / fallback) | 0.5 / 1 | 1.4 / 2.7 | 0.5 / 1.0 | 4 / 8 | 3.5 / 7 | DERIVED |
| **Qwen3.5-4B, fla** | **1.7** | **4.6** | 1.8 | 14 (and 8k does not fit 24 GB) | 11.5 | DERIVED |
| **Qwen3.5-4B, fallback** | **4.9 (3.7 to 12)** | **13 (10 to 32)** | 5 (4 to 12) | infeasible | **33 (25 to 80)** | DERIVED |
| MiniCPM5-2B | 0.7 | 1.9 | 0.7 | 6 | 4.8 | DERIVED |
| Qwen3-1.7B | 0.5 | 1.35 | 0.5 | 4 | 3.4 | DERIVED |
| Qwen3-4B | 1.25 | 3.4 | 1.3 | 10.5 (8k does not fit) | 8.5 | DERIVED |

Without state sharing, multiply by 1.9 (d = 2). With the lognormal mean
(1,770 tokens), multiply by 1.7. **Both multipliers together (3.2x) push the
Qwen3.5-4B fallback case to 42 A100-h for one target run, more than the entire
90 USD GPU line.** The requirement "every full run <= 8 A100-h" (REQ:340) holds
for every candidate except Qwen3.5-4B on the fallback path.

H100 at 3.95 USD/h (ASSUMPTION, `rl-wordle/PLAN.md:91-101` via system-map)
beats A100 at 2.50 per run: target Qwen3.5-4B fla costs 1.8 x 3.95 = 7 USD on
H100 against 11.5 USD on A100 (DERIVED). Prices belong to sd-cost; see
`needs_from`.

### 3.5 Can the M5 train it? (MEASURED rates from 2.3b, 1k to 2k tokens)

```
hours = tokens / tok_per_s / 3600
MVP 16M tokens:     MiniCPM5-2B 16e6 / 320 = 50,000 s = 14 h
                    Qwen3-1.7B  16e6 / 431            = 10 h
                    Qwen3-4B    16e6 / 178            = 25 h
                    Qwen3.5-0.8B 16e6 / 56            = 79 h (3.3 days)
                    Qwen3.5-4B  16e6 / 8              = 556 h (23 days)
Target 43M tokens:  MiniCPM5-2B 37 h, Qwen3-1.7B 28 h, Qwen3-4B 67 h,
                    Qwen3.5-0.8B 9 days, Qwen3.5-4B 62 days
```

Verdict: **dense 1.7B to 2B LoRA on the Mac is feasible.** An MVP run is one
night plus a morning. That matches REQ:343-345 in shape but is 1.7x slower than
its 550 tok/s guess (MEASURED 297 to 347 for MiniCPM5). Qwen3.5 of any size is
**not feasible on the Mac** until someone writes a VJP for the Metal DeltaNet
kernel or a chunked MLX DeltaNet. That would be a stretch learning project, not
a requirement. Sustained multi-hour throughput is unmeasured and could be 10 to
30% lower from thermals (ASSUMPTION, P0-8).

---

## 4. Training memory

Why this matters: memory decides the cheapest GPU class you can rent, and the
cheapest class is set by the largest sequence you train on, not the average.

Components, batch 1, bf16 frozen weights, LoRA r16 with fp32 master + grad +
Adam (16 B/param), gradient checkpointing, FlashAttention/SDPA (no s^2 matrix):

```
weights      = total params x 2 B
LoRA state   = LoRA params x 16 B
checkpoints  = s x hidden x 2 B x layers                          (one tensor per layer boundary)
recompute    = s x (12 x hidden + 4 x inter) x 2 B x 2            (one layer's fwd+bwd live at once)
DeltaNet     = (s / 64) x Hv x dk x dv x 4 B per layer            (chunk states, Qwen3.5 only)
logits, all positions, full vocab = s x V x 8 B                   (fp32 logits + fp32 grad)
logits, positions-first           = k x V x 8 B, k ~ 16  -> 32 MB (Qwen3.5); candidate rows -> < 1 MB
runtime      = 2 GB CUDA context + allocator slack               ASSUMPTION (1.5 to 3)
```

The full-vocab logits term, per token of sequence: Qwen3.5 248,320 x 8 =
2.0 MB, MiniCPM5 1.04 MB, Qwen3 1.2 MB (DERIVED). At 8k on Qwen3.5 that is
**16.3 GB for the logits alone**. That is consistent with pngwn's 20.7 GiB OOM
(RN:109-110). **Most of the saving comes from gathering positions first**
(logits only at answer positions). Restricting to candidate rows then also
removes the head's FLOPs (section 3.3).

Totals at 1k / 4k / 8k tokens per micro-batch. The range runs from the
analytic value to analytic with activations x2. The x2 comes from MLX
MEASURED: MiniCPM5 activations grew 0.97 GB per 1k tokens, against 0.37
analytic.

| Model | weights | LoRA | act 1k / 4k / 8k | **total with trick** 1k / 4k / 8k | total without trick (all-position full-vocab logits) 8k | min GPU (<=70%) | Label |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 1.5 | 0.2 | 0.16 / 0.64 / 1.3 | 3.9 / 4.4 / **5.0 to 6.3** | 21 | 24 GB (L4/A10G) up to 16k | DERIVED |
| Qwen3.5-4B | 8.4 | 0.5 | 0.45 / 1.8 / 3.8 | 11.4 / 12.8 / **14.8 to 18.6** | 31 | 24 GB up to 4k; **40 GB (A100 40 / L40S) for 8k** | DERIVED |
| MiniCPM5-2B | 5.0 | 0.4 | 0.38 / 1.5 / 3.0 | 7.8 / 8.9 / **10.4 to 13.4** | 19 | 24 GB up to 8k | DERIVED; Mac MEASURED 6.27 / 9.17 GB at 1k / 4k |
| Qwen3-1.7B | 3.4 | 0.3 | 0.32 / 1.3 / 2.5 | 6.0 / 7.0 / **8.2 to 10.7** | 18 | 24 GB up to 8k | DERIVED; Mac MEASURED 5.09 at 2k |
| Qwen3-4B | 8.0 | 0.5 | 0.47 / 1.9 / 3.8 | 11.0 / 12.4 / **14.4 to 18.2** | 24 | 24 GB up to 4k; 40 GB for 8k | DERIVED; Mac MEASURED 10.68 at 2k |

Qwen3.5 branch training adds k x 51.5 MB DeltaNet branch state, roughly x2
with gradients: +0.4 GB at k = 4 and +1.6 GB at k = 16 (DERIVED from 2.2).
The torch fallback materializes fp32 intermediates and could double the
DeltaNet activations (ASSUMPTION).

**Mac training memory (MEASURED):** MiniCPM5 needs 9.17 GB at 4k (48% of
19.07). Extrapolating at +0.97 GB per 1k gives about 13 GB at 8k, which is 68%
and borderline (DERIVED). Qwen3.5-4B already needs 15.84 GB at 512 tokens
(83%), so it is over the line before it even gets slow. Turning off
checkpointing on Qwen3-1.7B at 2k took 17.39 GB (91%): **always checkpoint on
the Mac.**

---

## 5. Teacher labelling throughput

Why this matters: labels are the only input that costs money per item. You want
to know whether time, rate limits or dollars run out first, and it is dollars.

### 5.1 API teacher (all ASSUMPTION: per-model limits and prices not fetched, RN:159)

```
per call: 1 state + d questions, ~1,300 input + ~500 output tokens = 1,800 tokens   (REQ:309)
latency per call: 6 s (3 to 15; much longer with "thinking")                         ASSUMPTION
tier limits: low 60 RPM / 100k TPM ; mid 500 RPM / 1M TPM                            ASSUMPTION

low tier:  RPM caps at 1 call/s ; TPM 100k / 1,800 = 55 calls/min, so TPM binds at 0.9 calls/s
           decisions/h = 0.9 x 3600 x 2 = 6.5k/h
           Little: concurrency needed = 0.9/s x 6 s = 6 in flight
mid tier:  RPM 500 -> 8.3 calls/s ; TPM 1M / 1,800 = 555/min = 9.3/s ; RPM binds
           decisions/h = 8.3 x 3600 x 2 = 60k/h ; concurrency = 8.3 x 6 = 50 in flight

MVP 9k calls:     low tier 9k / 0.9 = 10,000 s = 2.8 h ; mid tier 18 min
Target 24k calls: low tier 7.4 h ; mid tier 48 min
Strong-tier eval reference, 2k decisions = 1k calls x 45 s (thinking) / 8 concurrent = 94 min
```

Throughput is never the bottleneck. Cost (REQ:305-321) and terms of service
(A8) are. Provider batch APIs (about 50% off, 24 h turnaround, ASSUMPTION) are
a free win, because labelling is offline.

### 5.2 Open-weight teacher on a rented GPU (prefill-only, read option logits)

The workload: each question is a request of `[instructions | state | question]`
followed by 1 decode step with `allowed_token_ids = option letters` and
`logprobs = n_options`. Prefix caching makes the state (and the global
instructions) cost prefill once per state.

```
tokens per decision (prefix cache works) = (1,000 + d x 32) / d = (1,000 + 64) / 2 = 532  DERIVED
tokens per decision (no prefix cache)    = 1,032                                           DERIVED
MVP teacher tokens = 4.5k x 1,000 + 9k x 32 = 4.8M ; target 12k x 1,000 + 24k x 32 = 12.8M

Qwen3.8-27B dense (bf16 54 GB, fits one H100 80 GB with ~20 GB KV):
  2 x 27e9 = 54e9 FLOP/token ; H100 prefill MFU 40% (25 to 50%) = 396 TFLOP/s     ASSUMPTION
  -> 7.3k tok/s -> 7.3k x 3600 / 532 = 49k decisions/h (25k/h without prefix cache)
  MVP 4.8M / 7.3k = 660 s = 11 min ; target 12.8M / 7.3k = 29 min                  DERIVED
Qwen3.8-35B-A3B MoE (~3B active; bf16 ~70 GB does NOT leave KV room on 80 GB -> FP8 on H100 or 4-bit):
  2 x 3e9 = 6e9 FLOP/token ; MoE prefill MFU 15% (10 to 30%) = 150 TFLOP/s        ASSUMPTION
  -> 25k tok/s -> 170k decisions/h ; MVP 3 min ; target 9 min                      DERIVED
Fixed cost per session: pull 54 to 70 GB weights (1 to 5 min at 0.2 to 1 GB/s)
  + vLLM load + CUDA graph capture (3 to 10 min)                                   ASSUMPTION
  -> 15 to 25 min, which is larger than the MVP compute itself
```

Qwen3.5-family teachers are probably hybrid DeltaNet too (ASSUMPTION). vLLM
prefix caching for hybrid models needs state snapshots at block boundaries and
may be off or restricted. If so, tokens per decision are 1,032, not 532
(x1.9), which is still under an hour for the target.

**Mac as a free teacher (DERIVED, not recommended):** 27B at 4-bit is
27e9 x 0.56 B = 15 GB = 79% of 19.07. That is over the 70% line. At about
6.5e12 / 54e9 = 120 tok/s, the MVP takes 4.8M / 120 = 11 h. It works overnight
but is fragile. 35B-A3B at 4-bit (about 19.5 GB) does not fit.

---

## 6. Local inference on the M5

Why this matters: latency is the headline claim of a Jev-style system, and
here it is set almost entirely by one step, state prefill.

### 6.1 Latency per workload (MEASURED, synthetic weights, chunked prefill 1,024, section 2.2)

| Model | W0 256 x 8 | W1 1k x 16 | W2 8k x 16 | vs targets W0 0.5 s / W1 2.0 s / W2 10 s (REQ:200-203) |
|---|---|---|---|---|
| Qwen3.5-0.8B | **107 ms** | **358 ms** | **2.44 s** | all pass; meets the stretch targets (0.15 / 0.5 / 5 s) |
| MiniCPM5-2B | **318 ms** | **893 ms** | **6.58 s** | all pass with 1.5x to 2.2x margin |
| Qwen3-1.7B (real weights) | 290 ms | 817 ms | ~4 s prefill (DERIVED 8,192 / 2,199), **memory fails** (6.2) | W0, W1 pass; W2 fails on memory with copy branching |
| Qwen3-4B | 589 ms | 1,664 ms | ~13 s (DERIVED 11.8 s prefill MEASURED + ~1.5 s tails), **memory fails** | W0 fails target (meets MVP 0.8 s), W1 passes at 83% of budget, W2 fails |
| **Qwen3.5-4B** | **642 ms** | **1,806 ms** | **11.48 s** | **W0 misses target (meets MVP 0.8 s), W1 passes at 90% of budget, W2 misses target (meets MVP 15 s)** |

REQ:185-203 derived 1.6 s for W1, 0.54 s for W0 and 9.2 s for W2 on
Qwen3.5-4B. The measured values are **13 to 25% worse** on every workload.
Unchunked 8k prefill would make W2 18.5 s, which also misses the MVP target.

Marginal cost of questions (REQ S8), latency(k=16) / latency(k=1) at W1:
Qwen3.5-4B 1,806 / 1,319 = 1.37, MiniCPM5 893 / 637 = 1.40. Both meet the
1.6 target (DERIVED from MEASURED).

### 6.2 Branch memory

Per branch, from config (DERIVED, matches MEASURED 2.2 to 3 significant figures):

```
Qwen3.5-4B recurrent  = Hv x Dv x Dk x 4 B = 32 x 128 x 128 x 4 = 2.10 MB/layer x 24 = 50.3 MB
           conv       = (k-1) x conv_dim x 2 B = 3 x (2 x 2048 + 4096) x 2 = 49 KB/layer x 24 = 1.2 MB
           -> 51.5 MB per branch, independent of state length          (MEASURED 51.5)
           attn KV    = 2 x kv 4 x 256 x 2 B x 8 layers = 32 KB/token   (MEASURED 33.6 MB at 1,024)
Qwen3.5-0.8B recurrent 16 x 128 x 128 x 4 x 18 = 18.9 MB + conv 0.7 MB = 19.5 MB ; KV 12 KB/token
MiniCPM5-2B   KV = 2 x 2 x 128 x 2 B x 42 = 42 KB/token ; no recurrent state
Qwen3-1.7B    KV = 2 x 8 x 128 x 2 B x 28 = 112 KB/token
Qwen3-4B      KV = 2 x 8 x 128 x 2 B x 36 = 144 KB/token
```

Copy-branching memory = B x (prefix KV + recurrent) on top of weights:

| Model | W1 (B=16, 1k) | W2 (B=16, 8k) | 1k x B=64 | W2 peak / 19.07 GB | Label |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 16 x 32 MB = 0.5 GB | 16 x 120 MB = 1.9 GB | 2.1 GB | 4.1 GB = 21% | MEASURED peak |
| Qwen3.5-4B | 16 x 85 MB = 1.4 GB | 16 x 320 MB = 5.1 GB | 5.4 GB | **14.6 GB = 77%** (B=64 at 1k: **15.9 GB = 84%**) | MEASURED peak |
| MiniCPM5-2B | 0.7 GB | 16 x 352 MB = 5.6 GB | 2.8 GB | 11.3 GB = 59% | MEASURED peak |
| Qwen3-1.7B | 1.8 GB | 16 x 0.94 GB = 15 GB | | 3.4 + 15 = **18.4 GB = 97%** | DERIVED |
| Qwen3-4B | 2.4 GB | 16 x 1.21 GB = **19.3 GB** | | 8 + 19.3 = **27 GB, OOM** | DERIVED |

In the current implementation, `mx.repeat` of the prefix is lazy (branch copy
measured 0.3 ms), and the real copy happens when the attention cache
concatenates the tail keys. **Two engine consequences:**
1. **Shared-prefix attention** (queries of B branches read one prefix KV plus
   their own tail KV, never copying) removes the B x prefix-KV term. That is
   mandatory for dense 4B at W2 and for every candidate at 16k.
2. The DeltaNet fp32 state cannot be shared, because each branch evolves its
   own. Qwen3.5-4B pays 51.5 MB per question no matter what, so at p99 (64
   questions) the engine must **micro-batch branches** in groups of 16 to 32.
   That turns a memory ceiling into roughly linear latency.

### 6.3 Latency budget, W1, per lead candidate

| Hop | Budget | Qwen3.5-4B | MiniCPM5-2B | Qwen3.5-0.8B | Label |
|---|---|---|---|---|---|
| HTTP parse + validate | 5 ms | <= 2 ms | <= 2 ms | <= 2 ms | ASSUMPTION |
| Tokenize state + 16 questions | 10 ms | 2.2 ms | 2.2 ms | 2.2 ms | MEASURED (Qwen3 tokenizer) |
| State prefill 1,024 (chunked) | 1,300 ms | 1,165 ms | 552 ms | 224 ms | MEASURED |
| Branch (lazy copy) + 16 x 32 tails + 4-row gather + softmax | 650 ms | 641 ms | 341 ms | 134 ms | MEASURED |
| Per-type temperature + JSON (16 x 26) | 10 ms | 0.14 ms | 0.14 ms | 0.14 ms | MEASURED |
| Retries / TLS | 0 | 0 (localhost, no TLS) | 0 | 0 | n/a |
| **Total vs 2,000 ms target** | | **1,810 ms = 90%** | **897 ms = 45%** | **361 ms = 18%** | DERIVED |

**Qwen3.5-4B fits W1 at 90% of budget with a ~15% run-to-run spread, so its
p95 (target 2.5 s) is tight but plausible.** It does not fit W0 (642 ms vs
500) or W2 (11.5 s vs 10 s). By the brief's rule that is a design defect, not a
tuning problem: shipping Qwen3.5-4B means either relaxing W0/W2 to the MVP
thresholds or adding the 0.8B variant for the latency claim. 4-bit weights save
memory but should not speed up compute-bound prefill (ASSUMPTION: dequant cost
roughly offsets bandwidth savings when M >= 256).

Cold start (not in p50): 8.4 GB of weights from NVMe at 3 to 5 GB/s is 2 to
3 s, plus first-call Metal kernel JIT of 1 to 5 s (ASSUMPTION).

### 6.4 Eval throughput on the Mac (REQ:213 target <= 90 min)

```
per state = prefill(1k) + 8 tails : Qwen3.5-4B 1.17 + ~0.33 = 1.5 s ; MiniCPM5 0.55 + 0.17 = 0.72 s
test set 2,500 states: Qwen3.5-4B 3,750 s = 63 min (70% of budget) ; MiniCPM5 30 min      DERIVED
lognormal mean 1,770 tokens: Qwen3.5-4B prefill ~2.1 s -> 2,500 x 2.4 s = 100 min, misses   DERIVED
permutation tests reuse the state: +3 permutations x 8 tails = +1 s/state on Qwen3.5-4B
```

---

## 7. Storage and bandwidth

Why this matters: the numbers are small, and checking them rules out a class
of surprise.

| Item | Size | Label |
|---|---|---|
| Resumable checkpoint (LoRA 32.5M x (2 B weights + 12 B fp32 master + Adam)) | ~0.45 GB; every 20 min, keep 3 = 1.4 GB | DERIVED |
| Adapter only | 65 MB (4B), 50 MB (MiniCPM5) | DERIVED |
| Merged model bf16 / 4-bit | 8.4 GB / ~2.4 GB (Qwen3.5-4B) | DERIVED |
| Raw teacher responses | ~250 MB (REQ:288) | DERIVED |
| Tokenized dataset (43M tokens x 4 B int32) | 0.17 GB | DERIVED |
| Local disk free | 628 GiB (system-map) | MEASURED |

Bandwidth: a teacher weight pull of 54 to 70 GB per GPU session, repeated
unless cached on a provider volume (1 to 5 min each, ASSUMPTION). Base model to
the Mac: 5 to 9 GB once. Checkpoint upload: 0.45 GB per 20 min = 0.4 MB/s
average. Egress: under 20 GB per month total, about 2 USD at 90 USD/TB even if
billed (DERIVED). No cross-AZ traffic exists. The tokenizer runs at about
1,500 tokens / 2.2 ms = 0.7M tok/s on one core (MEASURED), against 25k tok/s
of training consumption on an A100 (DERIVED from 3.4), so data loading sits at
about 4% of one core.

---

## 8. Headroom check (anything above 70% is already a problem)

Why this matters: at one caller nothing queues, so the 70% rule applies to
memory and budget, where the failure mode is OOM or running out of money,
not added latency.

| Resource | Projected peak | Utilization | Verdict | Label |
|---|---|---|---|---|
| Mac working set, Qwen3.5-4B W2 | 14.6 / 19.07 GB | **77%** | over | MEASURED |
| Mac working set, Qwen3.5-4B 1k x 64 questions | 15.9 / 19.07 GB | **84%** | over; micro-batch branches | MEASURED |
| Mac working set, Qwen3.5-4B training at 512 | 15.8 / 19.07 GB | **83%** | over (and 8 tok/s) | MEASURED |
| Mac working set, MiniCPM5 W2 / train 4k | 11.3 / 9.2 GB | 59% / 48% | ok | MEASURED |
| W1 latency budget, Qwen3.5-4B | 1.81 / 2.0 s | **90%** | over | MEASURED |
| W1 latency budget, MiniCPM5 | 0.90 / 2.0 s | 45% | ok | MEASURED |
| GPU budget line (90 USD) if Qwen3.5-4B fallback: 3 target runs x 13 h + pilots 3 h + teacher/eval 2 h = 44 A100-h x 2.50 | 110 / 90 USD | **122%** | over | DERIVED |
| GPU budget line, Qwen3.5-4B with fla: 3 x 4.6 + 5 = 19 h x 2.50 | 48 / 90 USD | 53% | ok | DERIVED |
| GPU budget line, MiniCPM5-2B: 3 x 1.9 + 5 = 11 h x 2.50 | 27 / 90 USD | 30% | ok | DERIVED |
| 24 GB card, Qwen3.5-4B / Qwen3-4B training at 8k | 18.6 / 24 GB | **78%** | over; use 40 GB+ | DERIVED |
| Modal 24 h function limit, worst Qwen3.5-4B fallback run (32 h) | | 133% | over; resume required (REQ:354) | DERIVED |

---

## 9. Bottlenecks, in order

Why this matters: fixing the second bottleneck before the first buys nothing.

### Training

| Rank | Component | Saturates at | Multiple of plan | Then what |
|---|---|---|---|---|
| 1 | Qwen3.5 DeltaNet backward path (MLX per-token loop; CUDA torch fallback) | Mac: 8 tok/s at 4B (MEASURED). CUDA: 13 A100-h per target run (DERIVED) vs 8 h cap | **1x (already)** without fla; ~1.7x with fla (8 h cap / 4.6 h) | Verify fla loads (P0-2), or switch to MiniCPM5-2B / Qwen3-4B dense |
| 2 | GPU budget (90 USD line) | 36 A100-h at 2.50 USD/h | 1.9x plan with fla Qwen3.5-4B (36 / 19 h); 3.3x with MiniCPM5 | State sharing (1.9x to 3.7x), H100 instead of A100, fewer full runs |
| 3 | Per-GPU memory at 8k tokens (4B-class) | 70% of 24 GB at ~6k tokens | 1.3x of the 4k MVP length | 40 GB class for 8k; never compute all-position logits |

### Labelling

| Rank | Component | Saturates at | Multiple of plan | Then what |
|---|---|---|---|---|
| 1 | API dollars and terms of service (sd-cost owns the USD) | 60 USD label line = 240k cheap-tier decisions (0.00025 each) but only 6k to 15k strong-tier (REQ:319-321) | Cheap tier ~13x plan; strong tier below 1x for bulk | Open-weight teacher for bulk (and for anything released) |
| 2 | Open-weight session fixed cost (weight pull + engine warmup) | 15 to 25 min per session vs 11 min of 27B prefill for the MVP | 1x: overhead is already larger than the work | Cache weights on a volume; label everything in one session |
| 3 | Prefix caching disabled for a hybrid teacher | tokens per decision 532 -> 1,032 | 1.9x the work, still < 1 h at target | Group questions by state; or pick a dense teacher |

### Inference (M5)

| Rank | Component | Saturates at | Multiple of plan | Then what |
|---|---|---|---|---|
| 1 | Qwen3.5-4B state prefill at long states (DeltaNet Metal kernel is sequential over tokens) | W2 11.5 s vs 10 s; one-shot 8k 17.5 s | **1x (already)** | Chunked prefill (done in the measurement), relax W2 to MVP, or ship MiniCPM5 / 0.8B |
| 2 | Branch memory: fp32 DeltaNet state 51.5 MB/question plus copied KV | 70% of 19.07 GB (13.3 GB) at (13.3 - 8.4 - 0.8) / 0.085 = ~47 questions on 1k states, (13.3 - 8.4 - 0.3 - 0.8) / 0.32 = ~12 questions on 8k states (Qwen3.5-4B, DERIVED) | already over at W2 (16 > 12); 2.9x at W1 (47 / 16) | Micro-batch branches; shared-prefix attention for KV |
| 3 | Fixed per-forward cost on short states (W0) | Qwen3.5-4B W0 642 ms vs 500 | 1x | Only a smaller model fixes it (0.8B: 107 ms) |

---

## 10. At 10x

Why this matters: it shows which parts scale by renting more and which change
shape or hit a hard wall.

**10x dataset (150k to 400k decisions).** Tokens: 160M to 430M per 2 epochs
(shared). Qwen3.5-4B fla: 46 A100-h (18 H100-h, about 70 USD at 3.95) per
target run. **The budget breaks on one run.** MiniCPM5: 19 A100-h, about 48 USD
per run, and the Mac would need 15 days. Labelling changes shape: cheap-API
bulk becomes 150 to 360 USD, so an **open-weight teacher is mandatory**
(27B: 128M tokens / 7.3k tok/s = 4.9 H100-h, about 20 USD). Every run exceeds
Modal's 24 h limit on one A100, so checkpoint-resume and possibly multi-GPU
(DDP; the author has `dist-train`) become required.

**16k-token states.** Training memory for 4B at 16k: 8.4 + 0.5 + 7.6 to 15
(activations) + 2 = 18 to 26 GB, so **40 GB minimum and 80 GB comfortable**.
Dense attention FLOPs at 16k: MiniCPM5 adds 8 x 42 x 16k x 2048 = 11e9, equal
to its 6N, so it costs **2x**. Qwen3.5-4B adds 8 x 8 x 16k x 4096 = 4.3e9
(+20%), so **the hybrid wins at long context**. Inference at 16k: Qwen3.5-4B
prefill about 16k / 700 = 23 s, MiniCPM5 about 16k / 1,200 = 13 s. Copy
branching at B=16: MiniCPM5 5 + 16 x 0.70 = 16.3 GB (86%); Qwen3.5-4B
8.4 + 16 x (0.54 + 0.05) = 17.8 GB (93%). **Both need shared-prefix
attention.** The engine changes shape from "copy the cache" to "one KV, many
query groups" (cascade attention).

**255 options.** Letters run out at 26. Qwen tokenizers split digits one per
token, so "0" to "254" are not single tokens. Multi-token option labels cost
255 x ~3 = 765 extra tail tokens per question: W1 becomes 16 x 800 = 12.8k tail
tokens, which takes 12.8k / 900 = 14 s on Qwen3.5-4B and 12.8k / 4,000 = 3.2 s
on the 0.8B (DERIVED). **Hard ceiling:** you need 255 distinct single-token
labels (reserved or added tokens with trained embeddings), or a scalar
cross-encoder that pays per option. Candidate-row loss and readout stay cheap
(255 rows).

**Hard ceilings list:** Mac Metal working set 19.07 GB (MEASURED); Modal 24 h
per function (ASSUMPTION from prior notes); 200 USD total (MEASURED
constraint); the 255-option cap (Jev spec); trained max length (untrained
lengths are unvalidated, RN:140-142).

---

## 11. Phase 0 benchmark plan (the load test for an ML project)

Why this matters: each item costs minutes or about 1 USD and decides a one-way
door before real money is spent.

Traffic shape to reproduce in every inference benchmark: state lengths drawn
from the real dataset's distribution (lognormal-ish, p50 1k, p99 12k, not a
fixed 1k); questions per request p50 8 / p99 64; options p50 4 / p99 26; a
**cold** first request after model load; and real weights and text, not random
tokens (branch-equivalence error depends on confidence).

| # | Measurement | Setup | Pass threshold | Decision it drives | Cost |
|---|---|---|---|---|---|
| P0-1 | Mac LoRA step, real weights: MiniCPM5-2B vs Qwen3.5-0.8B at 2k, positions-first loss, grad ckpt | `synth_bench.py train` pattern on real checkpoints | MiniCPM5 >= 250 tok/s at <= 70% working set (already 297 synthetic); Qwen3.5 >= 300 tok/s would reopen Mac training (expect ~60) | Mac as free trainer for pilots: dense yes, Qwen3.5 no | 30 min, 0 USD |
| P0-2 | CUDA Qwen3.5-4B LoRA, 50 steps, seq 2k, **with and without** fla + causal-conv1d; same for Qwen3-4B dense | 1 A100 80 GB or H100; log which kernel function runs (not only tok/s) | Qwen3.5-4B tok/s >= 0.6x Qwen3-4B (P <= 1.7) with kernels loaded | Qwen3.5-4B stays a candidate only if it passes; otherwise MiniCPM5-2B or Qwen3-4B | ~0.5 h, 1 to 2 USD |
| P0-3 | Training memory dry run at 1k / 4k / 8k (and 16k) per surviving candidate | 20 steps, `torch.cuda.max_memory_allocated`, positions-first head, branch training k = 4 and 16 | <= 70% of card memory at the chosen max length | GPU class to rent (24 vs 40 vs 80 GB); max trained length | in P0-2 session |
| P0-4 | Branch equivalence, Qwen3.5-0.8B real weights, MLX and torch; plus fla `initial_state` gradient check | prefill + B tails with KV + conv + recurrent copies vs isolated re-encode; finite-difference or autograd check of dh0 | isolation (alone vs batched) <= 1e-6; branched vs re-encode fp32 <= 1e-4 (pngwn 1e-7 in torch, RN:136), bf16 <= 2e-2 (dense MEASURED 1.3e-2) | Engine correctness gate; whether branch-training on Qwen3.5 is possible; fix REQ S10 thresholds | 1 h, 0 USD |
| P0-5 | Mac latency, real weights, W0 / W1 / W2 and 1k x 64, per candidate | 20 warm + 1 cold, `perf_counter`, `mx.get_peak_memory` | within 15% of section 6.1; W1 p50 <= 2.0 s at <= 13.3 GB | Confirms or replaces the synthetic numbers; ship-model choice (Q2, Q4) | 1 h, 0 USD |
| P0-6 | Batched naive-path anomaly (0.12 prob diff in fp32 at B=16 x 1,056) | reproduce on MLX; compare torch CPU on the same inputs | explained or reported upstream | Trust in MLX batched attention for eval | 30 min |
| P0-7 | Open-weight teacher throughput | vLLM, Qwen3.8-27B (or the chosen teacher), 1,000 real decisions, allowed_token_ids + logprobs | >= 3k tok/s prefill; prefix-cache hit rate >= 40% (vLLM metrics); startup <= 20 min | Open-weight vs API for bulk labels (Q3) | 0.5 h, 2 USD |
| P0-8 | Soak: MiniCPM5 real-weights LoRA on the Mac for 2 h, and the eval harness over 500 states | watch tok/s over time, memory growth, log size | tok/s drop <= 20% after 30 min; memory flat (no leak); logs < 50 MB/h | Whether overnight Mac training is dependable | 2 h, 0 USD |
| P0-9 | Dataset token stats of the first build | tokenize; report mean, p50, p95, p99 of states and d | mean state <= 1,300 keeps every number here; > 1,700 multiplies GPU hours by 1.7 | Rescales sections 3 and 6.4 | 10 min |
| P0-10 | API teacher probe | 200 decisions per candidate teacher | measured cost per decision within 2x of 0.00025 USD; 429 rate < 5% at chosen concurrency | Teacher choice, concurrency setting | < 1 USD |

**Soak rule:** any run longer than 1 h must show flat memory and flat tok/s
before it is allowed to run unattended overnight (REQ:386).

---

## 12. Which assumptions matter

Why this matters: benchmark the few constants that sit near a ceiling, and
leave the ones that sit far from any ceiling alone.

**Benchmark in Phase 0 (near a ceiling or large swing):**
- P, the Qwen3.5 CUDA kernel factor (1.4 vs 4 to 10). It swings the GPU line
  from 53% to 122%. P0-2.
- MFU 25% on A100/H100. It scales all GPU hours linearly, and 15% would make
  the fla Qwen3.5-4B plan 88% of the GPU line. P0-2.
- Mean state length (1,000 vs 1,770). It is linear in everything. P0-9.
- Decisions per state d. Shared-token saving runs from 1.9x at d = 2 to 11x
  at d = 16. It is a dataset design choice for sd-data.
- 24 GB fit at 4k for 4B models (61 to 78%). P0-3.
- Open-weight teacher prefill MFU and prefix caching on hybrids. P0-7.

**Safely far from any ceiling (do not bother):** tokenizer speed (4% of one
core), JSON and calibration time (0.14 ms of a 2,000 ms budget), disk
(< 20 GB of 628 GiB), egress (< 2 USD), checkpoint upload bandwidth, label
throughput at any API tier, MiniCPM5 and Qwen3.5-0.8B Mac latency (at or under
45% of budget), and isolation (MEASURED exactly 0).

---

## 13. Open questions and cross-agent needs

- `> TODO: unverified` Why the batched naive re-encode in MLX differs from
  isolated re-encode by 0.12 in fp32 at B=16 x 1,056 (P0-6).
- `> TODO: unverified` Whether fla `chunk_gated_delta_rule` returns the
  gradient w.r.t. `initial_state`. Branch training for Qwen3.5 depends on it.
- `> TODO: unverified` Whether vLLM prefix caching works for the chosen
  (possibly hybrid) teacher.
- `> TODO: unverified` Sustained (multi-hour) M5 throughput under thermal
  load.
- Spec-sheet peak FLOPs for A10G and L4, and every hourly price, are recalled
  and not fetched (sd-cost owns prices).
- REQ S10 thresholds (1e-3 bf16 for branch vs re-encode) look unreachable. The
  measured bf16 split error is 1.3e-2 on a dense 0.6B. Suggest redefining S10
  as isolation (MEASURED 0) plus same-split parity.
