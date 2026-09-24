# Requirements and NFRs: nanohunch

Mode: GREENFIELD. Depth: standard. Author: sd-requirements. Date: 2026-09-23.

Every number carries one label:

- `MEASURED`: observed by us on this machine today, or published by the named
  source (research notes say which). Published third-party numbers are
  "measured by them", never reproduced by us unless stated.
- `DERIVED`: arithmetic from labelled inputs; the arithmetic is shown.
- `ASSUMPTION`: a guess, always with a range and a falsification step (see the
  ledger in section 4).

Shared ground truth: `_brief/research-notes.md` (cited as `RN:<line>`).

---

## 0. How the NFR checklist was adapted

This is a trained model plus a single-user local inference engine plus an
eval harness, built by one person for learning and a portfolio. Several
web-service checklist items have no meaning here. Each was either kept,
replaced with its ML equivalent, or dropped, and the reason is stated.

| Checklist item | Treatment | Replaced by / reason |
|---|---|---|
| RPS avg / peak / peak factor | REPLACED | Inference workload envelope (state tokens, questions per request, options per Choice) plus batch-eval throughput. There is one caller (the author, a demo, the eval harness), concurrency 1. |
| Read:write ratio | REPLACED | Compute split: labelling vs training vs eval, in USD and GPU-hours. That split picks the architecture the way read:write does for a web service. |
| Payload sizes | KEPT, re-unitized | In tokens (state p50/p99, question length, option count), with bytes as a secondary figure. |
| Latency budget split across hops | KEPT | Hops are local: tokenize, state prefill, question-branch prefill, readout, serialize. |
| Data volume / growth / retention | KEPT | Labelled decisions, raw teacher responses, checkpoints. Growth is by deliberate dataset versions, not traffic. |
| Consistency | REPLACED | ML consistency: permutation invariance, per-question isolation within a batched request, determinism, MLX vs PyTorch parity. |
| Availability as a number | DROPPED | No uptime SLO (`RN:12`). Replaced by a reproducibility requirement (a reviewer can rerun the eval from a clean clone). A hosted demo is best effort. |
| RTO / RPO | REPLACED | Training-run durability: max GPU progress lost on preemption (RPO analog) and time to resume (RTO analog). Plus "label cache is never lost" because labels cost money. |
| Compliance / residency | KEPT, reframed | Licensing and terms of service for a public release (base model, datasets, teacher outputs). No residency constraint. No real PII in training data. |
| Budget ceiling | KEPT | 200 USD total, one-off, not monthly (`RN:17`). The binding constraint. |
| Team and operations | KEPT | Solo, ~20 h/week (`RN:19`). No on-call. Nothing may need babysitting overnight except resumable training runs. |
| Tenancy / AuthN / AuthZ | DROPPED | Single user, localhost. The local server binds to 127.0.0.1 only. |

---

## 1. Functional scope

**In one sentence, user's words:** "a model similar to TypeSafe Jev but with
open-source small models" (verbatim request), for learning and a portfolio.

**Scope paragraph.** nanohunch is three things plus their artefacts.
(a) A trained decision model, fine-tuned (LoRA) from an Apache-2.0 small open
base model (0.8B to 4B, `RN:72-76`), that takes a text `state` and a list of
typed questions and, without generating text, returns for each question a
probability distribution and a confidence: **Choice** (pick one of a
caller-supplied list of options, returns the argmax option, the full
distribution, and a confidence), **Score** (a level on a caller-defined ordinal
scale, default 0..4, returns level, distribution, confidence), and **Noul**
(yes/no, returns P(yes)) (`RN:34-39`). Probabilities are calibrated with a
per-type temperature fitted on a held-out calibration split. (b) An inference
engine, runnable locally on the M5 Mac through MLX, that encodes the state
once, then scores every question in parallel and in isolation by branching the
cached state (full-attention KV plus Gated DeltaNet conv and recurrent state
for Qwen3.5-class models, `RN:134-137`), exposed as a Python library, a CLI,
and a localhost HTTP server with a Jev-shaped JSON contract. (c) An evaluation
harness that measures accuracy against a teacher-consensus reference (Jev's
method, `RN:46-50`) and against public gold labels where they exist, and
measures calibration (ECE raw and T-scaled, NLL, Brier, reliability diagrams),
permutation consistency, set-size sensitivity, and accuracy by input-length
bucket, and compares against baselines: B0 the untouched base model read via
option-letter logits, B1 one cheap API LLM answering through a constrained
decision harness, B2 (if accessible) Jev itself on the same test set, and
published pngwn numbers as an external anchor (`RN:113-119`). Portfolio
artefacts: a Hugging Face model card (with calibration evidence Jev does not
publish, `RN:66-68`), a released labelled dataset (only the parts whose
licences allow it), an eval report, a reproducible repo, and a demo (local
Gradio or an HF Space).

**Callers.** The author (interactive), the eval harness (batch), a demo UI.
No third-party production callers.

**External dependencies, and whether they are on the inference critical path.**

| Dependency | Used for | On inference path? |
|---|---|---|
| Hugging Face Hub | base weights download, artefact publish | No (only first load) |
| Teacher LLM APIs (cheap tier, one mid/strong tier for eval reference) | labels | No (offline, cached) |
| Rented GPU provider (RunPod / Lambda / Modal / HF Jobs class) | training, open-weight teacher, GPU eval | No |
| mlx, mlx-lm | local inference and optional local LoRA | Yes (local library) |
| PyTorch + transformers (+ flash-linear-attention kernels, unverified) | GPU training | No |
| Jev API | baseline B2 only | No |

### Out of scope (explicit)

1. Text generation, chat, explanations or rationales in the output.
2. Reproducing RLCD itself. Its details are unpublished (`RN:44-45`); we use
   supervised soft-label distillation plus temperature scaling, and may add a
   calibration-aware loss. Any RL stage is stretch, not scope.
3. Multi-tenant serving, auth, rate limiting, autoscaling, uptime SLOs,
   paid API, billing.
4. Structured-state schemas beyond "state serialized to text" (JSON is passed
   as text). Native program-state typing is out.
5. Dependent / joint decisions solved inside the model. Dependencies are
   handled by caller code (`RN:143-145`), as Jev's workflows do.
6. Multimodal inputs (images, audio), even though some bases are multimodal.
7. Pretraining or continued pretraining. Full-parameter fine-tuning of 4B
   (LoRA only unless a run proves it unnecessary).
8. Models above 4B parameters as the shipped artefact (9B does not fit the
   latency or memory envelope, section 2.4). 27B+ models appear only as an
   optional rented teacher.
9. Masked-diffusion arm C (`RN:111`), iterative refinement.
10. Non-English evaluation.
11. Beating frontier LLMs on accuracy. The claim is cost/latency/calibration at
    acceptable accuracy, as Jev's is (`RN:51-61`).
12. Training on data whose licence forbids redistribution of a derived model
    (CC-BY-NC components, `RN:99-100`). Such data may be used for eval only.
13. Custom Metal kernels as a requirement. Allowed as a stretch learning item
    (author has `metal_kernels`, `RN:22-24`), not on the critical path.

---

## 2. Quantified requirements (adapted)

### 2.1 Hardware and environment facts

| Fact | Value | Label | Source |
|---|---|---|---|
| Local machine | Apple M5, 10 cores, 24 GB unified memory | MEASURED | `sysctl hw.memsize` = 25,769,803,776 B; `RN:15` |
| Python / uv | 3.14.7 / installed | MEASURED | `RN:20` |
| mlx / mlx-lm versions already used by the author | 0.31.2 / 0.31.3 | MEASURED | `gemma-mlx/pyproject.toml:9-10`, import check |
| mlx-lm has a Qwen3.5 model file and a Gated DeltaNet implementation | yes | MEASURED (code read, not run) | upstream `mlx_lm/models/qwen3_5.py`, `mlx_lm/models/gated_delta.py`, GitHub main 2026-09-23 |
| mlx-lm Qwen3.5 in training mode disables the Metal DeltaNet kernel and falls back to a per-timestep ops loop | yes | MEASURED (code read) | upstream `qwen3_5.py:192` (`use_kernel=not self.training`), `gated_delta.py:639-646` fallback to `gated_delta_ops` (`:557-602`, sequential loop) |
| Author has used Modal for rented GPUs before | yes | MEASURED | `mini-vllm/pyproject.toml:19` |

### 2.2 Measured Mac prefill anchors (run today)

`mlx_lm.benchmark`, bf16 weights from local HF cache, `-g 1 -n 3`, averages,
mlx 0.31.2, mlx-lm 0.31.3, M5 24 GB, 2026-09-23. These are dense Qwen3
transformers, not Qwen3.5 hybrids; they anchor the derivations below.

| Model | Prompt tokens | Prefill tok/s | Peak memory GB | Label |
|---|---|---|---|---|
| Qwen3-0.6B | 256 / 1,024 / 4,096 | 5,365 / 6,625 / 5,088 | 1.47 / 1.85 / 2.22 | MEASURED |
| Qwen3-1.7B | 256 / 1,024 / 4,096 | 1,878 / 2,447 / 2,199 | 3.64 / 4.00 / 4.57 | MEASURED |

DERIVED effective compute: 1.7B has ~1.4B non-embedding params (ASSUMPTION,
Qwen3 model card from memory): 2 x 1.4e9 x 2,447 = 6.9 TFLOP/s. 0.6B has
~0.44B non-embedding: 2 x 0.44e9 x 6,625 = 5.8 TFLOP/s. So the M5 GPU
sustains roughly 6 to 7 TFLOP/s bf16 on prefill through MLX today.

DERIVED prefill for candidates (tok/s = 6.5e12 / (2 x non-embedding params)):

| Candidate | Non-embedding params | Prefill tok/s at ~1k tokens | Label |
|---|---|---|---|
| Qwen3.5-0.8B | ~0.5B (ASSUMPTION 0.4 to 0.6B) | ~6,500 (4,000 to 8,000) | DERIVED |
| MiniCPM5-2B | 1.98B (`RN:76`) | ~1,650 (1,200 to 2,000) | DERIVED |
| Qwen3.5-4B | ~3.4B (4B LM minus 248,320 x 2,560 = 0.64B embeddings, `RN:74`) | ~950 (500 to 1,300) | DERIVED; range widened because DeltaNet kernel efficiency on M5 is unmeasured |

### 2.3 Inference workload envelope (replaces RPS / payload)

| Item | Value | Label | Source / arithmetic |
|---|---|---|---|
| Concurrent requests | 1 | ASSUMPTION (range 1 to 2) | single user, local demo |
| Reference workload W0 (short) | 256-token state, 8 questions | ASSUMPTION | defines "Jev-like" latency checks |
| Reference workload W1 (headline) | 1,024-token state, 16 questions, avg 32 tokens per question incl. options, up to 26 options | ASSUMPTION (state range 300 to 2,000) | typical ticket / trace / invoice size; pngwn PR states spanned 100 to 16k (`RN:138`) |
| Reference workload W2 (long) | 8,192-token state, 16 questions | ASSUMPTION | long-input bucket |
| State length p50 / p99 in eval set | 1,000 / 12,000 tokens | ASSUMPTION (p50 400 to 2,000; p99 6k to 16k) | to be measured once datasets exist |
| Questions per request | p50 8, p99 64 | ASSUMPTION (1 to 128) | Jev workflows decompose a case into many atomic questions (`RN:47-48`) |
| Options per Choice | p50 4, p99 26; hard cap 255 | cap MEASURED (Jev spec `RN:35`); distribution ASSUMPTION | |
| Score levels | default 5 (0..4), supported 2 to 11 | 5 MEASURED (Jev UI `RN:37`); range ASSUMPTION | |
| Request bytes | p50 ~5 KB, p99 ~60 KB | DERIVED | ~4 chars/token x tokens, ASSUMPTION on chars/token (3 to 5) |
| Response bytes | <= 4 KB per question (255 floats as JSON) | DERIVED | 255 x ~12 chars |
| Tokens processed for W1 with state sharing | 1,024 + 16 x 32 = 1,536 | DERIVED | |
| Tokens processed for W1 with naive re-encode | 16 x (1,024 + 32) = 16,896 | DERIVED | |
| Sharing speedup at W1 | 16,896 / 1,536 = 11x | DERIVED | pngwn measured 14.1x context removal at k=16 (`RN:128-129`) |

### 2.4 Latency and memory on the M5 Mac (local, bf16 unless stated)

Who feels it: a human at a demo, and the eval harness wall clock. Latency is
not the binding constraint for the project; the budget is. But latency is the
headline claim of a Jev-like system, so it is measured and reported.

**Latency budget split across hops, W1 on Qwen3.5-4B, p50 target 2.0 s:**

| Hop | Budget | Estimate | Label |
|---|---|---|---|
| HTTP parse + JSON validate | 5 ms | < 2 ms | ASSUMPTION |
| Tokenize state + 16 questions (~1.6k tokens) | 10 ms | 2 to 10 ms | ASSUMPTION (HF tokenizers ~0.3 to 1M tok/s) |
| State prefill, 1,024 tokens | 1,300 ms | 1,024 / 950 = 1.08 s (0.79 to 2.05 s) | DERIVED from 2.2 |
| Branch state 16x (copy KV for 8 attention layers + DeltaNet states) | 50 ms | 10 to 50 ms | ASSUMPTION (tens of MB copied, `RN:134-137`) |
| Question-branch prefill, 16 x 32 = 512 tokens batched | 600 ms | 512 / 950 = 0.54 s (0.39 to 1.0 s) | DERIVED |
| Readout: gather candidate logit rows, softmax, per-type temperature | 10 ms | < 5 ms | ASSUMPTION (26-row gather is free, `RN:130`) |
| Serialize response | 5 ms | < 2 ms | ASSUMPTION |
| Python / framework overhead | 20 ms | unknown | ASSUMPTION |
| **Total** | **2.0 s** | **1.6 s (1.2 to 3.1 s)** | DERIVED; fits p50 at the central estimate, misses at the pessimistic end |

Implication: a 4B model on the Mac is a ~1 to 2 s system for W1, not a
70 to 500 ms system. Jev's 70 to 500 ms (`RN:42`) on the Mac is only reachable
with a <= 1B model or short states: W1 on Qwen3.5-0.8B = 1,536 / 6,500 =
0.24 s DERIVED. This is a design-shaping number (see Q2).

| Requirement | MVP | Target | Stretch | Label |
|---|---|---|---|---|
| W1 p50, shipped model | <= 2.5 s | <= 2.0 s | <= 0.5 s (0.8B distilled variant or 4-bit + kernel work) | targets = requirement; feasibility DERIVED above |
| W1 p95 | <= 3.5 s | <= 2.5 s | <= 0.8 s | requirement |
| W0 p50 | <= 0.8 s | <= 0.5 s | <= 0.15 s | requirement; DERIVED 4B: (256 + 256) / 950 = 0.54 s |
| W2 p50 | <= 15 s | <= 10 s | <= 5 s | requirement; DERIVED 4B: 8,704 / 950 = 9.2 s (DeltaNet may beat this at long context) |
| Marginal cost of questions: latency(k=16) / latency(k=1) at W1 | <= 2.0 | <= 1.6 | <= 1.3 | DERIVED expectation (1,536 / 1,056) = 1.45 |
| Sharing speedup vs naive re-encode at W1 | >= 5x | >= 8x | >= 11x | DERIVED ceiling 11x |
| Peak process memory, W2 | <= 16 GB | <= 12 GB | <= 6 GB (4-bit) | requirement. DERIVED 4B bf16: 8 GB weights + 1.5 to 3 GB (overhead scaled from 1.7B measured 4.57 GB at 4k with ~3.4 GB weights) = 9.5 to 11 GB |
| Headroom rule | leave >= 8 GB of 24 GB for macOS + apps | | | ASSUMPTION: macOS lets Metal wire ~ 65 to 75% of RAM by default (16 to 18 GB) |

### 2.5 Throughput (replaces peak RPS)

| Requirement | Value | Label | Arithmetic |
|---|---|---|---|
| Full test-set eval of the shipped model on the Mac | <= 90 min | requirement | DERIVED 4B: 2,500 states x (1,000 + 8 x 32) tokens = 3.1M tokens / 950 tok/s = 55 min (40 to 105 min) |
| Eval of B0 (same base, no LoRA) | same as above | DERIVED | same forward cost |
| Teacher labelling throughput | >= 20k decisions per day of wall clock | ASSUMPTION on API rate limits (range 5k to 200k/day) | 20k / 86,400 s = 0.23 req/s if one decision per call; batching per state cuts calls 2 to 8x |
| Open-weight teacher on rented GPU (if used) | 20k decisions in <= 2 GPU-hours | DERIVED | 8k states x 1.3k tokens = 10M prefill tokens / 3k to 20k tok/s (ASSUMPTION for a ~30B MoE on one H100/A100 in vLLM) = 0.14 to 0.9 h |

### 2.6 Quality requirements

Reference definitions, fixed now so later numbers are comparable:

- **Consensus label** (Jev method, `RN:48-50`): average of the probability
  distributions from 2 teachers (distribution read from logits, from verbalized
  probabilities, or from 5 samples, decided by the architect). "Accuracy" =
  argmax(model) == argmax(consensus). Ties broken by the stronger teacher.
- **Gold label**: dataset ground truth for public sets (MMLU-Pro, HotpotQA,
  BoolQ). Reported separately; never mixed with consensus accuracy.
- **ECE**: top-label ECE, 15 equal-width bins, also 15 equal-mass bins;
  both reported. Temperature fitted per question type on the cal split,
  applied to test. pngwn's binning is not published, so comparisons to
  their ECE are approximate (ASSUMPTION they used a similar top-label ECE).
- **NLL / Brier**: against consensus argmax (hard) and consensus distribution
  (soft cross-entropy), both reported.
- **Order consistency**: for Choice items with >= 3 options, re-score under
  reversed order and 3 random permutations. Report top-1 agreement and gold
  rank flip rate (pngwn's metric, `RN:124-125`) and mean total-variation
  distance between mapped distributions.
- **Isolation**: probabilities for question i must not change when other
  questions are added to the same request.
- **Long input**: accuracy by state-length bucket 0-512, 512-2k, 2k-8k,
  8k-16k tokens.
- **Paired significance**: model vs baseline on the same items, paired
  bootstrap (10k resamples) 95% CI on the accuracy difference.

| Metric (test split unless noted) | Anchor | MVP | Target | Stretch | Label |
|---|---|---|---|---|---|
| Accuracy vs consensus, in-distribution test | B0 zero-shot (unknown until measured) | >= B0 + 3 pts, CI excludes 0 | >= B0 + 6 pts | within 3 pts of the cheaper teacher itself | requirement relative to a Phase 0 measurement |
| Accuracy on pngwn `typed-decisions-v2` test (eval-only, not trained on) | A 0.807, B 0.752 (`RN:115`) | >= 0.70 | >= 0.752 (match arm B) | >= 0.807 (match arm A) | anchors MEASURED by pngwn; targets requirement. Lower bar because it is out of distribution for us (we do not train on its NC ticket component) |
| MMLU-Pro-style Choice slice, gold | A 0.625, B 0.415 (`RN:122`) | >= 0.50 (4B) | >= 0.60 | >= 0.65 | anchors MEASURED by pngwn |
| HotpotQA-style Noul slice, gold | A 0.939, B 0.870 (`RN:123`) | >= 0.85 | >= 0.90 | >= 0.94 | anchors MEASURED by pngwn |
| ECE after T-scaling, overall | A 0.026, B 0.015, PR eval 0.047 (`RN:118,139`) | <= 0.05 | <= 0.03 | <= 0.015 | anchors MEASURED by pngwn |
| ECE after T-scaling, per type (Choice, Score, Noul) | none published | each <= 0.07 | each <= 0.04 | each <= 0.02 | requirement |
| ECE raw (before T) | A 0.069, B 0.064 | report only | <= 0.06 | <= 0.03 (calibration-aware loss) | requirement |
| NLL T-scaled vs consensus argmax | A 0.518, B 0.640 (`RN:117`) | < B0 T-scaled NLL | <= 0.60 on pngwn test | <= 0.52 | anchors MEASURED by pngwn |
| Brier (multi-class) | none | < B0 | report | report | requirement |
| Order: gold rank flip rate under reversal | B 37.5% (`RN:124`) | <= 15% | <= 5% | 0% (invariant by construction) | anchor MEASURED by pngwn |
| Order: top-1 agreement under reversal and 3 permutations | B 0.664 (`RN:125`) | >= 0.90 | >= 0.95 | 1.00 | anchor MEASURED by pngwn |
| Set size: gold probability mass at 26 candidates (pngwn distractor protocol) | A 0.49, B 0.61 (`RN:126`) | >= 0.49 | >= 0.61 | >= 0.70 | anchors MEASURED by pngwn |
| Long input: accuracy in >2k bucket | base beat fine-tune above 2k in pngwn PR eval (`RN:140-142`) | >= B0 - 1 pt on the same bucket | >= B0 + 3 pts | >= B0 + 6 pts | requirement |
| Isolation: max abs prob diff, alone vs in a 16-question batch | pngwn 1e-7 fp32 (`RN:136`) | <= 1e-3 bf16 | <= 1e-4 bf16 | <= 1e-6 fp32 | requirement |
| Parity MLX engine vs PyTorch reference, same checkpoint | none | max abs prob diff <= 2e-2 bf16 | <= 5e-3 | <= 1e-4 fp32 | requirement |
| Determinism: rerun same input same device | none | identical to 1e-6 | identical | identical | requirement |
| Beat B1 (cheap API LLM, decomposed, single call per question) on cost | Jev 0.0004 vs luna 0.0033 USD/case (`RN:57-59`) | report | >= 10x cheaper per decision at <= 5 pts lower accuracy | >= 50x cheaper | requirement; our marginal cost is Mac electricity, so the claim is on latency and $0 marginal |

### 2.7 Model, context, options

| Requirement | MVP | Target | Stretch | Label |
|---|---|---|---|---|
| Shipped model size | 2B to 4B LoRA-merged, bf16 | 4B bf16 + 4-bit MLX export | plus 0.8B distilled variant | requirement; knowledge scales with params (`RN:121-123`) |
| Trained sequence length (max) | 4,096 | 8,192 | 16,384 | requirement; pngwn's 384-token truncation is the failure to avoid (`RN:140-142`) |
| Supported context at inference | 8,192 | 16,384 | 32,768 | requirement; base supports 131k to 262k (`RN:74,76`) but untrained lengths are unvalidated |
| Max options per Choice | 26 (letters, one logits row) | 64 | 255 (Jev cap) | requirement; letters up to 26 are free, multi-token options cost 2 to 4 ms each on A100 (`RN:130-131`) |
| Question types | Choice, Noul, Score | same | plus multi-select as N Nouls in caller | requirement |
| Output fields per question | answer, distribution, confidence (= max prob after T) | same | plus entropy | requirement mirroring Jev (`RN:34-39`) |

### 2.8 Data volume, growth, retention

| Item | Value | Label | Arithmetic / source |
|---|---|---|---|
| Train decisions (MVP) | 15k (range 8k to 25k) | ASSUMPTION | pngwn used 45,932 (`RN:96-97`); we need fewer because we start from a stronger base and target "beat B0", see A5 |
| Train decisions (target) | 30k to 50k | ASSUMPTION | matches pngwn scale |
| Mix | ~40% public gold-labelled (MMLU-Pro, HotpotQA, BoolQ class), ~60% teacher-labelled states across 4 to 8 synthetic workflows modelled on Jev's (`RN:46`) | ASSUMPTION (public share 20 to 60%) | |
| Decisions per state | ~1.5 to 4 | ASSUMPTION | pngwn 45,932 / 31,109 = 1.48 DERIVED; Jev-style decomposition gives more |
| Cal split | >= 2,000 decisions | DERIVED | T is one scalar per type; 3 types x ~700 is ample |
| Test split | >= 3,000 decisions (target 5,000) | DERIVED | accuracy SE at p = 0.8, n = 3,000: sqrt(0.16 / 3,000) = 0.0073, 95% CI +/- 1.4 pts; ECE finite-sample bias with 15 bins at n = 3,000 is ~0.01 (ASSUMPTION), so an ECE target of 0.03 needs n >= 3,000 |
| Human audit set | 200 to 300 decisions labelled by the author | requirement | ~3 to 5 h at 1 min per decision; validates consensus (A1) |
| Training tokens per epoch | ~10M (range 4M to 30M) | DERIVED | 8k states x ~1,000 tokens + questions; range from state count 4k to 15k and mean length 600 to 2,000 |
| Raw teacher responses (jsonl, kept forever) | ~250 MB | DERIVED | 50k decisions x ~5 KB |
| LoRA adapters per run | 20 to 80 MB | ASSUMPTION | r16 on attention + MLP projections of a 4B |
| Merged model on HF | ~8 GB bf16, ~2.5 GB 4-bit | DERIVED | 4B x 2 B; 4B x ~0.6 B |
| Growth | versioned dataset releases (v1 MVP, v2 target); no organic growth | requirement | |
| Retention | raw teacher responses, prompts, seeds, split hashes: forever (they cost money and are the reproducibility root). Intermediate checkpoints: last 3 per run | requirement | |

### 2.9 Budget and compute split (replaces read:write)

Total ceiling 200 USD, one-off (`RN:17-18`). MEASURED constraint.

| Line | Allocation | Label | Notes |
|---|---|---|---|
| Teacher API labels | <= 60 USD | requirement | DERIVED below: bulk labels are cheap, strong-tier eval reference is the expensive part |
| GPU rental (training, open-weight teacher, GPU eval) | <= 90 USD | requirement | |
| Reserve (reruns, mistakes, idle billing) | >= 50 USD (25%) | requirement | a forgotten GPU at 2 USD/h for a weekend = ~100 USD, so the reserve is real, not padding |
| MVP must fit in | <= 120 USD | requirement | leaves 80 for target/stretch |

**Teacher cost per decision (DERIVED from ASSUMPTION prices):**

- Cheap tier (luna / DS v4 flash class): input 0.10 to 0.50 USD/M tokens,
  output 0.40 to 2.00 USD/M (ASSUMPTION A6). One call per state with ~4
  questions: ~1,300 input + ~300 to 800 output tokens.
  Cost per call = 1,300 x 0.3e-6 + 500 x 1.2e-6 = 0.0010 USD (range 0.0003 to
  0.0023). Per decision = 0.00025 USD (0.00008 to 0.0006).
- Cross-check against Jev's published per-case cost (`RN:57-60`): luna 0.0033
  USD per case, if a case has 10 to 40 questions that is 0.00008 to 0.0003 per
  question. Consistent with the above.
- Two cheap teachers on 30k decisions: 2 x 30k x 0.00025 = 15 USD (5 to 36).
- Strong tier (sol / opus class, 25 to 55x luna per case, `RN:54-55`) on a
  2,000-decision eval reference subset: ~0.004 to 0.01 per decision =
  8 to 20 USD.
- Therefore at cheap tier, labels are not the binding cost. GPU time and
  author time are. At strong tier for bulk labels, 60 USD buys only
  6k to 15k decisions: do not use strong tier for bulk.

**GPU hours (DERIVED from ASSUMPTION A4, A100 80GB at 1.2 to 2.5 USD/h):**

- 90 USD buys 36 to 75 A100-hours.
- pngwn arm B (0.6B, 46k decisions, 2 epochs, <= 384 tokens) took 68 min on
  one A100 (`RN:107-109`). Implied ~2 x 46k x ~350 tokens = 32M tokens in
  4,080 s = 7.9k tok/s training throughput (DERIVED; ~6% MFU at 4N FLOPs/token,
  overhead-dominated as pngwn notes `RN:132-133`).
- 4B LoRA training FLOPs ~ 4 x 3.4e9 x tokens (ASSUMPTION: LoRA skips frozen
  weight grads, ~4N rather than 6N). For 2 epochs x 10M tokens: 2.7e17 FLOPs.
  At 30 to 100 TFLOP/s effective on A100 (10 to 30% MFU; lower if Qwen3.5
  DeltaNet falls back to reference kernels, `RN:137`): 0.8 to 2.5 h. With a
  3x penalty for slow DeltaNet kernels: up to 7.5 h. So one full 4B run =
  1 to 8 A100-hours = 1.2 to 20 USD. DERIVED.
- Budgeted GPU plan: 3 to 5 pilot runs on 0.8B/2B (<= 1 h each), 2 to 4 full 4B
  runs, 2 to 6 h open-weight teacher and GPU evals. Total 12 to 50 A100-hours
  = 15 to 125 USD. The upper end breaks the 90 USD line: kernel speed (A3) is
  the swing factor.
- Requirement: every full training run <= 8 A100-hours and <= 20 USD, with a
  hard stop (max_steps and a provider-side auto-shutdown).

**Mac as free compute (DERIVED, low confidence):** 2B dense LoRA training at
~1/3 of prefill speed = ~550 tok/s; 10M tokens = 5 h per epoch overnight.
Qwen3.5 in mlx-lm training mode uses the sequential DeltaNet ops loop
(`qwen3_5.py:192`, `gated_delta.py:639-646`), so 4B Qwen3.5 training on the Mac
is expected to be much slower than this estimate. Mac training is for
debugging and small pilots unless a Phase 0 benchmark says otherwise.

### 2.10 Consistency, durability, reproducibility (replaces consistency / availability / RTO / RPO)

| Requirement | Value | Label |
|---|---|---|
| GPU progress lost on preemption or crash (RPO analog) | <= 20 min of GPU time: checkpoint LoRA + optimizer + data cursor every <= 20 min to durable storage (HF Hub private repo or provider volume) | requirement |
| Time to resume a run (RTO analog) | <= 15 min from a fresh instance, one command | requirement |
| Label cache durability | every teacher request and raw response stored append-only, content-addressed by (teacher, prompt hash); never re-paid for the same item; backed up off the laptop | requirement |
| Idempotent labelling | rerunning the labeller skips cached items; a crash mid-run loses at most the in-flight batch | requirement |
| Pinned environment | `uv.lock` committed; model revisions pinned by HF commit hash; teacher model IDs and dates recorded | requirement |
| Seeds | fixed seeds for splitting (by state, never by decision, so no state leaks across splits), training, permutation tests | requirement |
| Rerun tolerance | eval rerun from the released checkpoint reproduces headline metrics within +/- 0.002 accuracy and +/- 0.002 ECE on the Mac | requirement |
| Clean-clone reproduction | reviewer reproduces the eval report from the released checkpoint in <= 2 h on a 24 GB Apple Silicon Mac, no API keys needed | requirement |
| Demo availability | best effort, no SLO | requirement |

### 2.11 Licensing and compliance (public release)

| Item | Status | Label | Consequence |
|---|---|---|---|
| Base models Qwen3.5, MiniCPM5, Gemma 4 | Apache-2.0 | MEASURED (HF cards via `RN:72-78`) | release allowed with attribution |
| LiquidAI LFM2.5 family | LFM Open License v1.0 | MEASURED (`RN:78-79`) | terms not reviewed; avoid unless reviewed |
| pngwn ticket component | CC-BY-NC-4.0 (`RN:99-100`) | MEASURED by source | eval only; do not train the public model on it |
| MMLU-Pro, HotpotQA, BoolQ, MNLI, ANLI licences | recalled as MIT / CC BY-SA / CC BY-SA / mixed / CC BY-NC | ASSUMPTION (unverified) | verify each before training; NC ones become eval-only |
| Teacher API terms: using outputs to train a model that is then released | unknown per provider; some commercial terms restrict training competing models | ASSUMPTION (A9) | may force open-weight teachers for bulk labels |
| TypeSafe `system-one-adapter-python` licence | unverified (`RN:63-65`) | ASSUMPTION | only reuse code if licence permits |
| PII | synthetic states only; no real customer data; if real GitHub text is used, keep repo licences and strip emails | requirement | |
| Residency | none | requirement | |
| Naming | do not use "Jev" or TypeSafe marks in the model name; describe as "a Jev-style open replication" | requirement | |

### 2.12 Team and time

| Item | Value | Label |
|---|---|---|
| Engineers | 1 | MEASURED (`RN:19`) |
| Hours per week | 20 (range 12 to 25 realized) | ASSUMPTION A7 (stated "20+", realized hours usually lower) |
| Calendar to MVP | <= 6 weeks (~100 to 120 h) | requirement |
| Calendar to target | <= 12 weeks (~200 to 240 h) | requirement |
| On-call | none; nothing may need a human overnight except a resumable training job with auto-shutdown | requirement |
| Familiar tech (reduces novelty tax) | MLX, inference engines with paged KV, PyTorch, Modal, distributed training | MEASURED from project names (`RN:21-24`), `mini-vllm/pyproject.toml:19` |

### 2.13 One-way doors (flagged for the architect)

1. **Readout head design** (causal letter-logit scorer vs scalar cross-encoder
   vs hybrid). Decides order invariance, max options, engine design, and what
   the MLX engine must implement. Hard to change after training runs are paid.
2. **Base model family** (Qwen3.5 hybrid vs plain Llama-style MiniCPM5).
   Decides whether the engine must branch DeltaNet recurrent state, and which
   kernels are slow on each platform.
3. **Teacher choice for bulk labels** (API vs open-weight). Decides whether
   the dataset and model can be released (A9). Relabelling later costs budget.
4. **Split key** (by state). Changing it later invalidates every past eval.
5. **Public JSON contract** of the local server (Jev-shaped). Cheap to change
   now, costly after the demo and model card reference it.

---

## 3. SLIs and success criteria

All SLIs are measured by the eval harness on frozen splits, committed as a
JSON report plus plots. "Where measured" means which split and which runtime.

| # | SLI | How / where measured | MVP threshold | Target | Stretch | Window |
|---|---|---|---|---|---|---|
| S1 | Accuracy vs consensus, delta over B0 | test split, MLX engine, paired bootstrap | >= +3 pts, CI > 0 | >= +6 pts | within 3 pts of cheap teacher | per release |
| S2 | ECE after T-scaling, overall and per type | test split, T fit on cal, 15 equal-width and equal-mass bins | <= 0.05 overall | <= 0.03 overall, each type <= 0.04 | <= 0.015 | per release |
| S3 | Order consistency | Choice items with >= 3 options, reversed + 3 permutations | top-1 agreement >= 0.90, flip rate <= 15% | >= 0.95, <= 5% | 1.00, 0% | per release |
| S4 | Long-input robustness | accuracy in >2k bucket vs B0 same bucket | >= B0 - 1 pt | >= B0 + 3 | >= B0 + 6 | per release |
| S5 | External anchor | pngwn typed-decisions-v2 test, eval only | >= 0.70 acc | >= 0.752 | >= 0.807 | once |
| S6 | Consensus validity | author's 200 to 300 decision audit: agreement of consensus with author | >= 80% | >= 85% | >= 90% | once, before training |
| S7 | Mac latency | W1 and W0, 20 warm runs + 1 cold, `time.perf_counter` around the engine call, report p50/p95 | W1 p50 <= 2.5 s | <= 2.0 s | <= 0.5 s | per release |
| S8 | Question marginal cost | latency(k=16) / latency(k=1) at W1 | <= 2.0 | <= 1.6 | <= 1.3 | per release |
| S9 | Memory | peak `mx.get_peak_memory()` at W2 | <= 16 GB | <= 12 GB | <= 6 GB | per release |
| S10 | Isolation and parity | alone vs batched; MLX vs PyTorch same checkpoint | 1e-3 / 2e-2 | 1e-4 / 5e-3 | 1e-6 / 1e-4 (fp32) | CI test on every engine change |
| S11 | Spend | provider invoices + API dashboards, logged in `spend.csv` | MVP <= 120 USD | total <= 200 USD | total <= 150 USD | cumulative, checked weekly |
| S12 | Reproducibility | clean clone, released checkpoint, Mac, no keys | report regenerated within tolerance in <= 2 h | same | plus training rerun from cached labels reproduces S1 within 1 pt | per release |
| S13 | Artefacts shipped | HF model card with reliability diagrams and per-type ECE, eval report, dataset card, demo | card + report + local demo | plus public dataset (licence-clean parts) + HF Space | plus a blog-style write-up with ablations | once |

**Minimum viable success (fits inside 120 USD and ~6 weeks):** a LoRA-trained
2B-to-4B model and an MLX engine that together hit S1, S2, S3, S4, S7, S9,
S10, S11 at MVP thresholds, with S6 >= 80% so the reference is trusted, and
S13 at MVP. Concretely: beats its own untouched base model by >= 3 points on
consensus accuracy with a significant paired CI, is calibrated to ECE <= 0.05
after temperature scaling, does not flip on option order more than 15% of the
time (vs 37.5% in the only open prior art), does not regress on long inputs,
answers 16 questions over a 1k-token state in <= 2.5 s on the M5 in <= 16 GB,
and anyone can reproduce the report from a clean clone.

**Project failure (kill signals, for the planner):** after the first full 4B
run, if S1 is below +1 pt over B0 with 15k decisions, or if 60% of the budget
is spent before a model passes S2 at MVP, stop scaling and pivot per Q1/Q2.

---

## 4. Assumption ledger (sorted by blast radius)

Owner for everything is the author (`@umang`) unless noted; "Phase 0" means
before any paid training run.

| # | Assumption | Value (range) | Blast radius if wrong | How to falsify | Owner |
|---|---|---|---|---|---|
| A1 | Teacher-consensus labels are a good enough reference for "accuracy" (Jev's method, `RN:48-50`) | consensus agrees with a careful human on >= 85% of decisions (range 65% to 95%) | Every accuracy number in the report means "agrees with two LLMs"; below ~75% the S1 delta is noise and the portfolio claim collapses. Also contaminates training targets | Author labels 200 to 300 stratified decisions blind (3 to 5 h), computes agreement and Cohen's kappa per type; also measure teacher-teacher agreement on 1k items (if < 80%, consensus is unstable) | @umang |
| A2 | Budget: 200 USD total covers labels + GPU for MVP and target | MVP 60 to 120 USD, target 120 to 200 USD | Project stops mid-way; no released model | Track `spend.csv` weekly; after first full run, extrapolate remaining runs; kill rule at 60% spend (section 3) | @umang |
| A3 | Qwen3.5 hybrid (Gated DeltaNet) trains at acceptable speed on rented GPUs via transformers + flash-linear-attention, and LoRA works with its layers | within 1x to 3x of a dense model of equal size per token | pngwn hit slow reference PyTorch kernels (`RN:137`); a 3 to 10x slowdown pushes a 4B run to 8 to 25 A100-h, breaking the GPU budget; forces MiniCPM5-2B (plain Llama) or Qwen3-4B-class dense base | Phase 0: 50-step LoRA run of Qwen3.5-4B and MiniCPM5-2B at seq 2k on one rented A100/H100 (~0.5 h, ~1 USD), log tokens/s and confirm fla kernels load (not reference path) | @umang |
| A4 | Rented GPU hourly price | A100 80GB 1.2 USD/h (0.9 to 2.5); H100 2.5 (1.8 to 4.0); L40S/A6000 48GB 0.8 (0.5 to 1.2); RTX 4090 24GB 0.4 (0.3 to 0.7) | 2x error halves the number of runs (36 to 75 A100-h becomes 18 to 37) | Check RunPod, Lambda, Modal, HF Jobs, Vast pricing pages on the day of booking; record in `spend.csv` | @umang |
| A5 | Labelled decisions needed to beat zero-shot B0 by >= 3 pts | 5k to 15k (range 2k to 50k) | If > 30k with strong-tier labels, budget fails; if the base already matches teachers zero-shot, the fine-tune adds only calibration, which changes the portfolio claim (pngwn saw base >= fine-tune on judgement questions, `RN:140-141`) | Learning curve on the 0.8B or 2B pilot: train on 1k / 3k / 10k, eval S1; extrapolate to 4B. Measure B0 first, it may already be strong | @umang |
| A6 | Teacher per-token prices (2026 models, not fetched `RN:159`) | cheap tier in 0.10 to 0.50 USD/M, out 0.40 to 2.00 USD/M; mid tier ~10x; strong tier 25 to 55x luna per case (`RN:54-60`) | If 5x higher, bulk labels cost 25 to 180 USD for 30k decisions: must switch to open-weight teacher or fewer labels | Read the providers' pricing pages; label 200 decisions with each candidate teacher and divide the invoice by 200 | @umang |
| A7 | Author time actually available | 20 h/week (12 to 25 realized) | At 12 h/week MVP slips from 6 to ~10 weeks; the risk is stalling before the eval harness exists, leaving no portfolio artefact | Log hours weekly for 2 weeks; if < 15, cut target scope (drop 0.8B distill, 255 options, HF Space) | @umang |
| A8 | Teacher API terms allow training on outputs and releasing the resulting model/dataset for non-commercial use | allowed for at least one cheap-tier teacher (probability 50 to 80%) | If not, the released dataset and model violate ToS: must relabel with open-weight teachers (+2 to 6 GPU-h, 3 to 15 USD) or not publish the dataset | Read each provider's terms (output use, competing models) before labelling; prefer open-weight teacher (Apache/MIT weights) for anything to be released | @umang |
| A9 | 4B LoRA fine-tune memory fits with the candidate-row loss trick (`RN:109-110`) and gradient checkpointing | 24 GB GPU: fits at seq <= 2k to 4k, batch 1; 40 GB: seq <= 8k; 80 GB: seq <= 16k | If 24 GB does not fit, the cheapest cards are unusable (+50 to 150% cost/h); if 80 GB does not fit 16k, long-input training caps at 8k | Phase 0 dry run: 20 steps at seq 1k/4k/8k/16k, record `torch.cuda.max_memory_allocated`. Arithmetic prior: 8 GB bf16 weights + ~0.3 GB checkpointed activations per 2k tokens + full-vocab logits avoided (248,320 x 2k x 4 B = 2 GB per sequence) | @umang |
| A10 | mlx-lm runs Qwen3.5-4B inference on the M5 and supports LoRA on its layers; training mode uses a sequential DeltaNet ops loop | inference yes (code present); LoRA functional but 3x to 30x slower than dense per token | If inference fails or is wrong, the engine needs a custom port (author has done ports: `gemma-mlx`); Mac training is off the table regardless | Load Qwen3.5-4B in mlx-lm, compare last-layer logits to transformers on 5 prompts (max abs diff); run `mlx_lm.lora` 10 steps at seq 1k and record it/s. Code evidence: `qwen3_5.py:192`, `gated_delta.py:639-646` | @umang |
| A11 | Qwen3.5-4B prefill on M5 via MLX | ~950 tok/s (500 to 1,300) | Below 600 tok/s, W1 misses the 2.0 s target (becomes ~2.6 s); pushes toward 2B or 4-bit | `mlx_lm.benchmark --model Qwen/Qwen3.5-4B -p 1024 -g 1` (5 min); anchors measured today in 2.2 | @umang |
| A12 | Order sensitivity is fixable for a causal letter scorer with permutation augmentation (and optional test-time permutation averaging) | flip rate 37.5% -> 5 to 15% | If not, S3 forces the scalar cross-encoder (per-option cost, set-size degradation `RN:126`) or averaging over 2 to 4 permutations (2 to 4x branch cost) | Pilot on 0.8B: train with and without random option shuffles, measure S3 | @umang |
| A13 | Training at realistic lengths (up to 4k to 8k) fixes the long-input regression seen in pngwn | >2k bucket >= B0 | If not, long-input claim fails; may need length-balanced sampling or longer training, +30 to 100% GPU cost | Bucketed eval of pilot runs trained at 512 vs 4k max length | @umang |
| A14 | Test set of >= 3,000 decisions resolves the targets | accuracy CI +/- 1.4 pts; ECE bias ~0.01 | Smaller sets make ECE 0.015 vs 0.03 indistinguishable, invalidating S2 stretch claims | Bootstrap CI on ECE from the test set; if CI width > 0.01, grow test set | @umang |
| A15 | macOS lets Metal use ~65 to 75% of 24 GB (16 to 18 GB) by default | 16 to 18 GB | If lower, 4B bf16 at 16k context may OOM; forces 4-bit or 8k cap | `mx.metal.device_info()` recommended working set; load 4B and run W2 | @umang |
| A16 | Public dataset licences are as recalled (MMLU-Pro MIT, HotpotQA/BoolQ CC BY-SA, ANLI CC BY-NC) | 1 to 2 of 5 turn out restrictive | Training mix shrinks by 10 to 30%; public release needs re-mix | Read each dataset card before inclusion; record licence in the dataset card | @umang |
| A17 | Jev API is accessible to an individual for baseline B2 | accessible, cost < 1 USD for the test set (DERIVED: ~4M input tokens x 0.042 USD/M = 0.17 USD, `RN:43`) | Without it, the headline comparison is against pngwn and B0/B1 only, which is weaker but fine | Try to sign up at typesafe.ai; run 10 questions | @umang |
| A18 | Mean state length in our datasets | ~1,000 tokens (400 to 2,000) | Scales training tokens, GPU cost, and eval time linearly | Tokenize the first dataset build; report p50/p95/p99 | @umang |
| A19 | pngwn test set can be used as an eval-only anchor (NC licence, non-commercial portfolio) | usable with attribution | If not, drop S5; lose the only direct numeric comparison | Read dataset licence and card; email author if ambiguous | @umang |
| A20 | Branching overhead (state copy) is small on MLX | 10 to 50 ms for 16 branches | If copying DeltaNet + KV state is slow (e.g., 200+ ms), S8 target fails; engine needs shared-prefix attention instead of copies | Microbenchmark copy of the cache after a 1k prefill on the Mac | @umang |

---

## 5. Ranked open questions

Ranked by how much the design changes with the answer.

| Rank | Question | Who answers | If answer is X | If answer is Y |
|---|---|---|---|---|
| Q1 | Is order invariance a hard requirement (guaranteed by construction) or a statistical target (S3 >= 0.95)? | author (with architect) | Hard: scalar cross-encoder (pngwn arm A style) or a permutation-invariant option encoding; per-option compute, 255 options costly, set-size weakness (`RN:126`); engine scores (state, question, option) rows | Statistical: causal letter scorer (arm B style) with permutation augmentation; one logits row per question, 26 options free, simplest engine; possibly 2-permutation averaging |
| Q2 | Which matters more for the portfolio headline: Jev-like latency (<= 500 ms on the Mac) or accuracy / knowledge? | author | Latency: ship 0.8B to 2B (W1 0.24 to 0.9 s DERIVED), accept knowledge gap (MMLU-Pro slice likely < 0.45 per arm B `RN:122`) | Accuracy: ship 4B (W1 ~1.1 to 2 s), latency reported honestly; optional 0.8B distilled variant as stretch |
| Q3 | Will the dataset and model be publicly released (HF) and is that compatible with the chosen teachers' terms? | author, after reading terms (A8) | Release both: bulk labels from open-weight teacher on rented GPU (soft labels from logits for free, `RN:160-162`), API teachers only for eval reference | Release model only or keep data private: cheap API teachers for bulk; faster start, no GPU teacher setup |
| Q4 | Base family: Qwen3.5 hybrid or plain-attention MiniCPM5-2B / a dense 4B? Depends on A3, A10 Phase 0 results | Phase 0 measurement | Qwen3.5-4B: best knowledge (`RN:76`: MMLU-Pro 78.0 vs 70.8), but engine must branch DeltaNet state and kernels may be slow on both CUDA and Mac training | MiniCPM5-2B: official MLX build, plain KV branching (reuse `mini-vllm` ideas), faster everywhere, ~7 pts less knowledge |
| Q5 | What is the reference for "accuracy": teacher consensus only, or consensus plus a human audit plus gold public labels? | author | Consensus only: matches Jev methodology, cheapest, weakest claim | Plus audit and gold: +3 to 5 h author time, strongest portfolio claim, may reveal teacher disagreement that changes which teacher labels bulk data |
| Q6 | Which eval workflows: replicate Jev's four domains (security incidents, agent traces, invoices, customer service `RN:46`) synthetically, or use public datasets plus 1 to 2 real-text workflows (e.g., GitHub PRs like pngwn)? | author | Jev's four: direct methodological comparison to Jev's table, but we must synthesize states (teacher cost for generation, +10 to 30 USD) | Public + real text: cheaper, more honest distribution, but no apples-to-apples with Jev's published table |
| Q7 | Max options per Choice to support in v1: 26 or 255? | author | 26: single logits row, trivial engine, pngwn distractor protocol comparable | 255: multi-token option labels or option-span scoring (2 to 4 ms per option on A100 `RN:131`, more on Mac), set-size eval up to 255 |
| Q8 | Hard calendar deadline (job applications, talk)? | author | Deadline <= 6 weeks: MVP only, 2B base, cheap teachers, no distill | No deadline: target tier, 4B, ablations, blog write-up |
| Q9 | Should the local server speak Jev's exact JSON schema / work with `system-one-adapter-python`? | author, after licence check | Yes: drop-in comparisons and B1 harness reuse; tie to their contract | No: own minimal schema, free to change |
| Q10 | Is Jev API accessible (A17)? | author | Yes: add B2, the most compelling head-to-head for under 1 USD | No: compare to pngwn + B0 + B1 only |

---

## 6. Numbers downstream agents should copy

| Name | Value | Label |
|---|---|---|
| Budget total / MVP / reserve | 200 / 120 / >= 50 USD | MEASURED constraint / requirement / requirement |
| Mac prefill, Qwen3-1.7B bf16 @1k | 2,447 tok/s, 4.0 GB peak | MEASURED |
| Mac prefill, Qwen3-0.6B bf16 @1k | 6,625 tok/s, 1.85 GB peak | MEASURED |
| Mac effective prefill compute | 6 to 7 TFLOP/s | DERIVED |
| Qwen3.5-4B Mac prefill | ~950 tok/s (500 to 1,300) | DERIVED |
| W1 latency, 4B | ~1.6 s (1.2 to 3.1) | DERIVED |
| W1 latency, 0.8B | ~0.24 s | DERIVED |
| State sharing speedup, W1 | 11x ceiling | DERIVED |
| 4B bf16 inference peak memory at 8k | 9.5 to 11 GB | DERIVED |
| Teacher cost per decision, cheap tier | 0.00025 USD (0.00008 to 0.0006) | DERIVED from ASSUMPTION |
| 4B LoRA full run | 1 to 8 A100-h, 1.2 to 20 USD | DERIVED from ASSUMPTION |
| Train decisions MVP / target | 15k (8k to 25k) / 30k to 50k | ASSUMPTION |
| Test / cal split minimum | 3,000 / 2,000 decisions | DERIVED |
| ECE T-scaled MVP / target / stretch | 0.05 / 0.03 / 0.015 | requirement from pngwn anchors |
| Order top-1 agreement MVP / target | 0.90 / 0.95 (vs 0.664 prior art) | requirement |
| Trained max seq MVP / target | 4,096 / 8,192 | requirement |
