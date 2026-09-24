# Reliability: nanohunch

Mode: GREENFIELD. Depth: standard (light, adapted). Author: sd-reliability. Date: 2026-09-23.

Labels: `MEASURED` (observed today, or published by the named source and not
reproduced by us), `DERIVED` (arithmetic shown), `ASSUMPTION` (a guess with a
range). Brief files are cited as `RN:<line>` (`_brief/research-notes.md`),
`REQ:<line>` (`_brief/requirements.md`), `MAP:<line>` (`_brief/system-map.md`).
The pngwn report was fetched today from
`huggingface.co/datasets/pngwn/typed-decisions-causal-experiment/raw/main/REPORT.md`
and is cited as `PNGWN §<section>`.

> Note on sources: the dispatch says the pngwn v1 label inversion and the ECE bug
> are in the research notes. They are not (`RN:90-145` has neither). Both were
> verified today directly in `PNGWN §1.1` and `PNGWN §2` / `§9`. The ECE bug was
> in the masked arm's own `calibration_curve()` (temperature applied to `scores`
> while `labels` were indexed from the uncalibrated array); it **raised** rather
> than silently mis-scoring. The silent variants of that bug are what we guard
> against (section 7, C8).

---

## 0. Scope, assumed design, and how the checklist was adapted

**Why:** a reliability review of something that has no users has to decide what
"down" means before it can say anything useful.

Nothing exists yet: the repo has zero commits (`MAP:4-5`), and
`architecture.md` is unwritten. This file assumes the design that the
requirements describe, and flags where the architect's choice changes a finding:

1. **Build**: public sets (MMLU-Pro, HotpotQA, BoolQ class) plus synthetic
   workflow states, on the Mac (`REQ:282`).
2. **Label**: cheap-tier teacher APIs (2 voters) for bulk, a strong tier for an
   eval-reference subset, or an open-weight teacher on a rented GPU (`REQ:300-321`).
   Stored in an append-only, content-addressed cache (`REQ:356-357`).
3. **Split**: by state (`REQ:359`), frozen manifest.
4. **Train**: LoRA on a rented GPU (Modal is the only provider with tooling,
   `MAP:169-174`), causal letter-logit scorer with loss restricted to candidate
   rows (`REQ:56-60`, `RN:106-110`). If the architect picks the scalar
   cross-encoder instead, C4 and C5 below shrink and order sensitivity (C6) is
   solved by construction (`PNGWN §3.1`).
5. **Calibrate**: per-type temperature on the cal split (`REQ:56-57`).
6. **Eval**: MLX engine on the Mac plus a PyTorch reference (`REQ:261`), gate,
   publish to HF Hub.

"Failure" here means one of four things, in order of expected cost: **(a) the
model's probabilities or the eval numbers are silently wrong**, (b) the 200 USD
budget is burned by a runaway path (`RN:17-18`), (c) paid work (labels, GPU
progress) is lost, (d) a reviewer cannot reproduce the report (`REQ:361`). There
is no uptime SLO (`RN:12`, `REQ:33`).

### 0.1 Checklist items dropped or adapted

| failure-modes.md item | Treatment | Reason |
|---|---|---|
| Thundering herd | DROPPED as stated | No shared cache and no client fleet. The only herd is N labeller workers retrying together, covered under retry storm. |
| Retry storm / metastable | KEPT | Labeller vs teacher rate limits is a real metastable loop (section 4.1). |
| Queue backup | DROPPED | Work lists are finite files bounded by dataset size; there is no latency SLO, so a backlog costs wall clock only. |
| Connection pool exhaustion | ADAPTED | One slow strong-tier teacher holding every concurrency slot starves the cheap teacher (bulkheads, section 5). |
| Head-of-line blocking | ADAPTED | One 16k-token sequence padding a training batch, and one long state stalling a labeller slot. |
| Noisy neighbor | ADAPTED | Other Mac apps competing for 24 GiB unified memory during eval and latency benchmarks. |
| Cold start at peak | ADAPTED | No peak. Kept as "every GPU job pays image pull + weight load before step 1". |
| Split brain | ADAPTED | Two launches of the same run writing one checkpoint directory (P5). |
| Lost update, dual write, duplicate delivery, non-idempotent retry | KEPT | Label cache and checkpoint store (section 8). |
| Clock skew | DROPPED except one rule | Nothing orders by wall clock across machines. Rule: order checkpoints by `global_step`, never by mtime. |
| Read-after-write on replica | DROPPED | No replicas. |
| Hot shard / hot key | DROPPED | No partitioned store. |
| Unbounded growth | DROPPED | ~250 MB raw teacher responses total (`REQ:288`), last 3 checkpoints per run (`REQ:292`). |
| Migration without rollback, schema+code together | ADAPTED | Dataset and split versions are immutable; a new label source is a new dataset version, never an in-place edit (section 11). |
| Config change with no canary | ADAPTED | A config drives a paid run: every GPU job runs a CPU preflight on the same config first (section 10). |
| Load test / capacity assumed | HANDED OFF | `capacity.md` owns throughput; this file only uses tokens/s as a health signal. |
| Page vs ticket, 3 a.m. dashboard | ADAPTED | No on-call (`REQ:386`). "Page" = phone push; the "3 a.m." view is the morning-after check of an overnight run. |
| Security-adjacent | HANDED OFF | Secrets in logs and HF token scope go to `security.md` (see `needs_from`). |

---

## 1. Ranked findings (probability x blast radius)

**Why:** you have ~20 h/week (`REQ:383`); spend it on the failures that would
invalidate the portfolio claim, not the interesting ones.

| # | Finding | Probability | Blast radius | Section |
|---|---|---|---|---|
| 1 | Label or option-index inversion, including permutation augmentation that shuffles options but not the target | medium-high (pngwn shipped it once, `PNGWN §1.1`) | every run trained on it, and the eval scores it as correct | C1 |
| 2 | Train/test leakage through near-duplicate states and shared templates/paragraphs | high for synthetic generators | S1 accuracy inflated, generalization claim false | C2 |
| 3 | Teacher consensus correlated errors: the gate cannot see them because test labels share them | medium | every consensus-accuracy number (`REQ:449`) | C9 |
| 4 | Forgotten or hung GPU billing | medium | up to 116 USD, 58% of budget, per forgotten 24 h H100 function (DERIVED, section 10) | B1 |
| 5 | Tokenization boundary or cache-branch divergence between trainer and engine | medium | shipped engine numbers differ from what was trained and gated | C4 |
| 6 | Silent slow-kernel fallback on Qwen3.5 (3 to 10x cost) | high if Qwen3.5 (`MAP:29-34`, `PNGWN §8`) | a full run exceeds its 20 USD cap (`REQ:340-341`) | B3 |
| 7 | Metrics code bug (ECE / NLL / bootstrap) | medium | S2 claims, the gate itself | C8 |
| 8 | Preemption with a broken resume, turning into a crash loop | high occurrence (Modal GPUs always preemptible, `rl-wordle/PLAN.md:111`), low blast if resume works | lost GPU hours, repeated cold starts | section 9 |
| 9 | Teacher model version silently changes or stops returning logprobs mid-labelling | medium | half the dataset labelled by a different distribution source | D1, C9 |
| 10 | Temperature fitted on the wrong split or wrong length mix | medium | S2 per-type and long-bucket ECE | C7 |
| 11 | Base-model pretraining contamination on MMLU-Pro / HotpotQA | high (public since 2024, ASSUMPTION) | absolute gold-slice claims only; S1 delta mostly unaffected | C3 |
| 12 | Label cache lost with the laptop | low | 23 to 35 USD to rebuy, and not exactly rebuyable if a teacher is retired (DERIVED, section 14) | section 14 |

---

## 2. Dependency matrix

**Why:** every external call is a place your money or your weekend can leak;
listing them with their timeouts is how you find the ones that wait forever.

"Critical" means the stage cannot proceed without it. No dependency is on the
inference path except local libraries (`REQ:80-87`).

| Dependency | Stage | Class | Timeout | Retry / backoff | Fallback | What you lose when it is down |
|---|---|---|---|---|---|---|
| Cheap teacher A (bulk) | label | critical for labelling | connect 10 s, total 60 s per call (ASSUMPTION: 1.3k in + 0.8k out tokens, no thinking) | 3 attempts, full jitter, base 2 s, cap 30 s, honour `Retry-After`; global retry budget 10% (section 5) | teacher B labels alone, marked `consensus=partial`, excluded from the consensus reference until A returns | labelling wall clock; nothing downstream if the cache is warm |
| Cheap teacher B (2nd vote) | label | critical for consensus | same as A | same | same, mirrored | same |
| Strong-tier teacher (eval reference subset, `REQ:316-318`) | label | optional (report only) | 240 s per call; reasoning/thinking budget capped explicitly (section 10) | 2 attempts | report consensus from cheap tier only; state it in the eval report | the strong-reference column of the report |
| Open-weight teacher on rented GPU (`RN:160-164`) | label | optional (the ToS-clean path, `REQ:456`) | vLLM server readiness 20 min; per request 120 s | 3 attempts on 5xx only | API teachers for private data; publish model without dataset | releasable dataset (Q3 in `REQ:480`) |
| Modal GPU (training, GPU eval) | train | critical | function `timeout=` set per job, never the 300 s default (`rl-wordle/PLAN.md:110`); max 24 h | `modal.Retries(max_retries=3)` only if `global_step` advanced since last attempt (section 9) | RunPod or other provider, resuming from the HF Hub checkpoint copy | training progress; GPU queue wait is wall clock, not money (ASSUMPTION: queued time not billed) |
| RunPod / other GPU | train | optional fallback | same in-process caps; no provider-side function timeout exists, so the self-stop watchdog is mandatory (B1) | none automatic | back to Modal | nothing unless Modal is down; note `runpodctl` is not installed (`MAP:174`) |
| HF Hub download (base weights, datasets) | all | critical at first load only | 30 min for 9.3 GB (DERIVED, `MAP:59-60`) in a CPU prefetch function, not in the GPU container | `huggingface_hub` built-in retries; 3 attempts | weights cached on a Modal Volume and in `~/.cache/huggingface`; then `HF_HUB_OFFLINE=1` | first run on a new machine only |
| HF Hub upload (checkpoint copy, label backup, release) | train, backup | **"optional" with no timeout is critical**: a synchronous upload that hangs holds a billing GPU idle | 300 s per checkpoint upload, in a background thread | 2 attempts, then alert and continue; the Volume copy is primary | Volume; upload at end of run | portability to a second provider; off-laptop label backup |
| HF hub kernels (`fla`, `causal_conv1d`) via `use_kernel_func_from_hub_with_fallback` (`MAP:29-34`) | train (Qwen3.5 only) | **nominally optional, actually cost-critical**: the fallback is silent and 3 to 10x slower (`REQ:333-335`, `PNGWN §8`) | kernel fetch happens at model init; set `HF_HUB_OFFLINE=1` after prefetch so it cannot hang | none | pure-torch path, but only if the tokens/s floor still passes (B3) | 3 to 10x GPU cost per run |
| PyPI / uv index + base image registry | image build | critical at build only | build step 20 min | Modal image cache; rebuild on failure once | image built from `uv.lock` export; last good image digest reused | cannot start new runs until build succeeds; `causal_conv1d` is sdist-only and was not source-built by pngwn (`PNGWN §8`) |
| Experiment log backend (trackio / W&B) | train | **"optional" with no timeout is critical**: a blocking `init()` or flush stalls the step loop | 10 s, non-blocking queue, drop on failure | none | local JSONL on the Volume is canonical | pretty charts only |
| Push notifications (e.g. ntfy, ASSUMPTION) | all | optional | 5 s | none | morning-after check (section 12.4) | overnight alerting |
| Jev API (baseline B2, `REQ:465`) | eval | optional | 30 s per request; run as a separate command, never inline in the gate | 2 attempts | report B2 as absent | one comparison column |
| GitHub remote | backup | critical for DR | n/a | n/a | local clone | code backup |
| Local Mac (orchestrator, labeller host, eval, latency bench) | label, eval | critical | see M1 to M3 below | n/a | GPU eval via PyTorch reference (parity-checked, `REQ:261`) | eval wall clock; latency numbers |

### 2.1 Failure submodes per dependency (each with mechanism)

**D1. Teacher model silently changes.** Mechanism: the config names a floating
alias; the provider repoints it mid-labelling; the cache key omits the resolved
version; labels from two models are averaged as if one. Trigger: any labelling
campaign spanning more than a few days. Blast radius: the consensus distribution
shifts for part of the dataset, invisibly. Detection: record the response's
`model` field (and any fingerprint) per call; assert it is constant per session;
re-ask 50 fixed canary prompts at the start of every session and stop if argmax
agreement with the stored answers is below 95% (ASSUMPTION threshold; temperature
0 is not fully deterministic). Cost of the canary: 50 x 2 teachers x ~0.001 USD =
0.10 USD per session (DERIVED from `REQ:310`).

**D2. Logprobs unavailable.** Mechanism: the chosen teacher does not return
logprobs (often the case when reasoning is enabled, ASSUMPTION), so the code
falls back to verbalized probabilities or k samples; different sources have
different calibration; mixing them changes the soft-label target without a
trace. Detection: every label row carries `dist_source in {logprobs, verbalized,
samples_k, onehot}`; the dataset build fails if one question type mixes sources
unless the config opts in; the eval reports metrics per `dist_source`.

**D3. Rate limits (429).** Mechanism and fix in section 4.1. Throughput need is
low: 30k decisions x 2 teachers / ~4 decisions per call = 15k calls; at 4 in
flight x 10 s each that is 15k / 0.4 per s = 10.4 h (DERIVED; 10 s latency is an
ASSUMPTION, range 3 to 30 s). So concurrency above 4 to 8 buys nothing and only
invites 429s.

**G1. GPU capacity unavailable.** Mechanism: the requested type (H100/A100) is
not scheduled for hours. Fallback: a GPU type list in the function config
(ASSUMPTION: Modal accepts a list; verify), otherwise another provider resuming
from the Hub checkpoint. Detection: "queued > 30 min" alert from the launcher.

**G2. Image build failure or drift.** Mechanism: unpinned installs resolve new
versions between runs; the prior Modal app in `mini-vllm/app.py:19-31` does
exactly this (`"torch"`, `"transformers"` with no versions). A transformers bump
changes tokenization or model code between the training image and the Mac env,
which C4 then catches only if it runs. Fix: build the image from `uv export
--frozen` of the committed `uv.lock`; record the image digest in every run
manifest; assert `transformers.__version__` equal on Mac and image at preflight.

**H1. HF auth or API change on upload.** Mechanism (MEASURED elsewhere): pngwn
lost two artifacts "to dead containers" because `upload_file` became
keyword-only in hub 1.31.0 (`PNGWN §9`). The exception fired after the work was
done, inside a container that then exited. Fix: at job start, before loading the
model, run `whoami`, create the repo, upload a 1 KB canary, download it, compare
sha256; fail the job if any step fails. Cost: seconds of CPU.

**M1. Mac memory pressure.** Mechanism: 4B bf16 at W2 needs 9.5 to 11 GB
(DERIVED, `REQ:206`) against a Metal working set of 16 to 18 GB (ASSUMPTION A15,
`REQ:463`); a browser plus IDE pushes the system into swap and eval slows 5 to
10x, or Metal aborts the process at item 2,000 of 2,500. Fix: eval writes one
JSONL row per item keyed by uid and skips done uids on restart; log
`mx.get_peak_memory()` and `vm_stat` pageouts at start and end.

**M2. Thermal throttling.** Mechanism: a 55 min eval (`REQ:213`) heats the
chassis; latency (S7) measured afterwards reports the throttled clock as the
model's latency. Fix: latency benchmark is a separate command, run after a
cooldown, recording thermal state (`pmset -g therm`); results with a throttling
flag are rejected. Accuracy metrics are unaffected by throttling.

**M3. Sleep.** Mechanism: lid close or idle sleep suspends the labeller (calls
time out, resumes idempotently: fine) and kills any non-detached `modal run`
(`rl-wordle/PLAN.md:113`: "`modal run --detach` or the job dies when you close
your laptop"). Fix: GPU jobs always launched with `--detach`; long local jobs
wrapped in `caffeinate -i`.

---

## 3. Single points of failure

**Why:** in a solo project every SPOF is accepted by default unless you write
down how you would find out it broke.

| # | SPOF | Disposition | Tripwire | Recovery |
|---|---|---|---|---|
| P1 | Label cache on one laptop disk (paid, possibly unrepeatable) | **removed**: pushed to a private HF dataset repo after every session and every 1,000 new labels | labeller refuses to start a new session if `backup_lag > 1,000 labels or > 24 h` | restore drill, section 14.3 |
| P2 | Split manifest (seeded split + group keys) | **removed**: committed to git, sha256 recorded in every checkpoint and eval report | preflight hash mismatch fails the job | `git checkout` the manifest; never regenerate with a new seed |
| P3 | Single GPU provider (Modal) | accepted | "queued > 30 min" or 2 failed image builds in a row | resume on another provider from the Hub checkpoint copy (section 9) |
| P4 | Single HF account and token | accepted | preflight `whoami` fails | re-auth; the Volume holds checkpoints meanwhile |
| P5 | One checkpoint directory per run name (two launches = two writers) | **removed**: a lock file `run.lock` with attempt id on the Volume; a second launch with a live lock exits | lock owner heartbeat older than 15 min is the only way to steal the lock | resume validates `config_hash` in the checkpoint manifest |
| P6 | The Mac as the only eval machine | accepted | parity test (MLX vs PyTorch) failing blocks the release | run the PyTorch reference eval on a GPU; report which runtime produced numbers |
| P7 | A single training seed per configuration | accepted, with a measurement | seed std from 3 pilot seeds (section 11) is larger than the gate's minimum effect | raise the gate threshold or add seeds; pngwn also ran one seed (`PNGWN §7`, item 8) |
| P8 | The author as the sole human auditor (S6, `REQ:417`) | accepted | author vs consensus agreement < 80% | treat consensus as untrusted; see runbook 3 |
| P9 | A teacher model version that may be retired | accepted | provider deprecation notice; canary drift (D1) | the cache is the only copy of those labels (P1) |
| P10 | The "best" pointer on the published model repo | **removed**: only the gate script may move the `best` tag; every release is an immutable tag | gate report missing for a tagged revision | move `best` back to the previous tag |

---

## 4. Cascading failure analysis

**Why:** in a pipeline the damage from a failure shows up two stages later, in
a number that looks plausible.

### 4.1 Teacher 429 -> retry storm -> biased dataset

Mechanism: 8 workers hit a 429 together, retry with fixed delays, arrive in
synchronized waves, and keep the limiter tripped after the quota refills
(metastable). If the limiter counts rejected requests against quota (ASSUMPTION,
varies by provider) the loop persists indefinitely. Second-order effect, the
one that matters: long states time out and hit the retry cap more often, so they
are dropped more often, so the labelled set is shorter than the requested set,
so the model trains short and regresses on the >2k bucket, which is exactly
pngwn's failure (`RN:140-142`). Trigger: any campaign with concurrency above
the provider quota. Blast: labelling stalls (cost: wall clock) and the dataset is
silently length-biased (cost: S4). Detection: retry fraction > 10% trips the
breaker (section 5); dataset build compares the length distribution of labelled
vs requested states per type and fails if labelled p90 < 0.9 x requested p90.

### 4.2 Label bug -> training -> gate passes

Mechanism: an index inversion or unmapped permutation (C1) is applied at dataset
build, which feeds both train and test. The model learns the inverted mapping,
and the gate, which compares against the same inverted test labels, reports high
accuracy. Blast: every run and every published number. Detection: only checks
that do **not** share the label path catch it: gold-slice accuracy for B0
(below chance is the tell), oracle/Bayes consistency on synthetic generators
(pngwn's tell was escalate Bayes accuracy 0.082 = 1 - 0.918, `PNGWN §1.1`), and
the text round-trip assertion. All three run at dataset build (C1).

### 4.3 Kernel fallback -> wall-clock cap -> undertrained model -> rerun

Mechanism: the hub kernel fails to load, transformers silently uses pure torch
(`MAP:29-34`), tokens/s drops 3 to 10x, the run hits `max_wall_seconds` at 30% of
planned steps, the checkpoint fails the gate, and the natural reaction is to
rerun, paying twice. Detection: tokens/s floor at step 50 (B3).

### 4.4 Broken resume -> crash loop

Mechanism: preemption at step 3,000; resume loads a half-written checkpoint and
crashes deterministically; `max_retries=3` restarts it 3 times, each paying image
pull + weight load (5 to 10 min, ASSUMPTION). Detection: the retry wrapper
requires `global_step` to have advanced since the previous attempt, else it exits
and pushes an alert. Atomic checkpoint writes (section 9) remove the cause.

### 4.5 Sync upload -> idle GPU

Mechanism: checkpoint upload to the Hub is synchronous in the step loop; the Hub
throttles or hangs; the GPU sits at 0% utilization, billing. Detection: step
watchdog (no step in 10 min -> alert and exit) and upload in a background thread
with a 300 s timeout.

### 4.6 Bulkhead: strong teacher starves cheap teacher

Mechanism: strong-tier calls with thinking take 60 to 240 s (ASSUMPTION) and
share one semaphore with the cheap tier; all slots fill with slow calls; cheap
labelling throughput drops to near zero. Fix: one semaphore per teacher.

### 4.7 Head-of-line in training batches

Mechanism: a 16k-token example lands in a batch of 1k-token examples; padding
multiplies the batch cost ~16x, and if that batch arrives at step 4,000 it OOMs
there, not at step 0. Fix: length-bucketed batching with a max-tokens-per-batch
budget; run the single longest batch first as a memory probe in preflight.

### 4.8 Noisy neighbor on the Mac

Mechanism: other apps take unified memory during eval (M1) or CPU during the
latency bench (M2). Fix and detection as in 2.1.

---

## 5. Timeout, retry and backoff budget

**Why:** per-call retry limits alone turn a provider blip into an overnight 429
loop; a budget shared across all calls cannot.

All values are ASSUMPTION starting points to be replaced by the p99 latency
observed in the first 200 calls (log it, section 13).

### 5.1 Labeller (inner to outer)

| Layer | Value | Check inner < outer |
|---|---|---|
| Connect timeout | 10 s | |
| Per-call total, cheap tier | 60 s | |
| Backoff between attempts | full jitter: `sleep = uniform(0, min(30, 2 * 2^attempt))`; attempt 1 <= 4 s, attempt 2 <= 8 s | |
| Attempts, cheap tier | 3 | worst case 3 x 60 + 4 + 8 = 192 s (DERIVED) |
| Per-item deadline, cheap tier | 240 s | 192 < 240 OK |
| Per-call total, strong tier | 240 s | |
| Attempts, strong tier | 2 | worst case 2 x 240 + 4 = 484 s (DERIVED) |
| Per-item deadline, strong tier | 600 s | 484 < 600 OK |
| Session cap | spend cap (section 10) or breaker opens 3 times in 1 h | |

Deadline propagation: each attempt's timeout is `min(per_call, deadline -
now)`; no attempt starts with less than 10 s remaining.

**Retry budget (fraction of traffic, not per call):** a token bucket shared by
all workers for one teacher. Each first-attempt success deposits 0.1 token; each
retry spends 1 token; bucket cap 50. Effect: retries are capped at ~10% of
successful traffic over any window; when the bucket is empty, failures are
recorded as `status=retry_budget_exhausted` and not retried. If more than 10% of
the last 500 calls failed, the breaker opens for 10 min, then probes with 1 call.

**On 429:** honour `Retry-After`, halve that teacher's concurrency (AIMD), add 1
back per 50 consecutive successes, floor 1, ceiling 8.

**Not retried:** 400, 401, 403, content-filter refusals, and responses that parse
but name an option outside the list. Each is written to the ledger with its
status. Malformed output gets at most 1 re-ask, and re-asks count against the
retry budget (they cost money, `REQ:310`).

### 5.2 Training job (inner to outer)

| Layer | Value | Check |
|---|---|---|
| Log backend call | 10 s, non-blocking, drop on failure | |
| Checkpoint write to Volume + fsync + `commit()` | measured; expect 3 to 11 s for 140 to 560 MB at 50 MB/s (DERIVED, ASSUMPTION bandwidth, section 9) | |
| Hub upload of checkpoint copy (background) | 300 s, 2 attempts | 600 s < 15 min checkpoint interval OK |
| Step watchdog | no completed step in 10 min -> dump stacks, alert, exit nonzero | 10 min > p99 step time (expect 1 to 10 s, ASSUMPTION) by > 60x |
| In-process `max_wall_seconds` | function timeout minus 10 min: checkpoint, upload, exit 0 | |
| Modal function `timeout=` | per job: `min(8 h, run_cap_usd / hourly_usd)` (`REQ:340-341`) | outermost |
| Modal retries | 3, gated on progress (4.4) | |

### 5.3 Eval (Mac)

Per-item soft limit 120 s (W2 is ~9.2 s DERIVED, `REQ:203`); an item over the
limit is logged and the run continues. Whole eval is resumable per uid, so there
is no outer timeout to tune.

---

## 6. Degradation modes

**Why:** deciding now what the pipeline does at 2x is cheaper than discovering
that the 20 USD cap truncated your only full run.

| Condition | What happens | Designed response |
|---|---|---|
| 2x data (30k -> 60k decisions) | training tokens and GPU cost 2x: 1.2 to 20 USD -> 2.4 to 40 USD per 4B run (DERIVED from `REQ:335`) | runs are token-budgeted (`max_train_tokens`), not epoch-budgeted; cap hit means stop and checkpoint, not overrun |
| 2x mean state length (1k -> 2k tokens) | same cost doubling; activation memory ~2x per sequence (`REQ:457`) | max-tokens-per-batch, longest-batch memory probe (4.7) |
| Cold label cache (new machine) | nothing is re-paid if the backup restores | restore from HF dataset repo; dry-run shows 100% hit rate before any live call |
| Cold weight cache | 9.3 GB download (`MAP:59-60`) at 20 to 100 MB/s = 1.5 to 8 min (DERIVED, ASSUMPTION bandwidth) | prefetch in a CPU function to the Volume; GPU never waits on the Hub |
| Hub read-only or down | training continues from Volume; uploads queue | publish deferred; no run blocked |
| One teacher down | single-teacher labels marked partial | excluded from the consensus reference; resumed later idempotently |
| Logprobs removed by a teacher | distribution source changes | new dataset version with a different `dist_source`; never mixed silently (D2) |
| GPU type unavailable | queue | GPU list, then other provider (G1) |
| Mac memory short | swap or abort | resumable eval; 4-bit export for eval only if parity with bf16 on 200 items is within 2e-2 |

---

## 7. Correctness of the model and eval (the real risk)

**Why:** every one of these produces a plausible number, not a crash; the only
defence is an assertion that runs before you look at the number.

When checks run:

- **T0 unit** (`pytest`, every commit, < 60 s CPU, tiny random-init configs of
  the real architectures: e.g. a 2-layer Qwen3.5 config with 1 DeltaNet + 1
  attention layer, `full_attention_interval` from `MAP:26-27`).
- **T1 dataset build** (on every build; writes the manifest; fails the build).
- **T2 job preflight** (start of every GPU job, CPU only, < 2 min, before
  weights load).
- **T3 in training** (per step, cheap).
- **T4 gate** (every candidate checkpoint).
- **T5 release** (test split, audit reconciliation, restore drill).

### C1. Label / option index inversion (rank 1)

- **Mechanism.** pngwn v1: `review_label = int(rng.random() < p_review)` against
  `REVIEW_OPTS = ["yes","no"]`, and `escalate_label = 1 if gold_sev >= 3 else 0`
  against `ESC_OPTS = ["yes","no"]` (`PNGWN §1.1`): the label index meant the
  opposite of the option at that index. Our pipeline has four more places to do
  the same: (a) permutation augmentation shuffles `options` but not the target
  distribution; (b) a teacher's answer letter is parsed against canonical order
  while the teacher saw a shuffled order; (c) Noul P(yes) mapped to index 1 in one
  module and 0 in another; (d) Score levels written high-to-low in one generator.
- **Trigger.** Any augmentation or teacher-order randomization, i.e. certain to
  be exercised (A12, `REQ:460`).
- **Blast radius.** All training runs on the dataset; test labels share it, so
  the gate passes (4.2).
- **Detection / checks.**
  1. T1: labels are stored as **option text or option id**, never as a bare
     index; the index is computed at the last step. Assert
     `options[label_idx].text == label_text` for every row.
  2. T0 property test: for random permutations `perm`,
     `target_perm[j] == target[perm[j]]` and `options_perm[j] == options[perm[j]]`,
     and the teacher parse of letter L under order `perm` maps back to the same
     option text.
  3. T1: for generators with a known posterior, Bayes accuracy against the drawn
     label must exceed `1/n_options + 3 SE` per question type (pngwn's tell was
     0.082, `PNGWN §1.1`).
  4. T4: any per-type slice where B0 or the candidate scores below
     `chance - 3 SE` fails the gate with "possible label inversion".
  5. T1: print 20 random rows fully decoded to text (state, question, options in
     presented order, target as text) into the build log; read them. This costs
     5 minutes and is the check that finds what the others miss.

### C2. Train/test leakage (rank 2)

- **Mechanism.** (a) Split by decision instead of state: a state's other
  decisions sit in train. (b) Split by state but not by source: HotpotQA
  questions share Wikipedia paragraphs; synthetic states from one template with
  different seeds differ by a few fields; MMLU-Pro may contain near-duplicate
  questions (ASSUMPTION). The model memorizes the paragraph or template, and the
  test measures memory. (c) Leaked lookup questions: pngwn's `team` field was a
  deterministic 1:1 function of a `Product` field in the state, scoring 1.0000 on
  all arms and inflating ~9% of rows (`PNGWN §1.1`). (d) The human audit set
  drawn from train. (e) Checkpoint selection on the test split.
- **Trigger.** Synthetic generators make (b) near certain.
- **Blast radius.** S1 inflated; generalization claim unsupported.
- **Checks.**
  1. T1: split key is a **group key** (source document id for HotpotQA,
     question id + source for MMLU-Pro, template id + entity seed family for
     synthetic), not the state id alone. Assert uid sets and group-key sets are
     disjoint across train / dev / cal / test.
  2. T1: MinHash near-duplicate scan across splits (5-gram word shingles,
     Jaccard >= 0.8, ASSUMPTION thresholds); fail if > 0.5% of test states have
     a train near-duplicate.
  3. T1: at least one whole template per workflow is held out as an OOD test
     slice, reported separately.
  4. T1: shallow-baseline scan per question template: a bag-of-words logistic
     regression on the state; any template it answers at >= 0.98 accuracy, or
     whose majority class is >= 0.95, is flagged as leaked or degenerate and
     reported with and without it.
  5. T2 and T4: the job refuses to run if the manifest sha256 does not match the
     one recorded at build. The test split is read only by the release gate
     (T5); checkpoint selection uses a separate dev split. Each test read is
     appended to `test_access.jsonl` (section 12).

### C3. Pretraining contamination

- **Mechanism.** MMLU-Pro and HotpotQA predate Qwen3.5 (2026-02, `RN:74`), so
  the base likely saw them (ASSUMPTION); gold-slice accuracy partly measures
  recall, for B0 and the fine-tune alike.
- **Blast radius.** Absolute gold-slice numbers and comparisons to pngwn
  (`REQ:249-250`). The S1 delta over B0 is mostly protected because B0 shares
  the contamination.
- **Checks.** T4: choices-only probe (options without the question); accuracy
  far above `1/n` flags artifacts or memorization. Report headline S1 on fresh
  synthetic states generated after the base model's release as the clean number.
  Label the gold slices "possibly contaminated" in the model card.

### C4. Tokenization boundary and cache-branch divergence (rank 5)

- **Mechanism.** The trainer tokenizes `state + question` as one string; the
  engine tokenizes the state once and each question separately and concatenates
  ids. BPE merges across the join differ (a state ending in `.` followed by a
  question starting with `\n` may merge into one token in the full string but
  not in the split one), or a BOS is added twice. The engine then feeds a token
  sequence the model never trained on; probabilities drift a little and nothing
  crashes. Separately, a branch that copies full-attention KV but not the Gated
  DeltaNet conv and recurrent state (`RN:134-136`) gives drifted numbers.
- **Trigger.** Every engine change; every tokenizer or transformers version bump
  (G2).
- **Blast radius.** The shipped engine's metrics diverge from the gated metrics.
- **Checks.**
  1. T0: one function `build_ids(state, questions) -> ids, answer_positions`
     used by the trainer, the PyTorch reference, and the MLX engine. Segments
     are tokenized separately with a fixed separator, by construction, in
     training too. Property test on 1,000 random plus adversarial boundaries
     (trailing whitespace, punctuation, partial UTF-8, CJK, emoji): trainer ids
     == engine ids byte-exact.
  2. T0: branched readout vs full re-encode, fp32, max abs logit diff <= 1e-4.
     pngwn got 1.5e-05 and notes a mask/cache logic error shows ~13 logit units
     (`PNGWN §4.3`), so the gap between bug and noise is 5 orders of magnitude.
     Include state lengths 63, 64, 65, 1,000 to cross conv-kernel and chunk
     boundaries (chunk size ASSUMPTION 64).
  3. T0: isolation: question i alone vs in a batch of 16 with variable-length
     neighbours and padding, max abs prob diff <= 1e-3 bf16 (`REQ:260`).
  4. T4: every eval record stores `sha256(ids)`; the release gate compares 200
     records against the trainer's ids for the same uids.
  5. T2: assert `transformers` and `tokenizers` versions on the image equal the
     Mac lockfile.

### C5. Option letters or score levels not single tokens

- **Mechanism.** The readout gathers one logit row per candidate. In context,
  the emitted token is `" A"` (with space) not `"A"`, or the tokenizer splits a
  candidate into two tokens; the gather reads a sub-token shared by several
  candidates, so they get identical logits. Score levels `0..10`: if the
  tokenizer splits digits individually (recalled for Qwen tokenizers,
  ASSUMPTION), `"10"` is two tokens and level 10 is unreadable. pngwn verified
  26 letters are single tokens for the Qwen3 tokenizer (`PNGWN §4.2`); Qwen3.5's
  vocab is different (248,320, `RN:74`) and MiniCPM5 is untested.
- **Checks.** T0 and T2: for every candidate label actually used, assert
  `encode(prefix + L) == encode(prefix) + [id_L]` (exactly one appended token,
  prefix unchanged) and all candidate ids distinct. Score levels above 9 use
  letters, or the assertion fails the config.

### C6. Order sensitivity hidden by a fixed gold position

- **Mechanism.** Generators and some public sets emit the correct option first
  (or at a fixed index); the model learns a position prior; test has the same
  bias, so accuracy is high and nobody runs S3. pngwn's arm B flipped 37.5% of
  gold rankings under reversal (`PNGWN §3.1`).
- **Checks.** T1: gold-position histogram per type after shuffling; chi-square
  against uniform, fail at p < 0.01; position-only baseline accuracy must be
  `<= 1/n + 2 pts`. T1 teacher side: each teacher labels 200 items in two
  orders; teacher flip rate is reported, and if above 10% (ASSUMPTION), bulk
  labels average both orders. T4: S3 suite (reverse + 3 permutations,
  `REQ:234-237`) is part of the gate.

### C7. Calibration fitted on the wrong data

- **Mechanism.** (a) T fitted on test, or on data overlapping test. (b) Cal split
  dominated by short public QA, test by long workflow states; the model is more
  overconfident on long inputs, so overall ECE looks fine while the >2k bucket is
  badly calibrated (Simpson's paradox). (c) T per type fitted but applied with
  the wrong type key.
- **Checks.** T0: `fit_temperature(split)` refuses any split whose hash is not
  `manifest.cal_hash`; the stored T carries that hash; T4 asserts it. T4: argmax
  is bit-identical raw vs T-scaled (pngwn's invariance check, `PNGWN §2`). T1:
  cal split stratified to match test over (type x length bucket x source) with
  >= 100 decisions per stratum (ASSUMPTION) or strata merged. T4: ECE reported
  per type and per length bucket with bootstrap CIs; if per-bucket fitted T
  differs from the global T by > 20%, flag for per-bucket temperature.

### C8. Metrics code bugs (rank 7)

- **Mechanism.** Known real instance: `calibration_curve()` applied temperature
  to `scores` while indexing `labels` from the uncalibrated array
  (`PNGWN §2`, `§9`). Silent variants: confidence exactly 1.0 falling outside
  the last bin; unweighted mean over bins; ECE on T-scaled probs but accuracy on
  raw; `log(0)` in NLL clipped differently between runs; bootstrap resampling
  items unpaired.
- **Checks (T0, every commit).** pngwn's harness self-tests 7/7 including
  temperature recovery to 3e-4 (`PNGWN §2`); ours:
  1. Perfectly calibrated sampler: `p ~ Dirichlet`, `y ~ Cat(p)`, n = 100k:
     ECE < 0.01.
  2. Temperature recovery: logits scaled by T0 = 2.0, fitted T within 0.01.
  3. Constant predictor: ECE equals `|conf - acc|`.
  4. Confidence 1.0 lands in the last bin; all-correct one-hot gives Brier 0,
     NLL 0.
  5. Cross-check against an independent implementation on random inputs to
     1e-6 (library choice ASSUMPTION).
  6. `len(probs) == len(labels)` asserted at every metric entry point.
  7. Paired bootstrap: C vs itself gives CI containing 0 and width < 1e-9.

### C9. Teacher consensus systematically wrong (rank 3)

- **Mechanism.** Two teachers share a bias (same pretraining, same misreading of
  a template, same position prior); averaging does not cancel correlated
  errors; the student learns them and the consensus test scores them as correct.
- **Blast radius.** Every consensus accuracy number (A1, `REQ:449`).
- **Checks.** T1: teacher-teacher agreement per type and template (flag < 80%,
  `REQ:449`); items with Jensen-Shannon divergence > 0.3 between teachers are
  flagged and reported, not silently averaged. T1: both teachers also label the
  gold slices, giving a measured teacher error rate. T5: author audit of 200 to
  300 stratified decisions before bulk training (S6, `REQ:417`). T4 guardrail:
  gold-slice accuracy must not drop more than 2 pts while consensus accuracy
  rises, which is the signature of learning teacher errors.

### C10. Loss and readout implementation

- **Mechanism.** Loss read at the wrong position (the letter token's own
  position instead of the one before it); candidate ids taken from a different
  tokenizer; question and answer position cut off by right-side truncation of a
  long concatenated string, so the loss reads a random state position.
- **Checks.** T0: restricted-row loss equals full-vocab CE restricted and
  renormalized to 1e-6 on a tiny model (pngwn measured 7.2e-07, `PNGWN §8`). T0:
  overfit test: 32 examples reach >= 0.99 train accuracy within 200 steps. T1:
  truncation removes state tokens only (middle or left); assert every example
  keeps its question segment and answer position; log the truncated count per
  length bucket. T0: soft targets sum to 1 +/- 1e-6, no NaN, length equals
  option count, epsilon floor before log.

---

## 8. Pipeline correctness: dual writes, duplicates, ordering

**Why:** the label cache and checkpoint store are small databases, and they fail
the way databases do.

| Risk | Mechanism | Fix | Detection |
|---|---|---|---|
| Non-idempotent paid call | the request succeeds, the process dies before the row is written, the retry pays again | cache key `sha256(teacher, resolved_model, prompt, params)` computed **before** the call; row written with `status=in_flight` first, completed after | ledger reconciliation vs provider usage dashboard weekly (S11, `REQ:422`) |
| Cache miss by construction | nondeterministic prompt content (timestamp, unseeded shuffle, dict order) changes the key every run, so every rerun re-pays the full dataset | prompts rendered from sorted, seeded inputs; T0 test renders the same item twice and compares hashes | T2 dry run on the label job: expected hit rate 100% for existing items; fail below 99% |
| Duplicate rows | retry after a timeout where the server actually completed | SQLite with `PRIMARY KEY(cache_key)`, WAL, single writer | dataset build asserts uid uniqueness |
| Interleaved writes | two labeller processes append to one JSONL | one writer process (SQLite) | parse-all on load; fail on any bad row |
| Dual write: Volume + Hub checkpoint | one copy newer than the other; resume picks the stale one or a partial one | manifest (`step`, `sha256` per file, `config_hash`, `data_cursor`) written last; resume takes the highest step with a valid manifest across both stores | resume log prints chosen source and step |
| Ordering | checkpoints sorted by mtime after a copy between stores | sort by `global_step` only | T0 test |
| Split brain | two launches of one run name | `run.lock` (P5) | second launch exits with a message |

---

## 9. Checkpointing and resumability

**Why:** Modal GPU functions are always preemptible with no opt-out
(`rl-wordle/PLAN.md:111`), so resume is the normal path, not an edge case.

- **Interval.** Every 15 min of wall clock at the next step boundary, plus at
  `max_wall_seconds` and on SIGTERM. Meets the <= 20 min requirement
  (`REQ:354`).
- **Contents.** LoRA weights, optimizer state, LR scheduler, RNG states (python,
  numpy, torch, cuda), `global_step`, `data_cursor` (epoch, index into a seeded
  per-epoch permutation of uids, not DataLoader internals), config hash, git sha,
  image digest, split manifest hash.
- **Size.** Adapter 20 to 80 MB (ASSUMPTION, `REQ:289`) = 10 to 40M params in
  bf16; fp32 master + Adam m and v = 12 B/param = 120 to 480 MB; total 140 to 560
  MB (DERIVED). At 50 MB/s (ASSUMPTION, range 20 to 200) that is 3 to 11 s per
  write, 0.3 to 1.2% of a 15 min interval (DERIVED).
- **Where.** Primary: Modal Volume, written to `ckpt-{step}.tmp/`, fsynced,
  renamed, then `volume.commit()` called explicitly and the manifest written
  last (ASSUMPTION: background commit is not guaranteed on preemption; verify in
  the Phase 0 kill test). Secondary: private HF model repo, every checkpoint in
  a background thread (5.2), so another provider can resume. Keep the last 3
  (`REQ:292`).
- **Resume.** One command: `train resume --run-id X` finds the highest valid
  manifest, verifies sha256 of each file and the split/config hashes, restores
  all state, logs the first 5 batch uids. Target <= 15 min from a fresh instance
  (`REQ:355`).
- **Max work lost per preemption.** 15 min interval + 5 to 10 min restart
  (image pull, weight load; ASSUMPTION) = up to 25 min of GPU time. At H100 3.95
  + 0.89 CPU/memory = 4.84 USD/h (`rl-wordle/PLAN.md:97,99-100`, ASSUMPTION
  today): 2.02 USD. At A100 80 GB 2.50 + 0.89 = 3.39 USD/h (`rl-wordle/PLAN.md:96`):
  1.41 USD (DERIVED).
- **Verification (T0 locally, then once on the real GPU in Phase 0).** Train 100
  steps straight; separately train 50, kill -9, resume, train 50. Assert batch
  uid sequences identical exactly and loss curves within 1e-3 (bf16
  nondeterminism). This is the same exit criterion the author already wrote: "A checkpoint
  survives a killed-and-restarted run" (`rl-wordle/TASKS.md:210-211`).

---

## 10. Budget failure and hard stops

**Why:** the budget is the binding constraint (`RN:17-18`); one forgotten
instance can end the project, so every paid path needs a stop that does not
depend on you remembering.

Budget lines: labels <= 60 USD, GPU <= 90 USD, reserve >= 50 USD (`REQ:300-302`).

| # | Runaway path | Mechanism | Worst case | Hard stop |
|---|---|---|---|---|
| B1 | Forgotten or hung GPU | a Modal function with `timeout=86400`, `modal serve`, a notebook, `min_containers > 0`, or a RunPod pod left on | 24 h x 4.84 = 116 USD, 58% of the total budget (DERIVED); a RunPod pod has no ceiling | Modal `timeout = min(8 h, run_cap / hourly)`; in-process `max_wall_seconds`; step watchdog (10 min); RunPod start script arms `sleep $MAX; stop pod` and an idle watchdog (GPU util < 5% for 20 min -> stop) (ASSUMPTION: provider API supports it); provider-side budget limit if offered (ASSUMPTION, verify); daily spend check into `spend.csv` |
| B2 | Crash loop | deterministic crash on resume with retries | 3 x (5 to 10 min) cold starts per incident | progress-gated retries (4.4) |
| B3 | Silent slow kernel | pure-torch DeltaNet/conv fallback (`MAP:29-34`); pngwn found `fla` bought ~2% because the conv was the bottleneck (`PNGWN §8`) | a run planned at 1 to 8 A100-h becomes 3 to 25 (`REQ:451`) | log the chosen kernel implementation at init; at step 50 assert tokens/s >= 50% of the Phase 0 baseline for that model and seq length, else checkpoint and exit |
| B4 | Thinking-token blowup | strong-tier "high thinking" emits 10 to 40x the assumed 500 output tokens (ASSUMPTION) | 8 to 20 USD eval-reference line (`REQ:318`) becomes 80+ | explicit reasoning/output token cap in every request; after the first 50 calls, abort if mean cost per decision > 2x the estimate |
| B5 | Strong tier used for bulk by config error | a model id typo or default | 60 USD buys only 6k to 15k decisions (`REQ:320-321`) | per-teacher allowlist of tiers per job type; pre-flight estimate printed and must be confirmed |
| B6 | Re-paying the dataset | cache miss by construction (section 8) | 15 USD per full relabel (DERIVED, `REQ:315`) | T2 dry-run hit-rate check |
| B7 | Open-weight teacher idling during a 50 to 70 GB download (ASSUMPTION size for 27B to 35B bf16) | GPU container downloads weights itself | 10 to 30 min of GPU per launch (ASSUMPTION) | CPU function prefetches weights to a Volume |
| B8 | Sweep creep | "one more run" | the reserve | global kill rule: 60% spent (120 USD) without a model passing S2 MVP -> stop scaling (`REQ:436-438`) |

Pre-flight for every paid job: estimate = tokens x price (labels) or planned
steps x measured s/step x hourly (GPU); the job refuses to start if `estimate >
min(job_cap, budget_remaining - reserve)`. Labeller session cap default 10 USD,
cumulative teacher cap 60 USD, enforced from the ledger in code, not from the
provider dashboard.

---

## 11. Reproducibility

**Why:** "anyone can rerun this" (`REQ:361`) is the portfolio claim that costs
the least to make true now and the most to retrofit.

- **Seeds** (all in config, all logged): split seed, training seed, per-epoch
  data permutation seed, augmentation seed, eval permutation seed, bootstrap
  seed. pngwn's arm A script set no global seed (`PNGWN §8`); do not inherit that.
- **Pinned.** `uv.lock` committed (`MAP:135-137`); the Modal image built from it
  (G2); HF base model and dataset revisions by commit sha; teacher model ids with
  the resolved version returned per call; image digest per run.
- **Hashes.** sha256 of each split's canonical JSONL and of the uid list, in the
  manifest, in every checkpoint and eval report.
- **Configs.** `configs/*.toml` committed; `config_hash` is part of the run id;
  a run with a dirty git tree refuses to launch a paid job.
- **Tolerances.**
  - Eval rerun, same checkpoint, same Mac: headline metrics within +/- 0.002
    accuracy and ECE (`REQ:360`); same input same device identical to 1e-6
    (`REQ:262`).
  - MLX vs PyTorch, same checkpoint: max abs prob diff <= 2e-2 bf16 (`REQ:261`).
  - Training rerun from cached labels, same seed: S1 within 1 pt (`REQ:423`).
  - Across seeds: unknown. Measure std of S1 and ECE over 3 seeds on the 0.8B
    pilot (ASSUMPTION cost: 3 x <= 1 GPU-h, `REQ:336`). If seed std exceeds
    0.5 pt, every gate threshold below is widened to 2 x seed std.

---

## 12. SLIs, regression gate, alerts

**Why:** a model with no gate gets replaced by whichever checkpoint finished
last; a gate turns "I think it's better" into a paired CI.

### 12.1 The eval dashboard (per checkpoint, one JSON + plots)

Rows: overall, per type (Choice, Score, Noul), per length bucket (0-512,
512-2k, 2k-8k, 8k-16k, `REQ:240-241`), per source (public gold, each synthetic
workflow, held-out templates), per `dist_source`. Columns:

| Metric | Definition source |
|---|---|
| accuracy vs consensus (+ paired delta vs B0 and vs incumbent, 95% CI) | `REQ:222-225,242-243` |
| gold accuracy (MMLU-Pro, HotpotQA slices) | `REQ:226-227` |
| ECE raw and T-scaled, equal-width and equal-mass, 15 bins, bootstrap CI | `REQ:228-231` |
| NLL and Brier, hard and soft | `REQ:232-233` |
| order: top-1 agreement, gold-rank flip rate, mean TV distance | `REQ:234-237` |
| isolation max abs diff; MLX vs PyTorch parity | `REQ:260-261` |
| coverage: `n_scored == n_manifest` | this file |
| reliability diagram per type | S13 |

### 12.2 Regression gate (candidate C vs incumbent I; I starts as B0)

Iteration uses the dev split; the release gate uses test, at most 3 reads per
release (ASSUMPTION budget), each logged in `test_access.jsonl`. That read count
is the project's error budget: past it, the test split is spent and further
tuning is reported as such.

**Preconditions (any failure rejects, no metrics are considered):** split
manifest hash match (C2); tokenizer asserts (C5); branch-vs-full and isolation on
50 items of this checkpoint (C4); metric self-tests pass at this commit (C8); T
provenance equals cal hash and argmax raw == T-scaled (C7); no per-type slice
below chance - 3 SE (C1); coverage 100%.

**Primary:** accuracy vs consensus, paired bootstrap 10k resamples (`REQ:243`):
C replaces I if the CI lower bound of (C - I) > 0, or if it is non-inferior
(lower bound > -0.5 pt) and T-scaled NLL improves with CI excluding 0.

**Guardrails (all must hold):**

| Guardrail | Threshold |
|---|---|
| T-scaled ECE overall | <= 0.05 (MVP, `REQ:251`) and <= I + 0.01 |
| T-scaled ECE per type | <= 0.07 (`REQ:252`) |
| order top-1 agreement | >= 0.90 (`REQ:257`) and >= I - 0.02 |
| gold-rank flip rate under reversal | <= 15% (`REQ:256`) |
| >2k bucket accuracy | >= B0 - 1 pt (`REQ:259`) |
| gold-slice accuracy | >= I - 2 pts (C9 signature) |
| isolation | <= 1e-3 bf16 (`REQ:260`) |
| Brier T-scaled | <= I |

On pass, the gate script writes the report, tags the Hub revision, and moves
`best` (P10). Nothing else may move it.

### 12.3 Alerts: page vs ticket

"Page" means a phone push; "ticket" means a line in the morning review.

| Signal (symptom) | Class | Threshold |
|---|---|---|
| GPU billing with no progress | page | no step in 10 min, or queued/idle > 30 min |
| Spend | page | job reaches 80% of its cap; cumulative > 60% with no S2 MVP model |
| Run exited nonzero or crash loop | page | any |
| Run finished | page | any (so nothing is left running) |
| tokens/s below floor | page | < 50% of Phase 0 baseline at step 50 |
| NaN / inf loss | page | any |
| Teacher breaker open | ticket | opened; page if it opens 3 times in 1 h |
| Parse failure rate | ticket | > 2% over the last 200 calls |
| Teacher disagreement | ticket | argmax disagreement > 20% on a type (`REQ:449` uses 80% agreement) |
| Canary drift | ticket, and labelling stops | < 95% agreement (D1) |
| Backup lag | ticket | > 1,000 labels or > 24 h (P1) |
| Dev ECE drift during training | ticket | > 0.02 above the previous eval point |

### 12.4 The morning-after view (this project's "3 a.m. dashboard")

One `status` command prints: run state and last heartbeat; last valid checkpoint
step and age; spend today, per job, cumulative vs 200 USD; tokens/s trend vs
baseline; train loss, restricted-softmax entropy and dev accuracy/ECE at the
last 5 eval points; alerts since yesterday. What you do with it: if a run is
stopped, runbook 1; if spend moved without a checkpoint moving, hunt the
forgotten instance first (B1).

---

## 13. Observability

**Why:** Modal keeps logs 1 day on the free tier (`rl-wordle/PLAN.md:101`), so
anything not written to your own storage is gone by the time you look.

### 13.1 Training, per step (local JSONL on the Volume is canonical)

loss (total and per type); restricted-softmax entropy mean and p10 (collapse
toward 0 suggests a leak or over-confidence; stuck near `log(n)` suggests wrong
rows or no signal); mean max-prob; train accuracy vs target argmax; grad norm
before clipping and clip fraction; lr; tokens/s and step time; GPU memory
allocated and reserved peak; batch sequence-length max and mean; truncated count;
per-type mix in batch; NaN/inf count; `data_cursor`; elapsed cost (elapsed x
hourly). Every 15 min: checkpoint write time, upload result, sha256. Every N
steps: dev subset (500 items) accuracy, NLL, ECE.

### 13.2 Labelling, per call and per session

Ledger per call: cache key, teacher, requested and returned model id, input /
output / reasoning tokens, price used, cost, latency, HTTP status, attempt count,
parse status, `dist_source`, chosen-position index. Per session: spend vs cap,
cache hit rate, retry fraction, 429 rate, parse-failure rate, refusal rate,
teacher disagreement rate and mean JS divergence per type and template,
chosen-position histogram per teacher (teacher position bias), canary agreement,
latency p50/p99 (feeds section 5).

### 13.3 Gaps: failures invisible without an addition

| Invisible failure | What to add | Where |
|---|---|---|
| Kernel fallback (B3) | log line with the resolved DeltaNet/conv implementation; tokens/s floor | model init, step 50 |
| Teacher version drift (D1) | returned `model` field per call; canary | ledger, session start |
| Swallowed upload failure (H1) | post-upload list + sha256 compare | checkpoint thread, preflight |
| Trainer vs engine token mismatch (C4) | `sha256(ids)` in eval records | engine eval writer |
| Silent truncation (C10) | truncated count per bucket | dataset build, batch log |
| Long-bucket miscalibration hidden by overall ECE (C7) | per-bucket ECE with CI | eval report |
| Length-biased labelling from timeouts (4.1) | labelled vs requested length distribution | dataset build |
| Test split overuse (C2) | `test_access.jsonl` | eval harness |
| Throttled latency numbers (M2) | thermal state per bench | latency bench |
| Spend between invoices | `spend.csv` from ledger + elapsed GPU hours daily | status command |

---

## 14. Disaster recovery

**Why:** the only thing here that is expensive to lose is the label cache, and
you will not find out whether the backup works until you need it.

| Asset | Primary | Backup | RPO | RTO | Restore tested |
|---|---|---|---|---|---|
| Code, configs, split manifest, `spend.csv`, eval reports | local git | GitHub remote, push each session | 1 session | 10 min clone | never |
| Label cache + raw responses (~250 MB, `REQ:288`) | local SQLite | private HF dataset repo (versioned) | 1,000 labels or 24 h (P1) | ~10 min download (ASSUMPTION) | **never** |
| In-flight training state | Modal Volume | HF model repo copy | 15 min + restart (section 9) | <= 15 min (`REQ:355`) | never |
| Released models | HF model repo, immutable tags | local copy of the tagged adapter | 0 | minutes | never |
| Base weights | HF upstream at pinned sha | Volume + local cache | n/a | 1.5 to 8 min (section 6) | n/a |

### 14.1 Plain statement

Nothing exists yet, so **no backup exists and no restore has ever been tested.**
Until the drill below passes once, the label cache is one laptop disk away from
being re-bought: 15 USD bulk + 8 to 20 USD strong-tier reference = 23 to 35 USD
(DERIVED, `REQ:315-318`), and not exactly re-buyable if a teacher version is
retired (P9).

### 14.2 RTO / RPO, stated

RPO: 1,000 labels (~0.25 to 1 USD of spend, DERIVED from `REQ:311`) for labels;
15 min of GPU for training. RTO: 2 h for a full clean-clone reproduction of the
eval report (`REQ:361`).

### 14.3 Restore drill (run after the first 1,000 labels, then before every release)

In a fresh temp directory: clone the repo, `hf download` the label dataset repo
at its latest revision, verify row count and sha256 against the manifest, run
the labeller in dry-run mode and assert a 100% cache hit rate, rebuild the
dataset and assert the split manifest hash matches. Record the date and elapsed
time in `reliability.md`; that elapsed time is the real RTO.

---

## 15. Runbooks (top 3 most likely incidents)

**Why:** you will be tired and annoyed when these happen; the first check should
already be decided.

### RB1. GPU run dies mid-epoch

- **Symptom.** Push: "run X exited nonzero", "no step in 10 min", or the morning
  view shows the run stopped before `max_steps`.
- **First check.** The run's JSONL on the Volume (not Modal logs; 1-day
  retention): last 20 lines. Classify: preemption / SIGTERM, OOM (note the
  `data_cursor` and batch max length), exception, watchdog. Then the latest valid
  checkpoint manifest (step, age) and the spend line.
- **First action.** Preemption: `train resume --run-id X`; confirm the first
  logged loss is within noise of the last pre-crash value and `global_step`
  continues. OOM on a specific batch: lower max tokens per batch, keep the global
  batch with gradient accumulation, resume, and record the config diff. Crash on
  resume twice: stop retries, reproduce locally with the tiny config (T0) before
  spending another GPU minute.

### RB2. Teacher returns malformed output or missing logprobs, or a 429 storm

- **Symptom.** Ticket: parse failures > 2% over 200 calls, `logprobs` null,
  breaker open, or canary drift.
- **First check.** Last 50 ledger errors: status codes, the returned `model`
  field (did the version change?), 3 raw responses read by eye, `Retry-After`
  values, the provider status page.
- **First action.** Stop the session; do not burn the retry budget. Version
  changed: pin the dated snapshot, rerun the canary, quarantine every label with
  the new model id. Logprobs gone: switch the remaining items to the planned
  fallback `dist_source` as a new dataset version, never mixed silently (D2).
  429s: halve concurrency, restart later; the cache makes the rerun free for
  completed items.

### RB3. Fine-tuned model worse than the base model (B0)

- **Symptom.** Gate report: CI of (C - B0) below 0 overall or on a slice.
- **First check, cheapest first.**
  1. Pipeline, not model: score 50 items with B0 through the trainer's readout
     and through the engine; same numbers? (C4)
  2. Inversion tell: any per-type slice below chance? (C1)
  3. Did it learn: train accuracy on 500 train items; overfit-32 test. (C10)
  4. Where: worse only on >2k bucket points at truncation or short training
     sequences (pngwn's 384-token lesson, `RN:140-142`); worse only on gold while
     better on consensus points at learned teacher errors (C9); worse on
     permuted orders points at position prior (C6).
  5. Training curves: entropy collapse early, grad-norm spikes, lr.
- **First action.** No new paid run until one of the above localizes the cause.
  Decode 50 random training examples to text by hand (state, question, options in
  presented order, target as text). That step is what would have caught pngwn's
  v1 inversion before a single GPU hour was spent.

---

## 16. Open questions and cross-agent needs

> TODO: unverified. Modal Volume commit semantics on preemption (is the last
> background commit guaranteed?). Verify with the Phase 0 kill test.
> TODO: unverified. Whether Modal and RunPod offer a hard provider-side spend
> limit, and whether queued time is billed.
> TODO: unverified. Whether the chosen cheap teachers return logprobs, and with
> reasoning enabled.
> TODO: unverified. Digit tokenization for Qwen3.5 and MiniCPM5 tokenizers (C5).
> TODO: unverified. Seed-to-seed std of S1 and ECE (section 11).
