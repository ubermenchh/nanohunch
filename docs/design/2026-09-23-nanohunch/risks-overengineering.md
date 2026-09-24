# Risk register: OVER-ENGINEERING lens

Author: sd-redteam (lens OVER-ENGINEERING). Mode: GREENFIELD. Depth: standard.
Date: 2026-09-23.

Read: `_brief/research-notes.md`, `_brief/requirements.md`, `_brief/system-map.md`,
`architecture.md`, `adr/0001..0004`, `data.md`, `capacity.md` (+ `capacity-bench/`),
`cost.md`, `reliability.md`, `security.md`. Reference: `anti-patterns.md`.
`plan.md` and `README.md` are still templates; `migration.md` is empty.

Labels: `MEASURED` (observed or read today, or published by the named source),
`DERIVED` (arithmetic shown), `ASSUMPTION` (guess with a range). Every hour
estimate in this file is ASSUMPTION at a hand-written-core rate, anchored to
the work-package table at `cost.md:430-449`.

Disposition rules (from the risk-register template): every finding ends
**fixed** (design changed, say where), **accepted** (with a tripwire and a
recovery action), or **refuted** (with a specific reason).

---

## Over-engineering lens

### 0. Verdict

The architecture core is sound and appropriately small: one Python process,
files plus the HF Hub, a per-request KV cache, no queue, no database
(`architecture.md:427-437`). The readout, format and base-family ADRs are good
decisions. The problem is everything the other five files layered on top of
it. Summed, the design as written is **264 to 471 hours** of work (DERIVED
below), against an MVP budget of **100 to 120 hours / 6 weeks**
(`_brief/requirements.md:384`, requirement). At the realized 12 to 25 h/week
(A7, `_brief/requirements.md:455`) that is 11 to 39 weeks before the first
portfolio artefact exists. For a solo learner, scope that never ships is the
dominant risk, and it is larger than every BREAK-IT risk in `reliability.md`
combined.

The strongest single simplification is one the fleet already measured and did
not act on: **dense 2B LoRA training on the M5 is feasible overnight**
(`capacity.md:378-380`, MEASURED on synthetic weights), and a working MLX
restricted-row LoRA step already exists in 20 lines
(`capacity-bench/synth_bench.py:86-100`). Training the MVP on the Mac in MLX
deletes the rented-GPU job, its preemption and checkpoint-mirroring machinery,
the torch reference engine, and the MLX-vs-torch parity problem, and drops MVP
cash from ~72 USD (`cost.md:198`) to ~5 to 15 USD.

### 0.1 As-designed hours (DERIVED from ASSUMPTION)

| Work | Source | Hours |
|---|---|---|
| Base plan, Candidate B on a plain-attention 2B (C1 column) | `cost.md:430-446` | 147 to 253 |
| data.md breadth beyond cost.md's adapter + generator lines (8 workflows x ~12 templates, option banks, scale library, 8 filters, HotpotQA span-to-Choice + union-find + padding, SQuAD, MNLI filter) | `data.md:41-65`, `data.md:109-144` | +35 to 60 |
| Real-text slice + PII scrub + pointers + hydration + takedown | `data.md:52`, `security.md:715-752` | +12 to 20 |
| Labeller ops beyond cost.md (retry budget, breaker, AIMD, bulkheads, SQLite state machine, canary, backup-lag gate, top-20 parse + tail imputation + reversal pooling + JSD tiers) | `reliability.md:295-325`, `data.md:148-179`, `data.md:200-214` | +15 to 25 |
| Leakage and quality detectors (MinHash LSH, bag-of-words LR, lookup entropy, state-free probe, chi-square, hot-group gates) | `data.md:133-144`, `data.md:288-310` | +10 to 16 |
| GPU ops beyond cost.md (Hub mirror, `run.lock`, progress-gated retries, step watchdog, upload canary, image digest, `status`, push alerts, launchd watchdog, `guard.py` phase caps) | `reliability.md:199-210`, `reliability.md:596-637`, `reliability.md:750-776`, `cost.md:527-540` | +15 to 25 |
| Regression gate automation, test-read budget, `best` tag policy, 3-seed study | `reliability.md:689-748` | +8 to 14 |
| Inference extras (P in {1,2,n}, T per (type,P), runtime verify + fallback, deadline/504, head_tail, server hardening) | `architecture.md:322-331`, `architecture.md:439-506`, `architecture.md:766-793`, `security.md:575-616` | +8 to 14 |
| Release ceremony (pre-commit trufflehog, `release/build.py`, redaction filter, fine-grained tokens, NOTICE, two-config Parquet, erasure) | `security.md:413-564`, `data.md:374-398` | +8 to 14 |
| Strong-tier reference + verbalized parsing + B1 harness | `cost.md:180-184`, `cost.md:191` | +6 to 10 |
| **Total** | | **264 to 471 h** |
| Calendar at 20 h/week / 15 h/week | | 13 to 24 / 18 to 31 weeks |

### 0.2 Baseline test: what each addition buys, in the units of its requirement

The honest baseline here is the one the architect named: one Python process,
files on disk plus HF Hub, a per-request KV cache (`architecture.md:431-437`).

| Addition | Number that forces it? | Verdict |
|---|---|---|
| Rented-GPU trainer (torch on Modal) | S1 needs a trained model; the Mac trains a dense 2B at 232 to 347 tok/s (`capacity.md:187-189`, MEASURED), ~14 h for the MVP cut (DERIVED in section 4) | Not forced at MVP. Forced at target (40k decisions, 8k seq, 4B) |
| Torch reference engine + MLX/torch parity (S10) | Only forced because training is in torch | Not forced if training is in MLX. Keep a one-off B0 parity script vs HF transformers (A10) |
| Second teacher | S1 is defined as agreement with a 2-teacher mean (`_brief/requirements.md:222-225`) | Forced. Keep, it is a loop over teacher ids |
| JSD weight tiers, single-teacher half weights | none | Delete; report JSD |
| SQLite WAL + 4-state status machine | one writer, ~24k to 60k rows, crash re-pays at most ~0.001 USD (`data.md:212-213`, DERIVED there) | Not forced. JSONL or a 2-column SQLite table |
| Retry token bucket, breaker, AIMD, bulkheads | concurrency 4 to 8, whole MVP labels in ~2.8 h at the lowest tier (`capacity.md:459`, DERIVED) | Not forced. Timeouts + 3 jittered retries |
| HF Hub checkpoint mirror, `run.lock` | preemption loses <= 25 min (`reliability.md:628-632`, DERIVED there) | Not forced |
| MinHash LSH, BoW-LR, lookup entropy, state-free probe | no measured leak mechanism that the group split + held-out template does not already cover | Not forced at MVP |
| Strong-tier reference (astra + fable) | "Jev comparability", but our test set is not Jev's workflows | Not forced; human audit + gold do the job |
| Inference permutation pooling, T per (type,P) | only if S3 < 0.90 after augmentation, unmeasured (A12) | Not forced until measured |
| HTTP server + browser hardening | in functional scope (`_brief/requirements.md:59-61`), not in the user's MVP definition | Defer to after MVP |
| Tree-mask packing | the 8 A100-h per-run cap (`data.md:101-105`) | Not forced: relax the self-imposed cap by ~1 h (O12) |

---

### 1. Summary (ranked by expected weeks lost to the author)

| # | Finding | Severity | Expected weeks at stake (ASSUMPTION) | State |
|---|---|---|---|---|
| O1 | Aggregate scope is 2.2x to 3.9x the MVP time budget, with no shippable artefact until the end | BLOCKING | 7 to 18 | open |
| O2 | data.md alone is roughly the whole MVP time budget | BLOCKING | 2.5 to 4 | open |
| O3 | Rented-GPU training and its ops stack for an MVP the Mac can train overnight | IMPORTANT | 1.5 to 2.5 | open |
| O4 | Labeller built like a multi-tenant client (retry budget, breaker, AIMD, bulkheads, state machine) | IMPORTANT | 0.8 to 1.2 | open |
| O5 | Branch-vs-re-encode tolerance 1e-3 is 13x tighter than MEASURED bf16 error: runtime fallback storm and a phantom bug hunt | IMPORTANT | 0.5 to 1 | open |
| O6 | reliability.md and cost.md defend a base the ADR rejected; docs contradict each other | IMPORTANT | 0.5 to 1.5 | open |
| O7 | Leakage and quality detectors beyond what the group split already guarantees | IMPORTANT | 0.5 to 1 | open |
| O8 | Inference extras and HTTP server hardening before the model exists | IMPORTANT | 0.5 to 1 | open |
| O9 | Automated regression gate, test-read budget, `best`-tag policy, 3-seed study | IMPORTANT | 0.4 to 0.7 | open |
| O10 | Strong-tier reference pair + B1 harness: a second label-parsing path and ~19 USD for a comparability they do not deliver | IMPORTANT | 0.3 to 0.5 | open |
| O11 | Release and secrets ceremony sized for a team repo | NIT | 0.3 to 0.6 | open |
| O12 | Tree-mask packing declared "required at target" to save ~10 USD of GPU per run | IMPORTANT (target only) | 0.4 to 0.8 (prob. x 1 to 2) | open |
| O13 | Dataset build and versioning ceremony (zstd, fixed floats, Parquet configs, atomic rename, erasure) | NIT | 0.3 to 0.5 | open |
| O14 | Teacher-side extras at MVP: reversal pooling by default, top-20 tail imputation, 26-option synthetic questions | NIT | 0.2 to 0.4 | open |
| U1 | Must stay: option-id keyed labels, permutation property test, decode-20-rows | BLOCKING if cut | n/a | keep |
| U2 | Must stay: branch-equivalence oracle, fp32 and unbatched | IMPORTANT if cut | n/a | keep, fix tolerance (O5) |
| U3 | Must stay: provider-side spend hard stops (not `guard.py`) | IMPORTANT if cut | n/a | keep |
| U4 | Missing: a calendar kill/cut rule and a floor artefact by week 3 | IMPORTANT | n/a | add |
| U5 | Must stay: the cheap correctness asserts (group split, split tokenization, single-token labels, teacher allowlist, `.gitignore`, metric self-tests, request caps, Mac-run resume) | IMPORTANT if cut | n/a | keep |

---

### 2. Findings

#### O1: Aggregate scope is 2.2x to 3.9x the MVP time budget, with no shippable artefact until the end

- **Severity:** BLOCKING
- **Mechanism:** each specialist hardened its own slice against its own
  failure catalogue. None of them owned the time budget, so every file adds
  controls and no file removes any. Summed: 264 to 471 h (section 0.1) vs
  100 to 120 h (`_brief/requirements.md:384`). The build order puts training
  and the release at the end, so the author does 150+ hours of labeller,
  leakage, gate and ops code before any artefact is publishable. The usual
  sequence for a solo side project: week 4 to 6, the synthetic generator is
  half done and the eval harness has no model to evaluate; motivation drops;
  hours fall from 20 to 10; the project stalls with nothing public.
- **Trigger:** starts on day 1. We are at it now: the plan has not been written
  yet (`plan.md` is a template), so it will inherit every control unless cut.
- **Blast radius:** the whole portfolio outcome.
- **Detection:** A7's weekly hours log (`_brief/requirements.md:455`), but
  nothing in the design converts "behind schedule" into "cut scope" (see U4).
  A human notices at week 6 when MVP is not done, i.e. too late.
- **Depends on:** A7 (20 h/week, realized 12 to 25), my hour estimates
  (ASSUMPTION, section 0.1).
- **Recommendation:** adopt the MVP cut line (section 4) and the later/delete
  lists (sections 5, 6) as the planner's scope. Make "publish the B0 report"
  milestone 1 (U4). The fleet's own cost file already says this in one line:
  "time binds before the budget does" (`cost.md:462`).

#### O2: data.md alone is roughly the whole MVP time budget

- **Severity:** BLOCKING
- **Mechanism:** data.md specifies 6 public sources with bespoke conversions
  (HotpotQA span answers to Choice over context entities, union-find grouping
  over paragraph titles, padding; SQuAD v2; MNLI genre filter;
  `data.md:45-50`, `data.md:274`), 8 synthetic workflows at ~12 templates each
  with option banks, a scale library and latent specs (`data.md:111-131`),
  8 degenerate filters (`data.md:135-144`), a real-text GitHub slice
  (`data.md:52`) that pulls in the entire PII scrub, pointer release and
  takedown process (`security.md:715-752`), and 7 eval sets (dev 1,000, cal
  2,000, test_id 4,000, test_ood 2,000, mmlu_pro_gold 1,000, pngwn anchor,
  audit 250; `data.md:60-65`). Estimated 99 to 161 h for data.md as written
  (ASSUMPTION: adapters 15 to 25, synthetic 30 to 50, filters 10 to 16,
  leakage 8 to 12, splits 4 to 6, audit 6 to 10, versioning 6 to 10, real text
  12 to 20, soft-label reading 8 to 12).
- **What it buys, in requirement units:** breadth of the in-distribution test
  (S1) and a held-out-workflow generalization slice. It does not move any MVP
  SLI threshold: S1, S2, S3, S4 need one train distribution, a cal split, a
  test split with a >2k bucket, and a held-out slice.
- **Smaller version:** public gold from 4 sources (BoolQ Noul; HotpotQA
  yes/no only as Noul, plus padded 2k to 4k variants for the long bucket; ARC
  and CSQA Choice); 3 synthetic workflows x 4 templates, d = 5 decisions per
  state, each template with 1 or 2 code-written questions whose answer is a
  latent spec fact (a free gold slice that catches label inversion); 3 filters
  (positional/compound regex, normalized duplicate options, both-teacher
  entropy > 0.95). Splits: train, cal 2,000, test 3,500 including one held-out
  template per workflow reported as an OOD slice, audit 200 drawn from
  cal/test, optional dev 500 carved from train-side groups (labels already
  bought). About 10k train decisions (within A5's 5k to 15k).
  Estimated 30 to 46 h.
- **Gives up:** workflow diversity (3 not 8), a whole held-out workflow, the
  real-text distribution, MMLU-Pro and SQuAD, the pngwn-style Choice-from-span
  items. The in-distribution test is narrower, so S1 generalizes less far,
  and the report must say "3 synthetic workflows".
- **Stops working when:** the report wants a "held-out domain" claim (needs a
  4th workflow held out whole) or the target tier (40k decisions,
  `cost.md:147`). Add workflows then; the generator code is the same.
- **Detection of the problem it avoids:** week 3 with no labelled build.
- **Depends on:** A5 (5k to 15k decisions suffice for +3 pts), A18 (mean
  state ~1,000 tokens).

#### O3: Rented-GPU training and its ops stack for an MVP the Mac can train overnight

- **Severity:** IMPORTANT
- **Mechanism:** the design trains in torch on preemptible Modal GPUs
  (`architecture.md:422`), which forces: a Modal wrapper, Volume checkpoints
  every 15 min with fsync and explicit commit, a background-thread HF Hub
  mirror with a cross-store manifest resume (`reliability.md:596`,
  `reliability.md:618-622`), `run.lock` with heartbeat stealing
  (`reliability.md:205`), progress-gated retries (`reliability.md:254-258`),
  a step watchdog, `max_wall_seconds`, a 1 KB upload canary
  (`reliability.md:166-171`), image-from-`uv export` with digest recording and
  a Mac/image transformers version assert (`reliability.md:158-164`),
  fine-grained GPU tokens (`security.md:534-545`), a launchd GPU watchdog
  (`cost.md:537`), and a torch reference engine plus MLX-vs-torch parity
  (`architecture.md:419`, S10). Meanwhile capacity MEASURED: MiniCPM5-2B LoRA
  on the M5 at 347 / 297 / 232 tok/s at 1k / 2k / 4k with 6.3 / 6.9 / 9.2 GB
  peak (`capacity.md:187-189`), and concluded "an MVP run is one night plus a
  morning" (`capacity.md:378-380`). The restricted-row MLX LoRA step already
  runs (`capacity-bench/synth_bench.py:86-100`).
- **What it buys:** iteration speed (a MVP run in ~1 to 2 A100-h instead of
  ~14 h on the Mac) and a path to 4B at 8k. It buys nothing an MVP SLI needs.
- **Smaller version:** train the MVP in MLX on the Mac: hand-written LoRA,
  the same `label_logits` as serving, batch 1 with gradient accumulation,
  gradient checkpointing always on (`capacity.md:436-437`), a local
  checkpoint every 30 min plus resume, `caffeinate -i`. MVP cut is ~10k
  decisions x ~1,300 tokens = 13M tokens per epoch (DERIVED, per-decision
  length from `data.md:94-99`: 19.7M / 15k); at ~260 tok/s blended = 14 h per
  epoch on MiniCPM5-2B (DERIVED). One framework for train and serve removes
  train/serve skew by construction. Keep a one-off B0 parity check vs HF
  transformers on 5 prompts (A10) instead of a torch engine.
- **Gives up:** fast iteration (one full run per night), CUDA training
  practice (the author already has Modal and training experience,
  `_brief/requirements.md:387`), and 4B at 8k (Qwen3-4B trains at 178 tok/s,
  so 13M tokens is ~20 h; `capacity.md:190`, DERIVED).
- **Stops working when:** one run exceeds ~15 to 20M tokens (about one
  night), i.e. the target tier (40k decisions, 8k seq) or a 4B base. Then add
  ONE Modal function: Volume checkpoint + resume, `timeout=` set, retries
  capped at 2, nothing else. Pilots stay on the Mac.
- **Must verify first:** P0-1 on real weights and the P0-8 2 h soak
  (`capacity.md:728`, `capacity.md:735`); thermals may cost 10 to 30%
  (`capacity.md:383-384`, ASSUMPTION). Keep batch 1 until P0-6 explains the
  0.12 fp32 gap on MLX batched attention (`capacity.md:208`,
  `capacity.md:214-216`).
- **Blast radius / detection:** one phase; the P0-8 soak fails or a run
  exceeds 24 h.
- **Depends on:** capacity's synthetic-weight rates holding on real weights
  within ~10% (`capacity.md:112-117`), A18.

#### O4: Labeller built like a multi-tenant client

- **Severity:** IMPORTANT
- **Mechanism:** the labeller gets a shared retry token bucket, a circuit
  breaker with 10-min open and probe, AIMD concurrency halving
  (`reliability.md:312-320`), per-teacher semaphores as bulkheads
  (`reliability.md:267-271`), a 50-prompt canary every session
  (`reliability.md:133-137`), a SQLite WAL cache with a 4-state status machine
  and 10-min `in_flight` reissue (`data.md:189`, `data.md:204-213`), a builder
  that refuses while `in_flight` rows exist (`data.md:402`), and a backup-lag
  gate that refuses new sessions (`reliability.md:201`). The workload: ~12k to
  29k calls per teacher (`data.md:71`), concurrency 4 to 8, finishing in about
  2.8 h at the lowest tier (`capacity.md:459`), money at risk from a crash
  ~0.001 USD (`data.md:212-213`). The whole MVP label set costs 6.4 USD
  (`data.md:73`).
- **What it buys:** protection against a metastable 429 loop and a length-
  biased dataset from dropped long states. The second is real; the cheap check
  for it (labelled vs requested p90 per type, `data.md:323`) catches it
  without any of the flow-control machinery.
- **Smaller version:** asyncio with a fixed concurrency of 4 per teacher,
  `httpx` timeouts (connect 10 s, total 60 s), 3 retries with full jitter
  honouring `Retry-After`, no retry on 4xx. Cache = append-only JSONL, one
  line per completed response `{cache_key, teacher_id, model_returned,
  request_params, prompt, response_body, usage, ts}`, loaded into a dict at
  start; skip a truncated last line. A crash loses in-flight calls only.
  Record `model_returned` per call and assert it is constant per session
  (that is the whole teacher-drift defence at MVP; the 50-prompt canary is
  optional). `hf upload` the file to a private dataset at the end of each
  session. SQLite with only `(cache_key PRIMARY KEY, json)` is an equally
  fine choice; the waste is the state machine, not the engine.
- **Gives up:** automatic behaviour under sustained 429s (you restart with
  lower concurrency by hand), at-most-once on crash (you may re-pay 4 to 8
  calls, ~0.001 USD), two concurrent labeller processes.
- **Stops working when:** more than one labelling process writes at once, or
  a campaign runs unattended for days at 50+ in flight. Neither is in scope.
- **Blast radius / detection:** one session; 429 rate in the per-session log.
- **Depends on:** A6 prices, provider rate limits (ASSUMPTION,
  `capacity.md:451`).

#### O5: The branch-vs-re-encode tolerance is 13x tighter than the MEASURED bf16 error

- **Severity:** IMPORTANT (a correctness defect in an over-built control)
- **Mechanism:** the engine compares branched vs full re-encode and, above
  1e-3 max abs prob diff, re-scores every branch the slow way, flags
  `branch_mismatch_fallback` and dumps fixtures; "the eval harness always
  verifies" (`architecture.md:766-771`, `architecture.md:788-793`; also
  ADR-0003 verification, `adr/0003-prompt-serialization-format-v1.md:81`).
  `architecture.md:789` calls 1e-3 "the bf16 isolation tolerance", but
  isolation (alone vs batched) and branch-vs-re-encode are different
  quantities. capacity MEASURED isolation = 0.0 and branch-vs-re-encode =
  1.3e-2 in bf16 on a dense model (`capacity.md:204-213`). Sequence at MVP:
  eval starts; most low-confidence items exceed 1e-3; the engine falls back to
  full re-encode for those requests; eval wall clock rises toward the naive
  cost, up to ~10x (`capacity.md:156-159`, MEASURED 10.2x); the report fills
  with `branch_mismatch_fallback`; the author spends days hunting a KV bug
  that does not exist.
- **Trigger:** the first eval run on real weights. Near-certain.
- **Blast radius:** eval throughput and the author's confidence in the engine.
- **Detection:** immediate (the flag), but the natural reading is "my
  branching is broken".
- **Recommendation:** keep the oracle (U2), change how it is used: run it as
  a test and on a 50-item eval sample, in fp32, unbatched, threshold 1e-4
  (fp32 MEASURED 4.0e-4 on random-token states, so set it after P0-4 on real
  text; pngwn got 1.5e-5 in torch). In bf16 compare against 2e-2. Delete the
  runtime per-request verify and automatic slow-path fallback in serve; keep
  the one startup golden fixture. Amend S10 and ADR-0003 as capacity already
  suggests (`capacity.md:780-782`).
- **Depends on:** capacity's measurement generalizing from Qwen3-0.6B random
  tokens to the chosen base on real text (P0-4).

#### O6: reliability.md and cost.md defend a base the ADR rejected; the docs contradict each other

- **Severity:** IMPORTANT
- **Mechanism:** ADR-0001 picks a plain full-attention base
  (`adr/0001-plain-attention-base-family.md`). reliability.md was written
  before the architecture existed (`reliability.md:28-30`) and its controls
  target Qwen3.5: hub-kernel dependency (`reliability.md:118`), the B3
  tokens/s floor for silent DeltaNet fallback (`reliability.md:653`), T0
  tests on a 2-layer DeltaNet config (`reliability.md:373-375`), DeltaNet
  state copying in C4 (`reliability.md:467-468`). cost.md's central numbers
  use Qwen3.5-4B at 3,500 tok/s (`cost.md:170`). The architecture keeps
  Qwen3.5-4B in the Phase 0 B0 bake-off (`architecture.md:275-277`,
  `architecture.md:404`), which needs non-trimmable `ArraysCache` branching in
  the engine just to score a baseline. Other live contradictions a solo
  author must resolve by reading 5,900 lines: split key `state_id`
  (`architecture.md:835`) vs `group_key` (`data.md:268`); option index arrays
  (`architecture.md:531-543`) vs id-keyed labels (`data.md:438-442`); tokens
  per epoch 10M (`cost.md:155`) vs 19.7M (`data.md:99`); ADR-0004 says bulk
  labels come from an open-weight teacher on a GPU line
  (`adr/0004-training-data-license-and-teacher-policy.md:24`, `:43`) while
  data.md uses two API teachers (`data.md:18-20`); ADR-0001 claims teacher
  labels are locked to the student tokenizer
  (`adr/0001-plain-attention-base-family.md:55-60`), which is false once
  labels are API top-logprobs stored by option id (this makes switching base
  cheaper, which is good news).
- **Trigger:** the planner or the author implements from reliability.md.
- **Blast radius:** 0.5 to 1.5 weeks of controls and engine code for a
  rejected design; confusion on every disagreement.
- **Detection:** only by a careful cross-read.
- **Recommendation:** strike every Qwen3.5-specific control from
  reliability.md and cost.md; score Qwen3.5-4B (if at all) in the bake-off
  with naive re-encode, no branching; mark data.md section 12 D1 and D2 as
  accepted and update `architecture.md` 6.3 and D5; amend ADR-0004 to
  "open-weight-licensed teachers, served via API" and ADR-0001's lock-in
  sentence.

#### O7: Leakage and quality detectors beyond what the group split already guarantees

- **Severity:** IMPORTANT
- **Mechanism:** on top of the group split, data.md adds MinHash LSH with 128
  permutations across splits plus within-question MinHash on options
  (`data.md:138`, `data.md:303-305`), union-find over HotpotQA titles
  (`data.md:274`), a per-template bag-of-words logistic regression leak probe,
  a conditional-entropy 1:1 lookup test, a "state not needed" probe that buys
  extra labels without the state, a spec-infidelity template drop
  (`data.md:140-144`), a chi-square position test (`data.md:339-341`), and
  hot-group gates (`data.md:288-292`).
- **What it buys:** protection of S1 against inflation. The mechanisms that
  actually inflated prior art were a split by decision, a leaked 1:1 field,
  and a fixed gold position (`reliability.md:416-424`). The group key
  (template + entity bundle; HotpotQA item id), a seeded canonical shuffle
  (`data.md:244-245`), a per-template majority-class baseline, and a held-out
  template slice cover those.
- **Smaller version:** group_key hash split (keep, it is a one-way door and
  cheap); exact `state_id` and `question_id` disjointness asserts; HotpotQA:
  group by item `_id` and drop any test item whose paragraph titles appear in
  train (a set difference, not union-find; with ~1,000 of ~90k items the
  overlap is small, ASSUMPTION); per-template majority baseline printed in the
  build report; normalized exact-match duplicate options; position histogram
  printed, not tested.
- **Gives up:** detection of near-duplicate states across templates, and of
  subtle lexical shortcuts inside a template. Both show up anyway as a large
  gap between the in-template test and the held-out-template slice.
- **Stops working when:** you add real text (repos repeat boilerplate) or a
  source with known near-duplicates. Add MinHash (datasketch, ~4 h) then.
- **Depends on:** synthetic states from different templates not being near
  copies (ASSUMPTION; the held-out slice is the tripwire).

#### O8: Inference extras and HTTP server hardening before the model exists

- **Severity:** IMPORTANT (the DNS-rebinding part alone is NIT)
- **Mechanism:** the engine is specified with P in {1, 2, n} permutation
  pooling and temperatures per (type, P) (`architecture.md:305`,
  `architecture.md:322-331`, `architecture.md:664`), an
  `unsupported_permutations` error and `available_permutations` in `/v1/info`,
  head_tail truncation, deadlines with 504s, `degraded` flags, a
  per-request verify with fallback (O5), and a server with Host and
  Content-Type checks against DNS rebinding, a socket-bind assert, a grep test
  for `share=True`, and a bounded queue returning 503 (`security.md:575-616`).
  All of this precedes any evidence that P = 2 is needed (A12 is unmeasured)
  and serves one caller on localhost with nothing secret in context
  (`security.md:590-592`).
- **What it buys:** S3 insurance (pooling) and protection of the Mac from a
  hostile web page burning GPU time. The real Mac risk is OOM from a huge
  input, which the request caps already handle (keep those, U5).
- **Smaller version:** P = 1 only, T per type; keep the eval permutation
  suite (reverse + 3 stored random orders) because S3 is an MVP SLI. MVP
  interface = Python library + CLI (`nanohunch decide in.json`). When the server
  is built: stdlib or FastAPI, bind 127.0.0.1, require
  `Content-Type: application/json`, reject a non-local Host header (about 10
  lines total, no tests beyond one).
- **Gives up:** the Jev-shaped HTTP contract at MVP, and test-time order
  averaging.
- **Add back when:** the M2 order pilot or the first trained model shows
  top-1 agreement < 0.90 (then add P = 2 reversed only); the demo or the
  model card needs a live endpoint.
- **Depends on:** A12.

#### O9: Automated regression gate, test-read budget, `best`-tag policy, 3-seed study

- **Severity:** IMPORTANT
- **Mechanism:** a gate script with 7 preconditions, a primary paired
  bootstrap rule with a non-inferiority branch, 8 guardrails, incumbent
  tracking, exclusive rights to move a `best` tag, and `test_access.jsonl`
  enforcing 3 test reads per release (`reliability.md:717-748`,
  `reliability.md:210`), plus 3 pilot seeds to widen thresholds
  (`reliability.md:689-691`). This project will produce 2 to 4 candidate
  checkpoints in its life (`cost.md:186-187`).
- **What it buys:** protection against a solo author fooling themselves by
  picking the last checkpoint. The paired bootstrap on the S1 delta is
  genuinely required (`_brief/requirements.md:242-243`) and is ~20 lines.
- **Smaller version:** `evaluate.py` emits one JSON report with every MVP
  metric and the paired bootstrap CI vs B0; the author compares reports by
  eye; the report records "test split read N times". Keep checkpoint
  selection off the test split (process rule). Report "single seed" as a
  stated limitation.
- **Gives up:** mechanical enforcement; seed-variance error bars.
- **Add back when:** there are more than ~5 candidates to compare, or a
  headline delta sits within ~1 pt of zero (then run 3 seeds on the Mac).

#### O10: Strong-tier reference pair and B1 harness at MVP

- **Severity:** IMPORTANT
- **Mechanism:** cost.md plans a 6-teacher pilot including gpt-6-astra and
  claude-fable-5.1 (3.3 USD, `cost.md:180`) and a 1,000-decision strong
  reference with both (15.8 USD, `cost.md:184`). Closed models expose no
  logprobs (`cost.md:103-105`, MEASURED), so this adds a second label-reading
  path (verbalized probabilities with reasoning), its own thinking-token cost
  guard (`reliability.md:654`), and SEC-1 segregation (`security.md:349-351`).
  The stated value is Jev comparability, but Jev's published accuracies are on
  Jev's own workflows (`_brief/research-notes.md:46-62`), so an astra + fable
  consensus on our test set does not make our numbers comparable to Jev's
  table. Only B2 (Jev's API on our items) does that, for under 1 USD
  (`_brief/requirements.md:465`).
- **Smaller version:** the reference outside the label path is the human
  audit (S6) plus gold slices, which reliability already names as the only
  checks that do not share the label path (`reliability.md:239-243`). Try B2
  after MVP. Defer B1.
- **Gives up:** a "matches the expensive teachers" column; the Jev reference
  pair.
- **Saves:** ~19 USD (DERIVED: 3.3 + 15.8) and one parsing path.

#### O11: Release and secrets ceremony sized for a team repo

- **Severity:** NIT
- **Mechanism:** a pre-commit trufflehog hook on every commit and push
  (`security.md:473-494`), `release/build.py` with a staging copy, trufflehog
  filesystem scan, regex pass and fail-closed upload (`security.md:522-527`),
  a root-logger redaction filter (`security.md:528-532`), fine-grained HF
  tokens for GPU boxes (`security.md:541-545`), a PII scrubber and takedown
  process (`security.md:733-752`), and NOTICE generation.
- **What it buys:** protection of a public repo and HF artefacts from key
  leaks. The real leak vectors are `.env` in git, SDK objects in the label
  cache, and whole-folder uploads.
- **Smaller version:** `.gitignore` with `.env` as the first commit (keep,
  U5); cache records written from an explicit field list (inherent to O4's
  schema, zero cost); `hf upload` of an explicit file list (weights, config,
  tokenizer, README, LICENSE), never a directory; GitHub push protection
  (free, `security.md:493-494`) plus one full-history trufflehog scan before
  the repo goes public; `chmod 600 ~/.modal.toml`; prepaid, capped teacher
  keys. With no real text at MVP (O2), the PII scrubber and takedown process
  are not needed; with no GPU at MVP (O3), fine-grained GPU tokens are not
  needed.
- **Add back when:** the public dataset release (then `release/build.py`
  with the licence gate is worth it) and the real-text slice.

#### O12: Tree-mask packing declared "required at target" to save ~10 USD per run

- **Severity:** IMPORTANT (target phase only)
- **Mechanism:** data.md computes the target naive run at 9.0 A100-h, over the
  self-imposed 8 h per-run cap, and concludes "tree packing is required at
  target" (`data.md:101-105`). Packing on a dense model needs a per-tail block
  mask plus per-token RoPE positions that restart at the state end
  (`capacity.md:250-256`); mlx-lm's RoPE takes a cache offset, not arbitrary
  position ids (ASSUMPTION from mlx-lm's cache-offset design, verify), and
  the torch side needs FlexAttention or an SDPA block mask with matching
  positions. Estimated 10 to 20 h plus a new equivalence test.
- **What it buys:** 3.3x fewer training tokens (`data.md:99`). At RunPod
  A100 secure 1.59 USD/h (`cost.md:54`, MEASURED) the target naive run costs
  9.0 x 1.59 = 14.3 USD (DERIVED); packing saves ~10 USD per run.
- **Smaller version:** relax the per-run cap to 10 A100-h, or run on an
  H100 (target naive 9.0 / 2.6 = 3.5 H100-h x 2.69 to 3.49 USD/h = 9 to 12
  USD, DERIVED using `capacity.md:345` speed ratio and `cost.md:55` prices).
- **Gives up:** ~10 to 30 USD across 1 to 3 target runs.
- **Add back when:** a run is repeated more than ~5 times, or d rises to 8+
  so the saving exceeds 5x. Otherwise treat packing as an optional learning
  stretch, not a requirement.

#### O13: Dataset build and versioning ceremony

- **Severity:** NIT
- **Mechanism:** canonical JSONL with sorted keys, fixed float format and
  zstd; a derived Parquet export with two licence configs; a manifest-hash
  version; tmp-dir writes with atomic rename; an erasure log that NULLs cache
  rows and squashes Hub history (`data.md:190-191`, `data.md:376-405`). Size:
  ~62 MB of text per build (`data.md:260-261`, DERIVED there).
- **Smaller version:** plain JSONL splits, a `manifest.json` with per-file
  sha256, seeds and counts (keep: this is the reproducibility root for S12);
  publish cal and test as JSONL with per-row licence and the AI-generated
  notice so a clean clone can reproduce the report. Parquet, two configs and
  erasure come with the public dataset release (target, S13).

#### O14: Teacher-side extras at MVP

- **Severity:** NIT
- **Mechanism:** every Choice item is labelled in two orders per teacher and
  pooled (`data.md:163-168`); tail imputation for options missing from
  top-20 logprobs (`data.md:157-159`); synthetic Choice sizes up to 26
  (`data.md:125-126`).
- **Smaller version:** measure each teacher's flip rate on 200 items first,
  as reliability itself proposes (`reliability.md:514-516`), and double only
  if > 10%. Cap MVP synthetic and public Choice at 20 options so top-20
  logprobs cover every candidate and imputation code is not needed; the
  engine still accepts 26.
- **Gives up:** set-size evaluation at 26 at MVP (not an MVP SLI; set size is
  absent from the minimum viable list, `_brief/requirements.md:426-434`).

---

### 3. Under-engineering check: what must stay

These are cheap and each guards a failure that produces plausible wrong
numbers or burns the budget. Do not let the cuts above take them.

#### U1: Option-id keyed labels, permutation property test, decode-20-rows

- **Severity if cut:** BLOCKING
- **Mechanism:** pngwn v1 shipped an index-inverted label; the gate scored it
  as correct because test labels shared the bug (`reliability.md:383-412`,
  `reliability.md:236-243`). Permutation augmentation is the likeliest place
  to repeat it.
- **Keep:** labels and distributions stored by option id with a canonical
  order (`data.md:242-249`); the property test `target_perm[j] ==
  target[perm[j]]` (`data.md:339`); a per-type below-chance tell on gold and
  spec-fact slices; print 20 fully decoded rows per build and read them
  (`reliability.md:410-412`). About 3 h total (ASSUMPTION).

#### U2: Branch-equivalence oracle, fp32 and unbatched

- **Severity if cut:** IMPORTANT
- **Mechanism:** a wrong position, mask or stale cache gives plausible
  probabilities, and a mask error shows as ~13 logit units vs 1.5e-5 noise
  (`reliability.md:479-481`). The oracle is the only check that sees it.
- **Keep:** `score_reencode` as a test on every engine change and on a 50-item
  eval sample, with the tolerances from O5. Run the oracle **unbatched**:
  MLX batched attention showed an unexplained 0.12 gap even in fp32
  (`capacity.md:208`), so a batched oracle can itself be wrong. Resolve P0-6
  before any batched MLX path, in the engine or the Mac trainer.

#### U3: Provider-side spend hard stops

- **Severity if cut:** IMPORTANT
- **Mechanism:** one forgotten A100 weekend is 76 USD, 38% of the cap
  (`cost.md:377`, DERIVED there); Modal Starter has no hard budget cap
  (`cost.md:68`, MEASURED).
- **Keep:** DeepSeek/OpenRouter prepaid balances topped up in <= 20 USD steps
  with per-key limits and auto top-up off (`cost.md:533`, `security.md:405-406`);
  when any GPU is used, Modal `timeout=` set per job; no RunPod pods at MVP;
  a hand-updated `spend.csv` weekly (S11). Delete `guard.py` phase caps, the
  launchd watchdog and `--confirm-usd` (a printed estimate suffices). With
  Mac training, the MVP has no GPU billing path at all.

#### U4: Missing: a calendar kill/cut rule and a floor artefact by week 3

- **Severity:** IMPORTANT
- **Mechanism:** the design has kill rules on accuracy and spend
  (`_brief/requirements.md:436-438`) but none on time, which is the binding
  constraint (`cost.md:462`). Without one, slippage converts into silent
  scope creep, not scope cuts.
- **Recommendation:** milestone 1 (end of week 3): B0 bake-off report
  published (Candidate A calibrated, engine latency, order suite, reliability
  diagrams). It is already a credible artefact: calibration evidence Jev does
  not publish (`_brief/research-notes.md:66-68`) plus a measured KV-branching
  engine on an M5. Rule: if milestone 1 is not public by end of week 4, cut
  to 2 synthetic workflows and the audit to 100 items; if the first trained
  model is not evaluated by end of week 8, publish B0 plus the training
  write-up and stop.

#### U5: The cheap correctness asserts

- **Severity if cut:** IMPORTANT
- **Keep (each under ~1 h, ASSUMPTION):** group_key split hashed per group
  (`data.md:268-270`, one-way door); split tokenization of prefix and branch
  everywhere (`adr/0003-prompt-serialization-format-v1.md` rule 2); the
  single-token label assert; the teacher allowlist assert in the loader
  (`security.md:342-348`, guards the release one-way door); `.gitignore` with
  `.env` as the first commit (`security.md:413-419`); 3 metric self-tests
  (calibrated sampler ECE < 0.01, temperature recovery, paired bootstrap of a
  model vs itself gives a zero-width CI; `reliability.md:544-553`); request
  caps in the engine entrypoint so a large input cannot hang the Mac
  (`security.md:618-625`); the human audit of 200 (A1 has the largest blast
  radius, `_brief/requirements.md:449`); and, because O3 moves training onto
  the laptop, a local checkpoint + resume and `caffeinate -i` for the Mac run.

---

### 4. MVP cut line

**Definition of done** (the user's): a trained model that measurably beats
its own zero-shot base on accuracy or calibration, an honest eval report, a
working local engine. Mapped to the brief: S1 (paired CI > 0), S2, S3, S4,
S6, S7, S9, S11, S13 at MVP thresholds; S10 reduced to isolation (MEASURED 0)
plus the fp32 oracle; S12 via published cal/test JSONL.

**Base:** MiniCPM5-2B-Base (plain Llama, ADR-0001-compatible). Reason, leading
with the number: it trains on the Mac at 232 to 347 tok/s (MEASURED
synthetic), serves W1 in 893 ms and W2 in 6.6 s at 11.3 GB
(`capacity.md:148-152`, MEASURED synthetic), and its 42 KB/token KV avoids the
Qwen3-4B W2 copy-branching OOM (`capacity.md:549-550`). Run the B0 bake-off
against Qwen3-4B-Base (both plain attention, same engine); if Qwen3-4B's B0 is
more than ~3 pts better, it becomes the target-tier model trained on one
Modal function. Pilots use MiniCPM5-2B itself on 1k decisions (~1.3M tokens,
~1.3 h on the Mac, DERIVED), so no separate pilot family is needed.

| # | Component (IN) | Hours (ASSUMPTION) |
|---|---|---|
| 1 | Scaffold (uv, ruff, pytest, `.gitignore` first), `schema.py`, `formatter.py` (nanohunch-fmt-v1, split tokenization, single-token labels) | 6 to 10 |
| 2 | `engine_mlx.py`: chunked prefill, `trim` branching, unbatched fp32 re-encode oracle, request caps | 12 to 18 |
| 3 | `readout.py`, `calibrate.py` (P = 1, T per type), CLI | 5 to 8 |
| 4 | `evaluate.py`: accuracy vs consensus and vs gold, ECE equal-width + equal-mass, NLL, Brier, order suite (reverse + 3 stored perms), length buckets, isolation, reliability diagrams per type, paired bootstrap vs B0, 3 metric self-tests | 18 to 26 |
| 5 | Public adapters: BoolQ, HotpotQA yes/no + padded long variants, ARC, CSQA | 8 to 12 |
| 6 | Synthetic: 3 workflows x 4 templates, spec-fact questions, LLM questions, 3 filters, group_key split, held-out template slice | 14 to 22 |
| 7 | Labeller: DeepSeek V4.1 Flash + Qwen3.6-35B-A3B logprobs via OpenRouter, JSONL cache, timeouts + jittered retries, id-keyed labels, consensus mean, JSD reported, session backup | 10 to 16 |
| 8 | Human audit, 200 decisions, uniform by type | 6 to 8 |
| 9 | Milestone 1: B0 bake-off and published B0 report | 4 to 6 |
| 10 | `train_mlx.py`: hand-written LoRA, restricted-row soft CE, permutation augmentation through ids, grad checkpointing, checkpoint/resume, overfit-32 test; first run on gold-only data to prove the loop before teacher labels exist | 16 to 26 |
| 11 | Runs and debugging (pilot on 1k, one or two full runs) | 12 to 24 |
| 12 | Model card, eval report, explicit-file HF upload, cal/test JSONL | 6 to 10 |
| | **Total** | **117 to 186 h** |

Calendar (DERIVED): 5.9 to 9.3 weeks at 20 h/week; 7.8 to 12.4 at 15 h/week.
Honest reading: the cut line meets the 6-week MVP only at its low end, so U4's
week-4 cut rule is part of the cut line, not an extra. It is still 2.3x to
2.5x smaller than the design as written (264 to 471 h).

**Data sizes (ASSUMPTION within A5):** ~10k train decisions (public gold ~4k,
synthetic ~6k from ~1,200 states at d = 5), cal 2,000, test 3,500 (includes the
held-out template slice), audit 200 from cal/test, optional dev 500 carved
from train-side groups. Teacher-labelled decisions per teacher = 6,000 + 2,000
+ 3,500 + 500 public check = 12,000 (DERIVED).

**Cash (DERIVED from `cost.md` MEASURED prices):** labels 12,000 x (0.00008 +
0.00014) = 2.6 USD, up to 3.8 if reversal pooling proves necessary (O14);
state generation ~1,200 states x 1.25 x 0.0008 to 0.0013 = 1.2 to 2.0 USD;
teacher probe < 1; OpenRouter 5.5% fee ~0.3; GPU 0 (Mac). **Total about 5 to
15 USD** with retries and a stranded minimum top-up, versus ~72 USD central
for the design's MVP (`cost.md:198`). This leaves ~185 USD for the target tier.

**What the cut line gives up relative to the design:** a Jev-shaped HTTP
server and public demo at MVP; a whole held-out workflow and real text; 4B
knowledge (the MMLU-Pro slice, not an MVP SLI); the external pngwn anchor at
MVP (first item in "later"); strong-reference and B1 columns; seed variance;
automated gating; CUDA training practice.

---

### 5. Later list (ordered by value per hour, with the trigger)

| # | Item | Trigger to add |
|---|---|---|
| 1 | pngwn anchor S5 eval (adapter + one Mac eval pass, ~4 to 6 h) | right after MVP; it is the most legible external number (0.752 arm B, `_brief/research-notes.md:116`) |
| 2 | B2: Jev API on our test items (< 1 USD) | A17 confirms access |
| 3 | HTTP `/v1/decide` + local Gradio, then ZeroGPU Space | the model card or demo needs a live endpoint |
| 4 | One Modal function for Qwen3-4B or 8k training (Volume checkpoint, resume, `timeout=`, retries <= 2) plus a torch trainer and a parity test | a Mac run exceeds one night, or Qwen3-4B B0 beats MiniCPM5 by > 3 pts |
| 5 | Workflows 4 to 8, a whole held-out workflow, MMLU-Pro gold slice, MNLI/SQuAD | target tier (40k decisions) |
| 6 | Inference P = 2 (reversed) pooling and T per (type, P) | top-1 order agreement < 0.90 on the trained model |
| 7 | Set-size curve, risk-coverage table, per-bucket ECE CIs, B1 baseline | the report wants the Jev-style cost comparison or selective-answering claim |
| 8 | Public dataset release via `release/build.py` (licence allowlist, trufflehog, Parquet configs, AI-generated notice) | S13 target |
| 9 | Real-text GitHub slice with PII scrub and pointer release | a "real distribution" claim is wanted |
| 10 | MinHash near-duplicate scan, 3-seed variance, choices-only contamination probe | real text added; a delta within ~1 pt of zero; a knowledge-slice claim |
| 11 | Strong-tier reference subset | a reviewer asks how the student compares to frontier consensus |
| 12 | 0.8B latency variant; Qwen3.5 hybrid branching; tree packing | learning stretch only (O12) |

### 6. Delete (no trigger inside this project's life)

Retry token bucket, circuit breaker, AIMD, per-teacher bulkheads (O4); SQLite
4-state machine and `in_flight` reissue (O4); backup-lag session refusal (O4);
HF Hub checkpoint mirror and cross-store resume, `run.lock`, upload canary,
launchd GPU watchdog, `guard.py` phase caps (O3, U3); runtime per-request
branch verification and slow-path fallback in serve (O5); JSD weight tiers and
single-teacher half weights (O4); `best`-tag policy and `test_access.jsonl`
enforcement (O9); root-logger redaction filter (O11); every Qwen3.5-specific
control in reliability.md and cost.md (O6); zstd and fixed-float
canonicalization (O13).

### 7. Assumptions this review depends on

- My hour estimates (all ASSUMPTION, section 0.1 and section 4), anchored to
  `cost.md:430-449`. If they are 2x too pessimistic, the as-designed total is
  still 132 to 236 h, above the 100 to 120 h MVP budget, so O1 holds.
- Mac training rates measured on synthetic weights hold on real weights
  within ~10% (`capacity.md:112-117`) and survive a multi-hour soak (P0-8). If
  P0-8 fails, O3 flips to "one minimal Modal function", and the rest of the
  cut line is unchanged.
- A5: ~10k decisions are enough to beat B0 by 3 pts. If not, the kill rule
  at `_brief/requirements.md:436-438` applies and the B0 report (U4) is the
  artefact.
- A7: realized hours. At 12 h/week even the cut line is ~10 to 15 weeks,
  which is why U4 exists.
- A12: augmentation alone reaches 0.90 order agreement. If not, add item 6 of
  the later list (about one day).
