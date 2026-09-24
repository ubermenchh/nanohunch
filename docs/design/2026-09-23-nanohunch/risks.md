# Risk register (consolidated, with dispositions)

Orchestrator-owned. Sources: `risks-overengineering.md` (OE), `risks-premortem.md`
(PM), plus the risks returned by every specialist (architecture ARCH, capacity CAP,
cost COST, data DATA, reliability REL, security SEC, requirements REQ).

Every finding ends **fixed** (design changed), **accepted** (with a tripwire),
or **refuted** (with a reason). No finding is open.

## Precedence rule (resolves contradictions between files)

The specialist files were written in parallel and disagree in places (OE O6).
When they disagree, this order wins:

1. This file's dispositions.
2. ADRs in `adr/`.
3. `risks-overengineering.md` section 4 (MVP cut line) for **scope**.
4. `architecture.md` for contracts and mechanisms.
5. Everything else (`data.md`, `capacity.md`, `cost.md`, `reliability.md`,
   `security.md`) is reference material. Any control in them that is not in the
   MVP cut line is on the "later" list (OE section 5) or deleted (OE section 6).
   Every Qwen3.5-specific control is moot because ADR-0001 rejected Qwen3.5.

## Summary

| # | Finding | Lens / source | Severity | State |
|---|---|---|---|---|
| R1 | Scope is 2.2x to 3.9x the MVP time budget; no artefact until the end | OE O1, O2 | BLOCKING | fixed |
| R2 | Stall at 60%: infrastructure before any end-to-end number | PM F1 | BLOCKING | fixed |
| R3 | Fine-tune does not beat B0 where it counts, or the win is hollow | PM F2, ARCH | BLOCKING | fixed (kill rule) |
| R4 | Closed-API teacher outputs or NC data reach the public release | SEC-1, SEC-2 | BLOCKING | fixed |
| R5 | Secrets leak via .env, label cache or config dumps | SEC-3, SEC-4 | BLOCKING | fixed |
| R6 | Label / option index inversion (pngwn v1 bug) | REL, OE U1 | BLOCKING | fixed |
| R7 | Rented-GPU trainer stack for an MVP the Mac can train overnight | OE O3 | IMPORTANT | fixed |
| R8 | Branch-equivalence tolerance 1e-3 is 13x tighter than MEASURED bf16 error | OE O5, CAP | IMPORTANT | fixed |
| R9 | Unexplained 0.12 fp32 gap in MLX batched re-encode | CAP, PM F5 | IMPORTANT | accepted + tripwire |
| R10 | Teacher logprobs via OpenRouter unavailable or unstable | PM F4, DATA | IMPORTANT | fixed (week-1 gate) |
| R11 | Teacher consensus correlated-wrong, exposed after training | PM F3, REQ A1 | IMPORTANT | fixed + tripwire |
| R12 | Train/test leakage via shared paragraphs and templates | REL, DATA | IMPORTANT | fixed |
| R13 | Order robustness target missed | ARCH, PM F6 | IMPORTANT | accepted + tripwire |
| R14 | Jev comparison dismissed; project redundant with pngwn | PM F7, F8 | IMPORTANT | fixed |
| R15 | Qwen3.5 slow training kernels and missed Mac latency | CAP, COST, REC | IMPORTANT | refuted |
| R16 | Forgotten GPU / runaway spend | COST, REL | IMPORTANT | fixed + tripwire |
| R17 | No calendar kill rule | OE U4 | IMPORTANT | fixed |
| R18 | Mac overnight training unreliable (sleep, thermals, no resume) | PM F9 | NIT | accepted + tripwire |
| R19 | Latency headline misread as Jev parity | REQ | NIT | fixed |
| R20 | Plan conflicts with user's learn-by-writing convention | REC | IMPORTANT | fixed |
| R21 | Crowded field: about ten open Jev replications exist (SemIf, decider, reflex, von, ...) | orchestrator, 2026-09-24 | IMPORTANT | fixed (reframed) + tripwire |
| R22 | Contamination from public eval items (JevBench public half, SemIf authored144) | orchestrator, 2026-09-24 | IMPORTANT | fixed |
| R23 | Core grows past "minimal" and the claim becomes false | orchestrator, 2026-09-24 | IMPORTANT | fixed + tripwire |

---

## Findings and dispositions

### R1: Scope far exceeds the MVP budget
- **Mechanism:** each specialist added controls sized for a team service; the
  as-designed build is 264 to 471 h (OE 0.1, ASSUMPTION hours) against the
  100 to 120 h MVP budget (`_brief/requirements.md:384`).
- **Disposition: fixed.** The plan implements the OE section 4 MVP cut line
  (117 to 186 h, about 6 to 9 weeks at 20 h/week). Everything else is on the
  OE section 5 "later" list with an explicit trigger, or deleted (OE section 6).

### R2: Stall before the first end-to-end number
- **Mechanism:** the cut line still places 79 to 120 h of hand-written
  infrastructure before the trainer (PM F1); first trained number would land
  week 5 to 11.
- **Disposition: fixed.** Walking skeleton by end of week 2 using STOCK
  `mlx_lm` (load, letter-logit readout, 10-line eval) and STOCK `mlx_lm.lora`
  on about 1k gold items, producing a trained-vs-B0 number. Each hand-written
  component then replaces a stock one and must reproduce the stock number
  (within a stated tolerance) before moving on. Weekly hours log.

### R3: No real gain over the zero-shot base
- **Mechanism:** S1 measured against the same teachers' consensus rewards
  template learning; on gold slices a base read by letter logits is near its
  ceiling (pngwn saw base equal fine-tune on judgement questions).
- **Disposition: fixed.** Pre-registered headline = delta over B0 on (i) gold
  public slices and (ii) the held-out template slice, with paired bootstrap
  CI; consensus agreement is secondary. Learning curve at 1k and 3k decisions
  BEFORE building workflows 2 and 3 and before bulk labelling.
  **Kill criterion:** if the held-out-template delta at 3k decisions is below
  +1 pt or its CI crosses 0, stop scaling data and publish the B0 +
  engine + calibration + order-robustness story. Asserts: base vs adapted
  logits differ on 5 prompts; overfit-32 passes when scored through the engine.

### R4: Non-releasable data in the release
- **Disposition: fixed** (ADR-0004, `security.md`). Bulk labels only from
  releasable teachers: DeepSeek V4.1 Flash API (ToS 4.2(3) permits, outputs
  marked AI-generated) and Apache-2.0 Qwen3.6-35B-A3B. OpenAI/Anthropic models
  never touch training, selection, calibration or published labels. ANLI and
  all pngwn artefacts are eval-only. A loader assert rejects rows whose
  `teacher_id` or `source_license` is not on the allowlist.

### R5: Secrets leak
- **Disposition: fixed.** First commit is `.gitignore` (with `.env`) plus
  `.env.example`. The label cache stores an allowlisted set of fields only (no
  request objects, no headers). HF uploads are explicit file lists, never the
  repo root. `chmod 600 ~/.modal.toml` (MEASURED 644 today). trufflehog
  pre-commit is on the later list (OE O11); a one-line `git grep` for key
  prefixes runs before every push.

### R6: Label / option index inversion
- **Disposition: fixed** (OE U1, `data.md`). Labels and distributions are
  keyed by option id with a seeded canonical order; property test
  `target_perm[j] == target[perm[j]]`; below-chance tell on gold slices; print
  and read 20 fully decoded rows per dataset build.

### R7: Rented-GPU stack for the MVP
- **Disposition: fixed.** MVP trains in MLX on the M5. Measured on synthetic
  weights: MiniCPM5-2B LoRA 347 / 297 / 232 tok/s at 1k / 2k / 4k
  (`capacity.md`, MEASURED); a 10k-decision epoch is about 14 h (DERIVED).
  One minimal Modal function is added only if a trigger in OE section 5 item 4
  fires (Mac run exceeds one night, or Qwen3-4B B0 beats MiniCPM5 by > 3 pts).

### R8: Branch tolerance unreachable
- **Mechanism:** split prefill/tail vs full re-encode differs 1.3e-2 in bf16
  and 4.0e-4 in fp32; isolation diff is 0.0 (`capacity.md`, MEASURED).
- **Disposition: fixed.** Trainer and engine use the identical split
  tokenization (ADR-0003 rule 2), so the bf16 split difference is part of the
  model, not an error. The oracle runs in **fp32, unbatched**, tolerance
  max abs prob diff <= 1e-3 (MEASURED 4.0e-4 today). Requirement S10 is
  redefined as: isolation diff == 0 AND fp32 oracle <= 1e-3. No runtime
  fallback path in serving (OE O5).

### R9: MLX batched-attention anomaly (0.12 in fp32)
- **Disposition: accepted, tripwire.** Time-box investigation to 2 h in
  Phase 0. Until explained: training at batch 1 with gradient accumulation,
  no batched re-encode anywhere, and a trainer-vs-engine per-example NLL
  parity test on 64 cal items (tolerance 2e-2 bf16). Tripwire: parity test
  failure blocks any run.

### R10: Teacher logprobs unstable
- **Disposition: fixed.** Week-1 gate: 50 items x 2 runs x 2 teachers,
  provider pinned (`provider.order`, `allow_fallbacks: false`,
  `require_parameters: true`), host and quantization logged per row. Pass:
  mean candidate_mass >= 0.9 and run-to-run JSD < 0.01. Fallback budgeted
  now: one teacher plus gold labels is an acceptable MVP.

### R11: Correlated teacher error
- **Disposition: fixed + tripwire.** 100-item author audit BEFORE bulk
  labelling, oversampling items where both teachers agree above 0.8;
  generator-derived gold for spec-fact synthetic questions as a continuous
  audit. Full 200-item audit before the final report. Tripwire: consensus vs
  human agreement below 75% on any question type removes that type from the
  headline.

### R12: Leakage
- **Disposition: fixed** (`data.md`). Split by `group_key` (source document,
  HotpotQA title union, workflow+template bundle), hash-assigned; state_id
  disjointness asserted; one held-out template per workflow. MinHash scan is
  on the later list (only needed once real text is added).

### R13: Order robustness
- **Disposition: accepted, tripwire.** Training uses permutation
  augmentation through option ids. Tripwire: if B0 flip rate > 30% at
  milestone 1, P = 2 (reversed) pooling for Choice becomes the default at
  inference; if the trained model's top-1 agreement is < 0.90, same action.

### R14: Framing and redundancy
- **Revision 2 note:** superseded in part by R21. pngwn arm B stays the order-flip and
  long-input reference; SemIf and decider-2b are added as accuracy and calibration references.
- **Disposition: fixed.** The project is framed against pngwn arm B, not Jev:
  it aims to fix arm B's 37.5% order flips and its long-input regression,
  calibrated, on a laptop. Every milestone reports order flip rate and the
  > 2k-token bucket. A B0 pass on the pngwn test set (eval-only, NC) goes into
  milestone 1. Jev appears only as a caveated footnote.

### R15: Qwen3.5 kernel and latency problems
- **Disposition: refuted.** ADR-0001 selects the plain-attention family
  (MiniCPM5-2B MVP, Qwen3-4B target candidate). Qwen3.5 measured 8 tok/s LoRA
  on the Mac and 642 ms W0 (`capacity.md`, MEASURED); it is a learning stretch
  on the later list only.

### R16: Runaway spend
- **Disposition: fixed + tripwire.** MVP has no GPU billing path (Mac
  training). OpenRouter/DeepSeek prepaid in <= 20 USD steps, auto top-up off,
  per-key limits. Any Modal job sets `timeout=`. Tripwire: `spend.csv`
  updated weekly; cumulative > 100 USD before milestone 3 halts paid work.

### R17: No calendar rule
- **Disposition: fixed** (OE U4), with dates re-derived by the orchestrator
  from the phase hours in `plan.md` (Phase 0 to 3 = about 70 h, so week 3 was
  not arithmetic-honest at 20 h/week). Milestone 1 (B0 bake-off report)
  public by **end of week 4**. If not public by end of week 5: cut to 2
  synthetic workflows and a 100-item audit. Milestone 2 (1k/3k learning curve
  and the R3 kill decision) by end of week 7. If no model trained on
  teacher-labelled data is evaluated by **end of week 9**: publish B0 plus a
  training write-up and stop.

### R18: Mac overnight training reliability
- **Disposition: accepted, tripwire.** Local checkpoint every 15 min with
  resume, `caffeinate -i`, 2 h soak test (P0-8) before the first overnight
  run. Tripwire: two failed overnight runs trigger the Modal item.

### R19: Latency misread as Jev parity
- **Disposition: fixed.** Report latency per workload (W0/W1/W2) with model
  size; MiniCPM5-2B W1 is 893 ms (MEASURED synthetic), not 70 to 500 ms.

### R20: Learn-by-writing convention
- **Mechanism:** `rl-wordle/AGENTS.md:9-10,52-60` says the user writes the
  core and agents pair.
- **Disposition: fixed.** The plan tells the user what to write and why;
  agents do not write core logic (formatter, engine, readout, calibration,
  metrics, loss, training loop). Stock tools are used only in the walking
  skeleton, as a numeric reference to reproduce.

### R21: Crowded field
- **Mechanism:** between 2026-09-16 and 09-22 at least ten open replications appeared (GitHub
  search, 2026-09-24). SemIf (frozen Qwen3.5-4B, about 640 core lines) scores 73.1 on JevBench v1.3
  against Jev's 74.4; decider ships trained 0.8B to 35B models (0.755 held-out accuracy at 2B).
  A generic "another open Jev" reads as a copy, and a 2B LoRA may lose to both.
- **Disposition: fixed (reframed).** The claim is **minimal**: six core files, at most 1,000
  lines, written by hand and trained on one Mac, with calibrated evals against SemIf,
  decider-2b and pngwn in one harness (`plan.md` Phase 3 step 9a, Phase 6 steps 16a and 18).
  Not "best". Tripwire: if decider-2b beats ours by more than 3 pts on PRIMARY (a), the report
  says so in the TL;DR with the cost comparison; the release is not re-scoped to chase it.

### R22: Eval contamination from public benchmark items
- **Mechanism:** JevBench's README states its public half can be trained on or selected
  against; SemIf's authored144 is public too. If any of those items or near-copies enter
  training, the external rows are inflated.
- **Disposition: fixed.** Phase 3 writes `configs/eval_only_hashes.txt` (normalized-text sha256
  of every state and question in both sets); Phase 4 drops training items whose hash matches
  and reports `dropped_eval_only`. Results on these sets are labelled "self-run, public items,
  not official". Near-duplicate (MinHash) detection stays on the later list.

### R23: Core grows past "minimal"
- **Mechanism:** each phase adds "just one more" feature to core files, and the minimal claim
  silently becomes false by Phase 6.
- **Disposition: fixed + tripwire.** `tools/loc.py` (Phase 0 step 14) counts non-blank,
  non-comment lines in the six core files; per-file budgets warn, the 1,000-line total exits 1
  and is part of every gate from Phase 2 on. Response order when over: delete, move glue out,
  then raise a per-file budget; never the total.
