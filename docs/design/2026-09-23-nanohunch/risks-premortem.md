# Risk register: PRE-MORTEM lens

Mode: GREENFIELD. Depth: standard. Reviewer: adversarial, lens PRE-MORTEM.
Written as if on 2027-03-23, six months after the design date.

Read (by section): REQ 3 to 4, ARCH 1, 2.4, 5, OE U4 to 7, CAP M2.4 and P0,
DATA 4, RN 4. Not read: plan.md, reliability.md, security.md.
Phases follow the cut line (`OE:613-627`): P0 = probes; Build = items 1 to 8;
M1 = published B0 report, week 3 (`OE:568-575`); Pilot = 1k run; Release.
Likelihoods are judgement (ASSUMPTION): probability of being the **primary**
cause of failure; they sum to 100%.

---

## Pre-mortem lens

### 0. The postmortem, in one paragraph

No trained model shipped. By week 9 the repo had a formatter, a KV-branching
MLX engine, an eval harness, a labeller and ~8k labelled decisions. The first
full run gave +1.1 pts over B0 with a CI crossing zero on gold slices.
Deciding whether that was a trainer bug, the data, or the truth took three
weekends; hours fell to 8 per week; the week-3 B0 report was never published
("it will be part of the real report"). The most likely cause is ordering:
every hand-written component was polished before any end-to-end number
existed, so the first signal arrived late, after the energy, and ambiguous.

### 1. Summary, ranked by expected cost

| Rank | Cause | Primary-cause likelihood | Blast radius | Earliest signal (phase) |
|---|---|---|---|---|
| F1 | Stalled at 60%: infrastructure before any end-to-end number | 35% | whole project | hours log < 15/week for 2 weeks; M1 not public by week 4 (Build) |
| F2 | Fine-tune did not beat base where it counts, or the win was hollow | 22% | portfolio claim | B0 per-slice numbers (M1); 1k pilot delta on held-out slice (Pilot) |
| F3 | Teacher consensus correlated-wrong, exposed after training | 8% | every accuracy number | teacher-teacher agreement and 100-item audit (P0 / Build item 8) |
| F4 | Teacher logprobs via OpenRouter unstable, redesign mid-project | 8% | label pipeline, 1 to 3 weeks | P0-7 probe (P0) |
| F5 | Numerical mismatch made evals untrustworthy | 5% | trust in every delta | P0-6 anomaly; trainer-vs-engine NLL parity (P0 / Build item 10) |
| F6 | Order robustness missed | 5% | S3 claim | B0 order suite (M1); pilot flip rate (Pilot) |
| F7 | Jev comparison dismissed as not apples to apples | 5% | write-up credibility | first draft review (Release) |
| F8 | External redundancy (better open model or open Jev-like release) | 5% | novelty | already partly true: pngwn shipped 2026-09-16 (`RN:92-94`) |
| F9 | Mac overnight training unreliable (sleep, thermals, crash, no resume) | 4% | 1 to 2 weeks | P0-8 soak (P0) |
| F10 | Budget blown | 3% | project stop | `spend.csv` weekly (any) |

### 2. Findings

#### F1: Stalled at 60% (most likely)

- **Severity:** BLOCKING. **Likelihood:** 35%.
- **Mechanism:** the cut line puts items 1 to 8 (scaffold, engine, readout,
  eval harness, adapters, synthetic workflows, labeller, audit: 79 to 120 h,
  `OE:615-622`) before the trainer (item 10, 16 to 26 h) and runs (item 11).
  At 20 h/week the first trained number arrives in week 5 to 8; at the
  realistic 12 to 15 h/week (`REQ:455`) it arrives in week 7 to 11. Each
  hand-written component is a legitimate learning objective (`ARCH:42-47`),
  so each one expands (the 0.12 MLX anomaly alone is a two-evening rabbit
  hole, `CAP:214-216`). The design corpus itself is ~6,300 lines and
  contradicts itself in places (`OE:299`), so the plan risks following the
  264 to 471 h design rather than the 117 to 186 h cut line (`OE:629-632`).
  Motivation for a solo learning project is fed by results; none arrive.
- **Trigger:** A7 below 15 h/week, which is the base rate for a side project
  that competes with a job. We are near it on day one.
- **Blast radius:** everything; no artefact, no portfolio piece.
- **Detection:** only via an hours log and U4's week-4 rule; else month 3.
- **Recommendation:** a walking skeleton by end of week 2 using stock tools:
  `mlx_lm` load, letter-logit readout in ~40 lines, a 10-line eval (accuracy,
  ECE, reversed-order flip) on 500 gold items (BoolQ, ARC), and a stock
  `mlx_lm.lora` run on 1k gold items, evaluated with the same script. Then
  replace each stock piece with the hand-written version, and require it to
  reproduce the stock number before moving on. This keeps the learning goal
  and front-loads the signal. Keep U4's calendar rule verbatim.

#### F2: Fine-tune did not beat base where it counts, or the win was hollow

- **Severity:** BLOCKING. **Likelihood:** 22%.
- **Mechanism:** three sub-paths, all producing "delta near zero on anything a
  reader trusts". (a) S1 is measured against the consensus of the same two
  teachers that produced the training targets (`REQ:412`), so on synthetic
  items the student gains +4 to +8 pts by learning template regularities,
  while on gold public slices (BoolQ, ARC, CSQA) a 2B base read by letter
  logits is already near its ceiling and LoRA on ~10k mostly synthetic
  decisions moves it 0 to -1. A reviewer reads the headline as "a distilled
  student agrees with its teacher on the teacher's data". This is exactly the
  pngwn result on real PRs: the untouched base matched the fine-tune on
  judgement questions (`RN:138-142`). (b) Calibration cannot rescue the claim:
  T-scaling alone brings B0 near ECE 0.02 to 0.05 (pngwn arm B 0.064 raw to
  0.015 scaled, `RN:118`), so "beats base on calibration" is weak. (c) A
  plumbing bug: the engine evaluates without the adapter, or with wrong
  alpha/r scaling, and reports B0 twice; delta is exactly zero with a tight
  CI and is misread as a real null.
- **Trigger:** depends on A5 (5k to 15k decisions suffice, `REQ:453`); M1's
  flip condition (B0 within 3 pts of the cheap teacher, `ARCH:401`) is a
  plausible outcome for a 2B to 4B base on public-style questions.
- **Blast radius:** the portfolio claim; the engine and eval survive.
- **Detection:** M1 per-slice B0 table (week 3) already shows where headroom
  exists. The 1k pilot shows direction by week 5 if F1's reorder is applied.
- **Recommendation:** pre-register the headline as delta over B0 on (i) gold
  slices and (ii) the held-out template slice, with consensus agreement as a
  secondary metric. Run the 1k and 3k learning curve before building
  workflows 2 and 3 and before bulk labelling. Kill/pivot criterion: if the
  held-out-template delta at 3k decisions is < +1 pt or its CI crosses 0,
  stop scaling data and pivot to the M1 story (engine, calibration, order
  robustness) per `ARCH:401`. Add two asserts: base vs adapted logits differ
  on 5 prompts; overfit-32 passes when scored through the engine, not only
  the trainer.

#### F3: Teacher consensus correlated-wrong, exposed late

- **Severity:** IMPORTANT. **Likelihood:** 8%.
- **Mechanism:** both teachers are open MoE models read with thinking off and
  `max_tokens=1` (`DATA:150-152`), which is far below their reasoning-mode
  accuracy and shares first-token biases. On synthetic workflows whose rules
  the author wrote, both misread the same ambiguous rule and agree
  confidently; high agreement is then taken as validity. The 200-item audit
  (`OE:622`) is tedious and slips to after training; it lands at ~72% on one
  type, below the 75% line (`ARCH:406`), invalidating trained targets.
- **Trigger:** A1 at the low end of its 65% to 95% range (`REQ:449`).
- **Blast radius:** every accuracy number, and the training targets.
- **Detection:** only via human or programmatic gold. Agreement between
  teachers does not detect it; it hides it.
- **Recommendation:** before bulk labelling, audit 100 items, oversampling
  items where both teachers agree with probability > 0.8. For spec-fact
  synthetic questions, derive gold from the generator and report teacher
  accuracy against it; this is a free, continuous audit.

#### F4: Teacher logprobs unavailable or unstable

- **Severity:** IMPORTANT. **Likelihood:** 8% as primary cause, ~30% that it
  costs at least a week.
- **Mechanism:** OpenRouter routes `qwen3.6-35b-a3b` across hosts with
  different precision and parameter support (`DATA:454`); one host silently
  drops `logprobs` or caps `top_logprobs` at 5; another ignores thinking-off
  so the first token is a reasoning marker and `candidate_mass` collapses;
  mixed hosts give soft labels that differ run to run. The labeller's
  retry-and-drop path (`DATA:154-158`) turns this into many `single_teacher`
  rows without an alarm.
- **Trigger:** P0-7 unverified (`DATA:455`). Closed strong models already
  expose no logprobs (`cost.md:103`), so the design rests on open-weight hosts.
- **Blast radius:** label pipeline; fallback is vLLM self-host (`DATA:454`).
- **Detection:** P0-7, if it measures repeatability, not only availability.
- **Recommendation:** in week 1, 50 items x 2 runs x 2 teachers with provider
  pinned (`provider.order`, `allow_fallbacks: false`,
  `require_parameters: true`); record host and quantization per row; pass
  requires mean candidate_mass >= 0.9 and run-to-run JSD < 0.01. Budget the
  fallback now: a single teacher plus gold labels is an acceptable MVP.

#### F5: Numerical mismatch made evals untrustworthy

- **Severity:** IMPORTANT. **Likelihood:** 5% as primary cause, high as a
  time sink.
- **Mechanism:** split prefill/tail differs from full re-encode by 1.3e-2 in
  bf16 (`CAP:207`), harmless if trainer and engine use the same split
  (`CAP:213`). The dangerous one is the unexplained 0.12 fp32 gap on batched
  naive re-encode (`CAP:208`, `CAP:214-216`). If the trainer uses batched
  forward at B > 1 and the gap is a real MLX kernel issue, gradients are
  computed on wrong activations and the trained model is quietly worse; if it
  is a bench bug, time is lost proving it.
- **Trigger:** any batched forward in `train_mlx.py` or the order suite.
- **Blast radius:** trust in every reported delta.
- **Detection:** P0-6 (30 min) plus a trainer-vs-engine per-example NLL test.
- **Recommendation:** time-box P0-6 to 2 h. Until explained, train at batch 1
  with gradient accumulation and forbid batched re-encode in eval. Add a
  parity test: training-forward NLL vs engine NLL on 64 cal items within
  2e-2 bf16.

#### F6, F7, F8: order, Jev framing, redundancy (IMPORTANT, 5% each)

- **F6 order:** augmentation leaves a residual letter prior; flip rate lands
  at 18% vs the 15% bar (`REQ:414`). Low because targets are pooled over two
  orders (`DATA:163-165`) and P = 2 pooling is a cheap fallback (`OE:664`).
  Signal: B0 order suite at M1. Change: if B0 flip > 30%, make P = 2 for
  Choice the default. Depends on A12 (`REQ:460`).
- **F7 Jev framing:** a Jev number beside ours on a test set we built and
  labelled with our teachers gets dismissed, and drags the whole report.
  Change: never headline Jev; headline delta over B0 and the pngwn anchor S5
  (`REQ:416`), with a B0-only S5 pass moved into M1 (~3 h). B2 is a footnote.
- **F8 redundancy:** pngwn published an open replication a week before this
  design (`RN:92-94`); a generic second one reads as a copy. Change: state on
  day one that the project fixes pngwn arm B's 37.5% order flips
  (`RN:124-125`) and long-input regression (`RN:139-142`), calibrated, on a
  laptop; every milestone reports those two numbers.

#### F9 and F10: Mac training reliability; budget (NIT, 4% and 3%)

- F9: rates are measured on synthetic weights (`OE:689-692`); a multi-hour
  run sleeps, throttles or OOMs and restarts from zero (target tier on Mac is
  37 to 67 h, `CAP:374`). Run the P0-8 soak before Pilot; keep resume and
  `caffeinate -i` (`OE:590-591`).
- F10: the cut line spends 5 to 15 USD (`OE:640-645`); hours, not dollars,
  bind. Prepaid OpenRouter credits with no auto top-up; a Modal spend limit
  only when that item triggers.

### 3. Five changes to insist on before work starts (ranked)

1. **Planner: schedule a walking skeleton that produces a trained-vs-B0 number
   by end of week 2** using stock `mlx_lm` and `mlx_lm.lora` on ~1k gold
   items with a 10-line eval. Each hand-written component then replaces a
   stock one and must reproduce its number. Enforce U4's week-4 and week-8
   rules and a weekly hours log. (F1, F2c)
2. **Planner: pre-register the headline metric as delta over B0 on gold slices
   and the held-out template slice, and run the 1k/3k learning curve before
   workflows 2 and 3 and before bulk labelling.** Kill criterion: < +1 pt or
   CI crossing 0 at 3k, pivot to the engine, calibration and order story.
   (F2, depends on A5)
3. **Planner: make P0-7 a week-1 gate with pinned providers and a
   repeatability test, and move a 100-item stratified author audit plus
   generator-derived gold ahead of bulk labelling.** (F3, F4, depends on A1)
4. **Planner: add a numerics gate before the first paid or overnight run:**
   P0-6 time-boxed to 2 h, batch-1 training until it is explained, a
   trainer-vs-engine NLL parity test, and an "adapter actually applied"
   assert. (F5, F2c)
5. **Planner: frame the deliverable against pngwn, not Jev:** B0 order flip,
   long-input bucket and a B0 pass on the pngwn test set all go into M1;
   Jev appears only as a caveated footnote. If B0 flip > 30%, P = 2 pooling
   for Choice becomes default. (F6, F7, F8)

### 4. Assumptions this review depends on

A7 hours (`REQ:455`, drives F1); A5 (`REQ:453`) and A1 (`REQ:449`); the
cut-line hour estimates (`OE:613-627`, ASSUMPTION); the 0.12 anomaly being
unexplained (`CAP:216`; if P0-6 finds a bench bug, F5 drops to NIT).
Likelihoods are judgement, not measurement.
