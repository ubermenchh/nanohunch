# Progress

## Current plan: nanohunch MVP, jarvis edition (proposed 2026-09-24, awaiting approval)

Same phases, gates, kill criteria and calendar rules as `docs/design/2026-09-23-nanohunch/plan.md`
(the design reference). What changed is who types what: jarvis writes scaffolding, glue, every
test and runs everything; you type the core from full-code hand-offs of at most ~40 lines, each
turning named tests green. `(Pn sk)` points at the design plan's phase and step.

Legend: `[jarvis]` autonomous · `[you]` hand-off or your judgement · `GATE` stop and check ·
`KILL` pre-registered stop rule · `~N` approximate lines you type.

### Phase 0: measure the unknowns (week 1)
- [x] 0.1 [you] pin Python 3.12, add mlx, mlx-lm, numpy (done 2026-09-24; venv is 3.12.13)
- [x] 0.2 [jarvis] download `openbmb/MiniCPM5-2B-Base` (4.7 GB in HF cache)
- [x] 0.3 [jarvis] scaffold (done 2026-09-24): runtime deps (pyyaml, httpx, python-dotenv,
      datasets, matplotlib), dev deps (pytest 9.1.1, ruff 0.16.8); pytest `testpaths = ["tests"]`
      + `slow` marker; ruff pinned to E/F/I/B/UP, line 120, the 3 capacity-bench copies excluded
      (byte-identical to the MEASURED scripts); `tools/loc.py`; `configs/refs.yaml` + 6 refs
      fetched as archives by `tools/fetch_refs.py` (all shas verified, all 12 plan-cited paths
      present); `bench/` copies. Qwen3-4B-Base: tokenizer and config only (weights wait for the
      Phase 3 bake-off or the Phase 1 kill fallback). Proof: pytest "no tests ran", ruff clean,
      loc 0 / 1000 exit 0, `.env data/ runs/ refs/ *.safetensors` all ignored. (P0 s2, s3, s14)
- [x] 0.4 [jarvis] P0-1 (done 2026-09-24): `bench/check_labels.py` checks Choice, Noul and all
      three Score schemes. Choice and Noul pass in both; ` 0`..` 9` fail in both (MiniCPM5
      `[242, d]`, Qwen3 `[220, d]`); options (a) and (b) both pass. New fact: Qwen3-4B-Base adds
      no BOS, MiniCPM5 does. Recorded in `reports/phase0.md`. (P0 s5)
- [x] 0.5 [you] DECIDED 2026-09-24: Score labels are option (b), space id then bare digit.
      Recorded in ADR-0003 Amendment 1 (rule 4 updated), DECISIONS, plan Phases 0 and 2;
      `check_labels` defaults to it.
- [x] 0.6 [jarvis] P0-4 (2026-09-24, `bench/equiv_real.py`): GPU fp32 1.08e-3, CPU fp32 bit-exact;
      DECIDED: oracle on CPU fp32, 1e-3 kept (R8 amended, plan Phases 0/2/5/6 updated).
      P0-6 (`bench/anomaly.py`, 14 min): explained (outcome A): the 0.12 gap needs the full-vocab
      head over every position of a huge batch on Metal; ADR-0002's last-position label readout
      batches fine (2.9e-4). Batch 1 kept for the MVP; R9 marked for re-review.
      Also built: `skeleton/fmt_ref.py` + `tests/test_fmt_ref.py` (3 passed),
      `bench/sample_items.py` (50 rows), `bench/make_raw_model.py` (MiniCPM5 raw dir verified).
      (P0 s7, s8)
- [ ] 0.7 [you] `chmod 600 ~/.modal.toml` (hygiene). No OpenRouter key needed any more (no
      account, 2026-09-24). An HF token is only needed at Phase 7 release; `.env` can wait.
- [x] 0.8 [jarvis] P0-7 REPLACED: local teachers in MLX (`bench/teacher_spike.py`). Gemma 4
      26B-A4B QAT 4-bit (2026-09-24): noul 0.88, choice 0.92 pooled, flip 24%, 14.8 GB. Qwen3.6-35B-A3B
      3-bit (2026-09-25): noul 0.72, choice 1.00 pooled, flip 4%, 16.8 GB. Consensus noul 0.84, choice
      1.00 (n=25). Same-day bit-identical; 0.025 drift vs a 12 h-old run, cause unknown. Next:
      ADR-0004 amendment 2 (open-weight teachers run by us), pending your gate confirmation. Was: Qwen3.6-35B-A3B 3-bit: fit in memory, tok/s on the 50 P0 items,
      labels single-token in each tokenizer, agreement with gold. $0. Then ADR-0004 amendment
      (open-weight teachers run by us). Research: `.jarvis/research/2026-09-24-free-teacher-options.md`.
      KILL: neither teacher usable and Modal fallback refused: gold + spec-gold training (R10).
- [x] 0.9 [jarvis] P0-8 DONE 2026-09-25 (102 min): 470-560 tok/s first hour, 710-890 after,
      min 467 after minute 30, memory 8.52 GB flat, no OOM. Kill criterion does not fire; Mac
      trains. Literal drop rule FAILs only because of a mid-run speed-up: **you confirm at 0.10**. (P0 s4, s12)
- [x] 0.10 GATE confirmed by you 2026-09-25. ADR-0004 Amendment 2 (local teachers), R10 updated.

### Phase 1: walking skeleton, a trained-vs-B0 number (week 2)
- [x] 1.1 [jarvis] (2026-09-24) `skeleton/prep_gold.py`: train=1000 eval=1000 shared_group_keys=0,
      20 decoded rows read (gold text correct). `skeleton/fmt_ref.py` done earlier. `tests/test_skeleton.py`:
      5 passed, the 2 reader tests red with ModuleNotFoundError (right reason). **Hand-off 1.2 given.**
- [x] 1.2 DONE 2026-09-24 (typed by you): 4/4 in test_skeleton.py; smoke on real MiniCPM5: probs
      sum to 1, noul identical under --reverse, choice remap correct (confident items keep their
      canonical top-1 in both orders; 1/5 flipped). Pending: ruff format (your file, your call).
      Was: [you] `skeleton/b0_reader.py` ~35: load, render one question as one sequence, gather
      the label rows at the last position (asserting each label is one id), softmax, `--reverse`
      flag with the display-to-canonical remap. Your first contact with the readout. Turns
      `test_reader_reverse_maps_to_canonical` and `test_reader_label_must_be_single_token` green
      (fake model and tokenizer, no weights needed). (P1 s3)
- [x] 1.3 [jarvis] DONE 2026-09-25: `skeleton/tiny_eval.py` (3 tests red then green; suite 10
      passed). B0 on 1,000 eval: noul 0.670 (predicts yes 97%, gold 64%), choice 0.790, flip 0.202,
      ece15 all 0.067; gate pass; deterministic (0.0). `reports/skeleton.json` + `reports/skeleton.md`.
- [x] 1.4 [jarvis] 2026-09-25. First run (2 epochs, lr 2e-5) failed the sanity rule (choice -3.2,
      flip 34%, learned position bias); own valid-split bug fixed. Bounded debug: 1 epoch at lr 5e-6
      = reference `lora`: noul 0.806 (+13.6), choice 0.802 (+1.2), flip 0.192, ece15 all 0.049.
      GATE pass. Plan Phase 1 step 7 and Phase 5 skeleton_repro updated to 950 iters, lr 5e-6.
      [you] 3-line interpretation in `reports/skeleton.md`, then commit (Phase 1 gate).
      Was: `to_lora_jsonl.py` + stock `mlx_lm.lora` on 950 gold items + eval.
      `reports/skeleton.md`: B0 vs stock LoRA. [you] 3-line interpretation. GATE (P1 gate:
      below-chance tell). These are the reference numbers Phases 2 and 5 must reproduce.

### Phase 2: the inference core (week 3)
- [~] 2.1 [jarvis] part 1 tests written and red (6, ModuleNotFoundError: fmt), `tests/conftest.py`
      (`tok` fixture, slow-skip). Part 2 (render) tests come with hand-off 2.3. Was: `tests/test_fmt.py` red (prefix identical across questions, never
      re-tokenize the join, label not single token raises, identity-then-reversed-then-cyclic
      perms incl. the non-self-inverse `(1, 2, 3, 0)` case, too many options, head_tail flag,
      golden v1 incl. Score entries).
- [x] 2.2 DONE 2026-09-25 (typed by you): 6/6 part 1 tests, fmt.py 49/130. Dropped: docstring and
      the n=2 dedupe comment (worth restoring). Not ruff-formatted (your call). Was: `fmt.py` part 1 ~40: dataclasses, errors, `FORMAT_VERSION`, `label_vocab`,
      `permutations_for`.
- [x] 2.3 DONE 2026-09-25 (typed by you): 12/12 test_fmt, fmt.py 91/130. First try had one line
      with two silent-failure bugs (branch BOS + missing `\nAnswer:`), caught by the fmt_ref oracle
      tests, fixed. Open nit: `fmt.py:86` zip needs `strict=True` (ruff B905; my hand-off's omission).
      Was: `fmt.py` part 2 ~50 (two blocks): helpers + `render`.
      6 render tests red (ImportError). Hand-off rehearsed in an isolated copy: 12/12 pass.
      head_tail truncation cut (OE O8; plan + registry updated). Was: `render` (split tokenization, caps, truncation).
- [x] 2.4 [jarvis] `tests/test_engine_readout.py` (6) and `tests/test_engine.py` (4 slow, real
      model: head path, isolation == 0.0, chunked vs one-shot, engine vs your reader) written red;
      part 2 hand-off rehearsed in /tmp: 10/10 in 17 s.
      Was: `tests/test_engine_readout.py`, `tests/test_engine.py` (slow) red.
- [x] 2.5 DONE 2026-09-25 (typed by you): 6/6 readout tests, engine.py 44/200, green first try.
      Formatting only (blank lines). Was: `engine.py` part 1 ~50: `label_logits`, `pool`, `Answer`, `to_answer`.
- [x] 2.6 DONE 2026-09-25 (typed by you): 10/10 (4 slow real-model + 6 readout), engine.py 74/200,
      green first try. Nit: comment typo `tier` -> `tied` (engine.py:65). Was: `engine.py` part 2 ~40: `MLXBranchScorer`, chunked prefill, trim branching.
- [x] 2.7 DONE 2026-09-25 (typed by you): 12/12, CPU fp32 oracle 4.5e-6 (limit 1e-3). engine.py
      81/200. R8 correctness proven. Nit still open: engine.py:65 `tier` -> `tied`. Was: part 3 ~8: `score_reencode` (no dtype arg:
      registry + plan changed so no second model copy is ever loaded). 2 new slow tests red;
      rehearsed 12/12, CPU fp32 oracle 4.5e-6 (limit 1e-3).
- [x] 2.8 [jarvis] `tests/test_calibrate.py`: 4 red (recovery at 20k/5%, mixed option counts,
      bounds, apply). Plan's 5k/2% recovery was seed-flaky (measured). Hand-off rehearsed 4/4.
- [x] 2.9 DONE 2026-09-25 (typed by you): 4/4, calibrate.py 41/170, ruff clean. Was: part 1 ~50: `Calibration`, `fit_temperature`, `apply`.
- [~] 2.10 [jarvis] 2026-09-25: `cli.py eval`, `bench/compare_skeleton.py`, `bench/latency_p2.py`,
      `reports/phase2.md`. Acc gate pass (+0.2/-0.2 pt), flip pass; per-item 2e-2 fails on 27/1000
      (max 2.72e-2) = bf16 kernel-shape rounding (fp32 CPU: <= 2.8e-6). W1 1.73x synthetic (batch 1).
      DECIDED 3e-2 (you) -> match gate PASS. Golden fixture written (24 cases incl. Score),
      `test_golden_v1_frozen` red then green. Suite 40 passed; core 213/1000. PHASE 2 GATE PASS.
      [you] tag `nanohunch-fmt-v1` + commit. Was: `cli.py decide|eval` wiring, compare against `reports/skeleton`, W0/W1/W2
      latency. GATE: within 0.5 pt per type and 2e-2 per item; isolation 0; fp32 oracle
      <= 1e-3; `tools/loc.py` pass. Freeze `nanohunch-fmt-v1` (tag). KILL: oracle > 1e-3
      after 6 h: full re-encode everywhere.

### Phase 3: eval harness, bake-off, external anchors, Milestone 1 (week 4)
- [x] 3.1 [jarvis] 2026-09-25: `configs/split.yaml` (salt generated; **you commit it before any test
      number**), `tests/test_split.py` (4) + `tests/test_calibrate_metrics.py` (10) red. Two of my own
      tests were wrong (mass-bin tie ordering, too-weak group correlation), found in rehearsal, fixed.
      Hand-offs 3.2+3.3 rehearsed: 18/18. Was: `tests/test_calibrate_metrics.py`, `tests/test_split.py` red (calibrated
      sampler ECE < 0.01, conf 1.0 in last bin, bootstrap of a model vs itself is 0, groups).
- [x] 3.2 [you] DONE 2026-09-26: `calibrate.py` metrics (`accuracy`, `ece`, `nll`, `brier`,
      `flip_rate`); focused metric tests pass.
- [x] 3.3 [you] DONE 2026-09-26: `calibrate.py` `paired_bootstrap` (with groups) and
      `dataset.py` `assign_split`; focused metric and split tests pass (14 total). Corrected the
      `paired_bootstrap` return annotation to `tuple[float, float, float]`.
- [~] 3.4 [jarvis] Adapters and config written; `sources/external.py` converts SemIf and
      JevBench; `sources/public.py` converts BoolQ, ARC, CSQA, HotpotQA and builds padded eval
      rows. Eval-only hashes generated (475 unique). Full `eval_v1` build verified offline from
      local dataset caches: cal=2,000 and test=3,500; split disjointness, gold bounds, source
      counts, padding count and hashes audited. Added compatibility for current CSQA/Hotpot Hub
      schemas and length-safe Hotpot padding. SemIf (144) and supported JevBench (213) anchors
      exported. pngwn fields remain unfilled because its schema is still unavailable.
- [x] 3.5 [you] DONE 2026-09-26: `EvalItem`, `Predictor`, and the first `run_eval` loop.
      The output filename correction and Ruff import cleanup are verified; evaluator smoke test
      passes. Full suite not rerun.
- [x] 3.6 [you] DONE 2026-09-27: length buckets, canonical permutation scoring, option-count ECE,
      `family_balanced_accuracy`, and ID-aligned grouped baseline bootstrap. All 13 evaluator
      tests pass and Ruff is clean.
- [~] 3.7 [jarvis] Partial 2026-09-27: config-driven `eval`, `build`, `fit-cal`; user-owned
      `load_items`, `B0Predictor`, and `AdapterPredictor` implemented and fake-engine tested.
      Real MiniCPM5 calibration completed on 2,000 cal rows; immutable model revision and
      `fitted_on` provenance recorded in `runs/b0/calibration.json`. Real 3,500-row test eval
      completed: accuracy .638, flips .239-.289; SemIf authored144 family-balanced accuracy
      .5278 vs published .686 (gate fails by 10.8 points beyond tolerance); JevBench 213 supported
      public items scored at .4789. Bounded Choice-label surface diagnostic in
      `reports/m1/semif_label_probe.json`: replacing frozen spaced labels with bare uppercase
      tokens on the nanohunch prompt lowers accuracy from .5278 to .375; this does not isolate
      token spelling under SemIf's prompt. Checkpoint and prompt audit: SemIf's published native
      BF16 row is `openbmb/MiniCPM5-2B` at `12a3808a956f869c767195e9266b59c4d21d92e2`; only
      `openbmb/MiniCPM5-2B-Base` revision `96a57cd572a02506b4500f54427dca24970c1bac` is cached
      here. SemIf uses a chat-template system/user prompt with JSON evidence, criterion and
      lettered options; nanohunch uses a plain preamble, state/question separators, and option
      lines. Thus both checkpoint and serialization differ; observed gap cannot be diagnosed as
      a readout bug from the current comparison. Qwen3-4B-Base calibration and 3,500-row test
      eval completed: accuracy .7974, paired delta vs MiniCPM5 +.1594 (95% CI .1316 to .1867),
      flips .116 to .144; the paired baseline check passes. After you updated `engine.py` for
      Qwen3.5's nested `language_model` wrapper, the 2 model-path tests pass and a real
      `score_reencode` smoke returned finite logits (2,) with tied head shape (248320, 2560).
      Qwen3.5-4B-Base calibration completed on 2,000 cal rows (T choice=.9202, noul=.7892;
      snapshot `1001bb4d826a52d1f399e183466143f4da7b741b`). Its 3,500-row test eval scored .8271,
      flips .121 to .159. SemIf authored144 family-balanced accuracy is .7014 (candidate .7917,
      evidence .8750, rule .4375), 11.2 points below SemIf's published .8132. That reference is
      Qwen3.5-4B, not this Base checkpoint, and uses a different prompt; document the result as
      non-comparable, not a proven readout defect. Remaining: Phase 3.8 decisions, latency and
      report tables/plots; PNGWN schema is still unavailable. MiniCPM5 SemIf gap remains as above.
      Q4 uses cal T=1 and paired group bootstrap: MiniCPM5 .6295 vs Qwen3-4B .8150, delta +.1855
      (95% CI +.1581 to +.2133), clearly over the pre-registered +.03 threshold. This triggers
      the plan's Modal-training item and Phases 5-6 resequencing, pending your decision. Test flip
      rates stay below the 30% P=2 trigger. `tools/loc.py` passes at 451/1,000 core lines.
- [x] 3.8 [you] Q4 DECIDED 2026-09-28: proceed with Qwen3-4B-Base and its costed Modal
      training route. Cal-only T=1 accuracy .8150 vs MiniCPM5 .6295; paired group-bootstrap
      delta +.1855 (95% CI +.1581 to +.2133), above +.03. R13 does not trigger (flip below
      30%). Calibration-grouping rule CLOSED 2026-09-28: checked `option_count_buckets` in
      the real `reports/m1/{minicpm5_b0,qwen3_4b_b0}/metrics.json` -- bucket `2` is 1700
      items, all Noul; bucket `3-5` is 1800 items, all Choice. Neither qtype has a second
      populated bucket, so the "compare buckets within a qtype" rule has nothing to compare
      on `eval_v1`, exactly as the plan predicted. Decision: keep a single T per
      `(qtype, n_perms)`, no `:{bucket}` key for M1. Re-check at Phase 6 step 15 once v2 adds
      Choice items with 6-8 options. 3.9 report prose/tag remains open.
- [~] 3.9 [you] write `reports/m1/README.md` prose (jarvis supplies tables and figures);
      push, tag `m1`. One-way door: split salt frozen. Calendar: public by end of week 5.
      Tables/figures done 2026-09-28: full bake-off (3 models), by-qtype and by-length-bucket
      breakdowns, flip rates, reliability diagrams, SemIf/JevBench anchor rows with per-tier ECE,
      latency (3 models), core line count, all decision/caveat sections. Left: your
      `## Interpretation` section, `git tag m1`, push.

### Phase 4: data v1, labeller, audit before bulk (weeks 5-6)
- [x] 4.1 [jarvis] Phase 4 contract tests written: randomized option remap, release safety,
      lookup leakage and the 20-distinct-row threshold, below-chance tell, teacher label math/cache,
      and audit agreement. Test suite remains red only on pending user-owned
      `evaluate.audit_agreement`. (P4 s1)
- [x] 4.2 [you] `dataset.py` constants, `to_display`, `assert_row_releasable`, and `build_row`
      verified. Build-row test covers stable option IDs, deterministic Choice order, fixed Noul
      order, and ascending Score values; dataset tests pass 11/11. (P4 s2-3)
- [x] 4.3 [you] `dataset.py` `renormalize`, `pool_orders`, `consensus` verified by 3 label-math tests.
- [~] 4.4 [you] `find_lookup_questions` and `leaks_answer_verbatim` implemented and tested;
      `teachers_disagree`, `route_split`, `sample_audit` remain. Dataset tests pass 13/13;
      Ruff clean. (P4 s6)
- [~] 4.5 [jarvis] `label.py` now has request construction, retry/error handling, label extraction,
      and JSONL cache. Mock cache test passes; spend-ledger integration, public adapters, generator
      driver and prompts remain.
- [ ] 4.6 [you] `sources/triage.py` ~40: `sample_spec`, `spec_questions` (these define ground
      truth for the spec-fact questions).
- [ ] 4.7 [jarvis] generate states, 600-decision pilot label (< 0.40 USD), audit CLI.
- [ ] 4.8 [you] blind audit 100 items (~2.5 h), then `evaluate.py` part 3 ~15:
      `audit_agreement`. Tripwire: < 75% on a type removes it from the headline.
- [ ] 4.9 [jarvis] label to target (cap 6 USD) and build v1 (3k train decisions, cal >= 2k,
      test >= 3.5k with >= 300 `test_ood` spec-gold decisions; eval-only hashes dropped).
      [you] read the 20 decoded rows (R6). GATE (P4 gate).

### Phase 5: the trainer, numerics gate, Milestone 2 (weeks 6-7)
- [ ] 5.1 [jarvis] `tests/test_train.py`, `gates/test_numerics_gate.py`,
      `gates/test_overfit32.py` red; configs.
- [ ] 5.2 [you] `train.py` part 1 ~35: `LoRALinear`, `apply_lora` (zero-init B, scale
      alpha/rank).
- [ ] 5.3 [you] `train.py` part 2 ~40: `restricted_soft_ce`, `permute_question`,
      `make_example` (one example per decision; target via `dataset.to_display`, the only remap).
- [ ] 5.4 [you] `train.py` part 3 ~40: `run_loop` (grad accumulation, cosine schedule,
      logging).
- [ ] 5.5 [you] `train.py` part 4 ~40: checkpoint save/find/load, `train()` auto-resume,
      `load_adapter_model`.
- [ ] 5.6 [jarvis] overfit-32, resume drill on the real model, skeleton repro (within 2 pts of
      stock LoRA), numerics gate (per-item NLL parity 2e-2, adapter applied on each prompt, fp32
      oracle 1e-3, overfit through the engine). GATE before any overnight run. KILL: gate still
      red after 8 h: stock `mlx_lm.lora` for the curve.
- [ ] 5.7 [you] write `reports/m2/prereg.md` (the kill rule in your words) and commit it
      before 5.8. Read reflex's rejected-adapter report first.
- [ ] 5.8 [jarvis] curve_1k and curve_3k runs + eval -> `reports/m2/`. KILL: held-out
      template delta at 3k (`test_ood` spec gold, CI resampled by state) < +1 pt or CI crosses 0: skip Phase 6, publish the B0 + engine +
      calibration story. [you] apply the rule.

### Phase 6: scale, full run, pre-registered eval (weeks 8-9, only if 5.8 passed)
- [ ] 6.1 [you] workflow 2 (security triage) and 3 (invoice match) spec rules in
      `sources/triage.py`, ~30 each (two hand-offs). `test_ood` routing already exists
      (`route_split`, Phase 4).
- [ ] 6.2 [jarvis] tests red, then [you] `calibrate.py` part 4 ~30: `bootstrap_ci`,
      `risk_coverage`.
- [ ] 6.3 [jarvis] build and label v2 (about 10k train, ~3 USD more).
- [ ] 6.4 [you] 200-item audit (~4 h).
- [ ] 6.5 [jarvis] full training, one epoch (estimate must fit one 12 h night, else cut long
      synthetic states or `max_seq`, then R7), numerics gate, fit-cal for trained and B0,
      `tools/decider_predictor.py` + smoke test.
- [ ] 6.6 [you] `reports/final/preregistration.md`, committed before 6.7.
- [ ] 6.7 [jarvis] final eval once: trained vs B0, pngwn, SemIf method, decider-2b, JevBench
      public; tripwires applied. KILL: test_ood CI lo <= 0: release the B0 path instead.

### Phase 7: release (week 9-10)
- [ ] 7.1 [jarvis] `release.py` + tests, licence and secret gates, staging, private HF upload,
      clean-download check.
- [ ] 7.2 [you] model card and final report prose (jarvis fills numbers and tables); flip the
      repo public; tag `v0.1.0`.

### Effort (ASSUMPTION, to be replaced by the hours log)
| Who | What | Hours |
|---|---|---|
| you | ~31 hand-offs, about 1,150 typed lines (core 1,000 + triage 100 + b0_reader 35), at ~45 min each incl. reading and debugging | 23 |
| you | reading `refs/` before Phases 2, 3, 5 | 4 |
| you | audits (100 + 200), decoded-row reads | 8 |
| you | decisions, pre-registrations, M1 / M2 / final prose, model card | 12 |
| you | slack for red gates | 10 |
| | **your hands-on total** | **~57** (was ~160 when you also wrote tests and glue) |
| jarvis | scaffolding, all tests, glue, runs, reports | not your hours |

Calendar is now bound by gates and unattended runs (labelling, overnight training), not typing:
about 5 to 6 weeks at 20 h/week. The design plan's calendar rules stay as they are (M1 public by
end of week 5, stop rule end of week 9).

## Pending decisions

- Phase 3.9: only the `## Interpretation` prose in `reports/m1/README.md`, `git tag m1`, and push
  remain (yours). Q4 is decided for Qwen3-4B-Base (ADR-0001 Amendment 3).

## Metrics
| Date | What | Before | After |
|---|---|---|---|
| 2026-09-27 | SemIf label-surface diagnostic, 144 items | spaced labels .5278 | bare labels .3750 |
| 2026-09-28 | Qwen3-4B-Base vs MiniCPM5 paired test accuracy | MiniCPM5 .6380 | Qwen3 .7974; delta +.1594 (CI +.1316,+.1867) |
| 2026-09-28 | Qwen3-4B-Base vs MiniCPM5 cal T=1 accuracy, grouped paired bootstrap | MiniCPM5 .6295 | Qwen3 .8150; delta +.1855 (CI +.1581,+.2133) |
| 2026-09-28 | Qwen3.5-4B-Base SemIf authored144 family-balanced accuracy | SemIf reference .8132 | Base run .7014 (checkpoint/prompt differ) |

## Jarvis-written (escape hatch used)
- (none)

## Session log (newest first, 2-3 lines each)
- 2026-09-28: All 6 M1 eval reruns and all 3 latency benches finished; `bench/build_m1_report.py`
  regenerated `reports/m1/summary.json`, `reports/m1/latency.md`-embedded README, and reliability
  PNGs with real numbers. Along the way found and fixed a real test-isolation bug in
  `tests/test_engine_model_paths.py` (jarvis-owned, `tests/**`): its `_scorer()` fixture forced a
  fresh `engine` import under fake MLX seams via `monkeypatch.delitem(sys.modules, "engine",
  raising=False)`; since "engine" had never been imported yet, `raising=False` made that call a
  total no-op that monkeypatch never tracks for cleanup, so the fake-mx-bound `engine` module
  (`mx.float32 = object()`) leaked into `tests/test_engine_readout.py::test_label_logits_matches_
  full_product`, which failed only in full-suite order, never in isolation. Fixed by forcing
  monkeypatch to record a restore-to-absent action regardless of prior state. Full fast suite is
  now 105 passed, 1 failed (the pre-existing, expected Phase 4 `audit_agreement` gap only).
  `uv run ruff check .` clean repo-wide; core line count 620/1000. Phase 3.9 is done except the
  `## Interpretation` prose in `reports/m1/README.md`, `git tag m1`, and push -- all yours.
- 2026-09-28: M1 rerun batch: stages 1-3 (minicpm5_b0, qwen3_4b_b0, qwen35_4b_b0) succeeded with
  the new metrics shape. Stage 4 (minicpm5_semif_authored144) crashed: `run_eval`'s flip-probe
  loop does `item.meta[name]` for `perm_seed_1/2/3` on every Choice item with >=3 options, but
  `convert_semif`/`convert_jevbench` never generate those (only `eval_v1`'s own build does, Phase
  3.4); SemIf's 144 items are all 3-option Choice, so every item hit the KeyError. JevBench has
  139 Choice items with >=3 options too, so stage 6 would have hit it as well. Not an
  evaluate.py bug -- the plan never asks for flip stats on external anchors, and the pre-session
  metrics.json for these runs already had `"flip": {}`, confirming a prior run used a config
  without perm-seed flip probes. Added `configs/eval_m1_external.yaml` (same as eval_m1.yaml,
  `flip_suite: []`) and resumed stages 4-6 with it.
- 2026-09-28: Started Phase 3.9 (M1 report). `evaluate.py` hand-off closed: `run_eval` now reports
  per-qtype and overall nll/brier/ece, accuracy by length_bucket, and a risk-coverage table (all
  called for by `phase-3.md` step 7 but missing before). 18/19 evaluator tests pass (the 19th,
  `audit_agreement`, is a pre-existing Phase 4 gap). Also fixed a jarvis-owned bug in
  `tools/prepare_anchors.py`: JevBench's easy/original/hard files were combined without tagging
  which file each row came from, so `tier` always read "public"; now tagged from the filename and
  `data/built/external/jevbench_public.jsonl` regenerated (48/60/105 easy/original/hard). Added
  `--reencode` to `bench/latency_p2.py` for Qwen3.5-4B (DeltaNet can't trim). Wrote
  `bench/build_m1_report.py` (reliability diagrams, JevBench per-tier ECE, SemIf family-balanced
  accuracy, and a generated `reports/m1/README.md` with an explicit unwritten Interpretation
  section for you). Smoke-tested against stale pre-rerun data: numbers matched prior recorded
  values (semif family-balanced .5278, minicpm5_b0 test acc .6380, qwen3_4b delta +.1594 CI
  matches). Kicked off a background rerun of all 6 M1 eval runs (metrics.json shape changed) plus
  queued latency benchmarks for all 3 models after. Both running; not yet complete.
- 2026-09-28: Closed the Phase 3.8 calibration-grouping decision: real M1 metrics confirm each
  qtype has exactly one populated option-count bucket (Noul=2, Choice=3-5), so the within-qtype
  ECE comparison the plan specifies cannot fire on `eval_v1`. Kept single T per `(qtype, n_perms)`;
  deferred the re-check to Phase 6 step 15. Only 3.9 (M1 report prose + tag) remains open in Phase 3.
- 2026-09-28: `build_row(raw, salt=...)` handoff verified: dataset suite 11/11, Ruff clean.
  Label + evaluator tests 19 pass, with the planned missing user-owned `evaluate.audit_agreement`
  as the only failure. Phase 4.2 complete; next core work is the remaining Phase 4.4 dataset filters.
- 2026-09-28: Added a focused `build_row` test from the Phase 4 schema; it fails with the intended
  missing-function ImportError and Ruff passes. The test fixes the API assumption as
  `build_row(raw, salt=...)`, since the plan specifies behavior but not a signature.
- 2026-09-28: Added Jarvis-owned `label.py` request/cache implementation. Label tests 4/4 pass;
  focused dataset/label/evaluate suite 29 passed, one expected missing `evaluate.audit_agreement`;
  Ruff clean. Spend ledger integration and remaining Phase 4 glue are outstanding.
- 2026-09-28: Corrected the ambiguous-queue fixture; dataset tests then passed 9/9. Added a
  threshold regression: duplicating decisions across 10 rows must not count as 20 rows. It exposes
  `find_lookup_questions` counting occurrences instead of distinct state ids; Ruff passes.
- 2026-09-28: User corrected `find_lookup_questions` to unpack spec pairs, store answers by field
  value, and count distinct state ids. Dataset tests pass 10/10; focused dataset/label/evaluate
  suite is 25 passed and 5 expected missing-API failures. Ruff clean.
- 2026-09-28: Checked label-math handoff: `dataset.py:65` has undefined `QType` (missing import);
  three dataset test imports fail before reaching function bodies and Ruff reports F821. `label.py`
  remains absent as expected for the cache test.
- 2026-09-28: Expanded Phase 4.1 tests after go-ahead. Focused dataset/label/evaluate suite has
  23 passes and 5 expected missing-API failures; Ruff clean. First implementation handoff is the
  lookup-leak filter, measured against at least 20 teacher-labelled rows per question.
- 2026-09-28: Reran after the `DENY_SOURCES` correction: all 8 current dataset tests pass and
  Ruff is clean. Full Phase 4.1 contract tests remain to be added before moving to later items.
- 2026-09-28: Added the release-safety test matrix. `tests/test_dataset.py`: 6 expected failures
  on missing `assert_row_releasable`, 2 passes; Ruff clean. Hand-off the dataset constants and
  release validator before continuing the rest of Phase 4.1 tests.
- 2026-09-28: User chose the Qwen3-4B route under Q4; recorded in ADR-0001 Amendment 3. Began
  Phase 4 because no `data/built/v1` or Phase 4 implementation exists in the workspace. Added
  initial dataset tests; after the `to_display` handoff, 2 pass and the license test remains red.
  Ruff flags unsorted imports in user-owned `dataset.py:3`; next handoff is release validation.
- 2026-09-28: Verified the `leaks_answer_verbatim` handoff with two focused tests: teacher-gold
  option text found in the state is flagged, absent text and spec-gold decisions are not. Dataset
  tests pass 13/13 and Ruff is clean. Next Phase 4.4 handoff: `teachers_disagree` (P4 s6).
- 2026-09-27: Ran the SemIf bare-label probe on all 144 authored items. It scored .375 versus
  .5278 with the frozen spaced-label format under the nanohunch prompt. Audit found SemIf uses a
  distinct chat/JSON prompt and publishes a different MiniCPM5 checkpoint revision; only our
  Base checkpoint is cached. The comparison does not isolate label spelling or establish a
  readout bug. The Phase 3 SemIf sanity gate remains failed pending its bounded disposition.
- 2026-09-28: Qwen3.5 scorer path handoff verified: model-path tests 2/2, predictor tests 3/3,
  Ruff clean, and real cached-model re-encode smoke produced finite 2-way logits. Cal fit used
  2,000 cal rows; test eval scored 3,500 (.8271 accuracy); SemIf anchor scored 144 (.7014 family
  balanced). Qwen3.5 SemIf reference uses a different non-Base checkpoint and prompt, so retain
  the caveat. Qwen3-4B paired test delta vs MiniCPM5 is +.1594 (95% CI +.1316,+.1867).
- 2026-09-28: Q4 computed from aligned T=1 cal sidecars only (no test leakage): MiniCPM5 .6295,
  Qwen3-4B .8150, paired group-bootstrap delta +.1855 (95% CI +.1581,+.2133). The +.03 rule
  fires; user decision is pending on advancing the Modal training item and resequencing Phases 5-6.
- 2026-09-26: Phase 3 steps 3.2 and 3.3 verified; focused checks 14 passed. Full suite remains
  unverified in this environment because Python aborted importing MLX in `test_oracle_fp32_cpu`.
  Next: 3.4 eval data adapters and eval-only hashes.
- 2026-09-26: Phase 3.4 adapters, configs and 475 eval-only hashes added; 21 focused tests pass.
  Data build cannot reach Hugging Face (DNS blocked). Step 3.5 evaluator handoff is now red on
  the missing `evaluate.py` module.
- 2026-09-26: Evaluator smoke test passes after the metrics filename correction. Ruff identified
  import cleanup still needed in user-owned `evaluate.py`; full suite has not been rerun.
- 2026-09-26: Phase 3.5 evaluator smoke test and Ruff both pass after the import cleanup. Next:
  Phase 3.6 permutation remapping, metric buckets, baseline alignment and family-balanced accuracy.
- 2026-09-26: Phase 3.6 part 1 tests written; seven cases fail for missing `length_bucket` and
  `remap_top1`, while the 3.5 smoke test remains green. Handed off these two helpers.
- 2026-09-26: Phase 3.6 part 1 tests pass (8 total). Ruff found an unused pandas import in
  user-owned `evaluate.py`; the Jarvis-owned test import spacing was corrected.
- 2026-09-26: Removed the unused import; all 8 evaluator tests pass and Ruff is clean. Next:
  continue Phase 3.6 with end-to-end permutation remapping and stored flip results.
- 2026-09-26: Wrote the end-to-end permutation test. It fails at missing `metrics["flip"]`; the
  existing evaluator tests still pass. Handed off permutation scoring and canonical flip output.
- 2026-09-26: Permutation scoring and stored canonical top-1 pass all 9 evaluator tests; Ruff is
  clean. Found a comment typo (`ot` should be `to`) at `evaluate.py:43`.
- 2026-09-26: Corrected the permutation comment typo; evaluator tests remain 9/9 and Ruff is clean.
- 2026-09-26: Added uneven-family test for `family_balanced_accuracy`; it fails on the missing
  function as intended. Test Ruff passes.
- 2026-09-26: `family_balanced_accuracy` passes the uneven-family test; evaluator suite 10/10 and
  Ruff clean. Found three comment typos in the new function.
- 2026-09-26: Two family metric comments corrected. One docstring still reads "each family each
  weight"; requested the exact wording correction.
- 2026-09-26: Family metric docstring corrected. All 10 evaluator tests pass and Ruff is clean.
- 2026-09-26: Added option-count ECE and baseline alignment tests. All three fail as expected:
  option-count metrics and baseline comparison are missing. Handed off option-count ECE first.
- 2026-09-27: Option-count ECE test passes. Ruff found import ordering and the new comment says
  `predictins`; baseline alignment tests are still red as expected.
- 2026-09-27: Option-count ECE verified green and Ruff clean. Baseline alignment tests fail for
  missing comparison behavior and missing-item validation; handed off baseline integration.
- 2026-09-27: Baseline tests now pass, including reordered row alignment and clustered resampling;
  all 13 evaluator tests pass and Ruff is clean. Asked for one malformed explanatory comment fix.
- 2026-09-27: Corrected the baseline comment; all 13 evaluator tests pass and Ruff is clean.
  Phase 3.6 complete. Next: 3.7 CLI fit-cal/eval wiring and bake-off, after the public-data build
  and pngwn field map can be verified with Hugging Face access.
- 2026-09-27: Phase 3.7 started: CLI eval test failed because legacy `main()` did not accept argv;
  replaced its parser with eval/build/fit-cal commands and config overrides. Eval CLI plus evaluator
  and calibration tests: 18 passed; Ruff and 1,000-line core budget pass. Fit-cal and real model
  runs remain blocked by missing evaluate.py loader/predictors and unavailable HF data access.
- 2026-09-27: Added `load_items` JSONL contract test; it fails as expected because
  `evaluate.load_items` is absent. Handed off the loader implementation. Next: verify after typing,
  then write predictor tests against the scorer/render boundaries.
- 2026-09-27: Loader's main contract test passes. Added optional-meta case based on public builder
  rows; it exposed `evaluate.py` using `row.get("meta" or {})`, which passes `None` to `dict` when
  meta is absent. Handed off the one-line correction; loader is not complete until both pass.
- 2026-09-27: Your correction verified: both loader cases pass; all 15 evaluator tests pass. The
  core total remains 412/1,000, though `evaluate.py` is now 152/130 against its soft per-file
  target. Next: predictor contract tests and the B0 hand-off.
- 2026-09-27: Added two fake-engine B0 predictor tests for canonical probabilities, permutation
  forwarding, re-encode selection and raw T=1 log-probabilities. First attempt imported MLX and
  aborted; fixture now injects a fake engine module before evaluate imports it. Tests fail as
  expected on missing `B0Predictor`; handed off the implementation.
- 2026-09-27: B0 predictor hand-off matches the intended contract; 2 predictor tests and combined
  predictor/evaluator/CLI checks pass (18 total), Ruff clean. Core is 444/1,000, while
  `evaluate.py` is 184/130 against its soft target; later reduce or relocate glue before proposing
  any per-file increase. Next: adapter predictor contract and tests.
- 2026-09-27: Added adapter predictor test; it fails on missing `AdapterPredictor` as expected.
  The implementation is a thin adapter-path specialization of the verified B0 predictor.
- 2026-09-27: Adapter hand-off verified: all 3 predictor tests pass; combined predictor, evaluator
  and CLI tests 19/19; Ruff clean. Adapter path is forwarded into `MLXBranchScorer`, protecting
  against silently scoring B0. Core 451/1,000; `evaluate.py` 191/130 soft target. Next: fit-cal
  CLI tests and implementation, using the new T=1 predictor interface.
- 2026-09-27: fit-cal CLI tests added red then green. Calibration output carries the HF snapshot
  hash, format version, per-qtype T and cal-data SHA; `cal_items.jsonl` captures T=1 probabilities.
  Added cal-only/bound-T safeguards and `--reencode`/configured max_context forwarding. Focused
  CLI/predictor/evaluator/calibration suite 27 passed; Ruff clean; core 451/1,000. Real run awaits
  eval_v1 data availability (not present under `data/built/`).
- 2026-09-25: Phase 2 closed: engine matches skeleton (acc +/-0.2 pt), CPU fp32 oracle 4.5e-6,
  format frozen. Next: Phase 3 (eval harness, calibrate.py part 2, dataset.assign_split, M1).
- 2026-09-25: fmt.py complete (render green after one debug round). Next: 2.4 engine tests, then
  engine.py part 1 (label_logits, pool, Answer, to_answer).
- 2026-09-25: fmt.py part 1 green. Part 2 tests written; head_tail cut per O8; render handed off.
- 2026-09-25: Phase 0 closed. Phase 2 started: fmt.py part 1 handed off (label_vocab without the
  plan's cache, recorded in plan Phase 2).
- 2026-09-25: Qwen3.6 spike done (interrupted download resumed; deleted 14 GB of stale partials).
  Both teachers pass. Phase 0 now only needs your gate (0.10). Your interpretation is in skeleton.md.
- 2026-09-25: Phase 1 numbers done: stock LoRA 1 epoch lr 5e-6 beats B0 by 7.4 pts overall, kills
  the always-yes habit. Remaining in Phase 0: Qwen3.6 spike, gate 0.10. Phase 1: your interpretation.
- 2026-09-25: stock LoRA hurts choice by 3.2 pts and adds position bias; gate rule fires. Note:
  I ran `ruff check --fix skeleton/` which covers your b0_reader.py; it was already clean
  (you had formatted it at 00:50; mtime unchanged), so nothing changed. Scoped my ruff calls since.
- 2026-09-25: P0-8 soak done: Mac is 1.7-2.9x faster than the synthetic estimate; 10k-decision
  epoch about 5-8 h. Next: 1.4 stock LoRA skeleton, Qwen3.6 spike, then the Phase 0 gate (0.10).
- 2026-09-25: 1.3 done. Headline B0 finding: BoolQ readout is near always-yes. Next: P0-8 soak
  (2 h), then 1.4 stock LoRA (to_lora_jsonl + mlx_lm.lora), Qwen3.6 teacher spike.
- 2026-09-24: your first hand-off (b0_reader) green first try, matches the hand-off line for line.
  Next: 1.3 tiny_eval + full B0 runs (jarvis), Qwen3.6 teacher spike, P0-8 soak.
- 2026-09-24: Gemma 4 local teacher passes (Gemma chat-template gotcha: empty thought block).
  1.1 done; handed off 1.2 `skeleton/b0_reader.py`. Next: your 1.2, then Qwen3.6 spike, P0-8 soak.
- 2026-09-24: CPU oracle applied to plan/R8/ADR-0003. P0-6 explained in 14 min (full-vocab head
  on a huge batch, Metal only). Next: 0.7 `.env` (you), then 0.8 teacher probe and 0.9 soak.
- 2026-09-24: Score = (b) recorded. P0-4 run: the sandbox has no Metal, so model runs go
  unsandboxed. GPU fp32 oracle 1.08e-3 at 1k tokens (0.0 at 477); CPU fp32 bit-exact. Next: P0-6.
- 2026-09-24: plan approved; 0.3 scaffold done and green (see 0.3). Nothing committed yet: the
  doc fixes, `.jarvis/` and the scaffold are all in the working tree. Next: 0.4 P0-1 label check.
- 2026-09-24: aligned this checklist with the fixed design plan (cumulative test counts, one
  remap, gates/ dir, cal-based Q4, one-epoch full run, test_ood kill slice). Added two fake-model
  reader tests to Phase 1 so step 1.2 has a red-green check. MiniCPM5 download confirmed (4.7 GB).
- 2026-09-24: switched execution from chisel/pair to jarvis; plan rewritten as [jarvis]/[you]
  steps. You had pinned 3.12 and added deps; MiniCPM5 downloaded. Found the Score digit problem
  in both tokenizers and measured the fix for option (b). Next: approve plan, then 0.3 scaffold.
