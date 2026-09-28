### Phase 6: Scale data to v2, run the full training, and evaluate against B0 under a pre-registered protocol

**Goal:** `reports/final/metrics.json` exists. It comes from a single test-split run of the trained
adapter and B0, under a protocol that was committed before the run, and it states the PRIMARY delta
with a 95% CI.
**Effort:** 4.5 engineer-days (27 h: 24 h as before plus 3 h for the decider-2b and JevBench rows in steps 16a and 18; includes a 2 h interruption buffer; the unattended
training wall clock is on top of that). **Depends on:** Phase 5, and **only if** the Phase 5 kill
criterion passed (held-out-template delta at 3k decisions >= +1 pt, with the CI above 0).
**Parallel with:** nothing, since this assumes 1 engineer. Overnight training overlaps with the
audit (step 9) and with writing the pre-registration (step 16).
**Risk:** high. This phase settles A5 (data beats B0), A12 (order) and A13 (long inputs) on the
test split. It is the third look at test, after M1 and M2 (v2 test is a superset of v1 test), so
the pre-registration lists both earlier looks and nothing is tuned on test.

**Why this phase exists.** Phase 5 showed that a small run moves the held-out-template slice. It
did not show that the full recipe produces a model worth releasing. This phase adds the two
remaining synthetic workflows, grows the data to about 10k training decisions, trains at 4,096
tokens overnight, refits calibration, and measures once on test. The measurement is written down
before you run it, so the result is a claim and not a search.

**What you will understand after this phase**
- **Pre-registered metric.** You commit to the exact metric, slice, CI method and success
  threshold in git *before* you look at test numbers. If you pick them afterwards, you will
  (without meaning to) choose the slice where you happened to win. Out of 12 slices, one can clear
  a 95% CI by luck alone. The commit hash inside `metrics.json` proves the order of events.
- **Temperature per question type.** Choice, Score and Noul are miscalibrated by different amounts
  (pngwn arm B needed T = 1.59 overall). Fitting one T per `f"{qtype}:{n_perms}"` on cal divides
  the logits so that 0.8 confidence means right 80% of the time. T changes confidence only, never
  the top-1 answer.
- **Selective answering and risk-coverage.** A calibrated model can decline its least-confident
  decisions. Sort test items by confidence, keep the top 90% (coverage 0.9), and measure accuracy
  on what you kept. If accuracy does not rise as coverage falls, the confidences carry no ranking
  information, even when ECE looks fine.

**Consumes (names produced by earlier phases; do not rename):** `assign_split`, `route_split` and
the salt in `configs/split.yaml` (frozen at Phase 3); the Phase 4 synthetic generator
driver in `dataset.py`, the `label_decision` cache in `label.py`, the Phase 4 audit sampler, CSV
format and `audit_agreement`; `train(cfg_path)`, which resumes from the newest checkpoint in
`out_dir` (Phase 5); `gates/test_numerics_gate.py`, which reads the adapter path from
`NANOHUNCH_ADAPTER` (Phase 5); and `run_eval`, `fit_temperature`, `paired_bootstrap`, `ece`,
`flip_rate`.

**Changes**
| File | Change |
|---|---|
| `configs/workflows/security_triage.yaml` | New. Workflow 2 (security alert triage): templates `t1` to `t4`. State fields: alert source, asset, indicators, log excerpt. Questions: Choice severity (5 options) and owning team (up to 8); Score confidence of compromise (0 to 9, ADR-0003 rule 3); Noul "needs escalation?". Each question has a generator-gold rule wherever the answer is a spec fact. `t4` uses different wording and field order and is held out. |
| `configs/workflows/invoice_match.yaml` | New. Workflow 3 (three-way match of PO, invoice and goods receipt): `t1` to `t4`. Questions: Choice match status (4 options) and discrepancy type (up to 6); Score payment risk (0 to 9); Noul "approve for payment?". `t4` is held out. |
| `configs/data_v2.yaml` | New. `split_config: configs/split.yaml` (the same file v1 reads; the salt is never copied). Targets: train 10,000 decisions (gold 4,000; synthetic 6,000 = 1,200 states x 5), cal 2,000, test 3,500. `held_out_templates` lists the workflow-1 template already held out in `configs/data_v1.yaml`, plus `security_triage/t4` and `invoice_match/t4`. `max_tokens: 4096` for training rows; test keeps longer states for the `gt4k` bucket. |
| `configs/label_v2.yaml` | New. The same two `TeacherSpec` entries as the Phase 4 label config, the same cache path, and input `data/built/v2/unlabelled.jsonl`. Decisions already in the cache cost nothing. |
| `configs/train_full.yaml` | New. Copy every key from the Phase 5 3k-run config. Change only `data_path: data/built/v2/train.jsonl`, `out_dir: runs/full_v2`, `n_decisions: null` (all of v2; the inherited 3000 would silently train on 3k), `max_seq: 4096`, `epochs: 1`, `batch_size: 1` (R9), `ckpt_minutes: 15`. Keep `select_by: last` (the only value Phase 5 accepts; a cal-NLL selector is a later-list item). Key names must match Phase 5's `TrainConfig` exactly, because `load_config` rejects unknown keys. |
| `configs/eval_final.yaml` | New. Full content in step 16. |
| `dataset.py` | No new core code: Phase 4's `route_split` already sends held-out templates to `"test_ood"` before `assign_split` runs. Glue: write `data/built/v2/manifest.json` in the v1 manifest format (same keys, same `dataset_version` rule). |
| `cli.py` | Glue: `build --config C [--decode N]` prints N decoded rows; `label --config C [--limit N]`; `train --config C [--dry-run]` prints total training tokens and estimated hours; `fit-cal --adapter PATH --split cal --n-perms P --out F`, where `PATH` may be the literal `none` for B0; `eval --config C [--split S] [--predictor NAME]`. |
| `calibrate.py` | **You write:** `bootstrap_ci` and `risk_coverage` (signatures below). |
| `evaluate.py` | Reuse Phase 4's `audit_agreement` unchanged. Glue: reliability diagram PNGs (15 bins, one per qtype plus overall), the length-bucket table, the risk-coverage table, and the pngwn comparison table. |
| `tests/test_dataset.py`, `tests/test_calibrate_metrics.py`, `tests/test_evaluate.py` | 7 new tests, listed in steps 3 and 10. |
| `tools/decider_predictor.py` | New, outside the line budget. `DeciderPredictor(model_dir: Path, device: str = "mps")` implementing the `Predictor` protocol by calling `refs/decider`'s own inference code on the downloaded `Mapika/decider-2b` snapshot (Apache-2.0, base `Qwen/Qwen3.5-2B-Base`, full fine-tune; the HF repo ships `decider/infer.py`). It maps our `Question` to its request schema and its distributions back to canonical option order. |
| `reports/final/preregistration.md` | New. Committed **before** step 17. |
| `reports/audit/v2_audit.csv` | New. 200 rows with your judgements. |
| `ledger/spend.csv`, `ledger/hours.csv` | Append this phase's rows. |

**Produces (interfaces later phases use)**
- `calibrate.bootstrap_ci(stat: Callable[..., float], *arrays: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]`:
  returns (point, lo, hi) at 95%, resampling item indices jointly across `arrays`.
- `calibrate.risk_coverage(conf: np.ndarray, correct: np.ndarray, coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5)) -> list[tuple[float, float, float]]`:
  returns (coverage, accuracy, confidence_threshold) per coverage.

- `data/built/v2/{train,cal,test}.jsonl` plus `manifest.json`; test rows have `split` in
  `{"test", "test_ood"}`. `runs/full_v2/adapter/`, `runs/full_v2/calibration.json`,
  `runs/b0/calibration_v2.json` (both serialized `Calibration`).
- `reports/final/metrics.json` with top-level keys `prereg_commit`, `created_at`,
  `model_revision`, `dataset_sha256`, `headline_excluded_types`, `primary`, `secondary`,
  `length_buckets`, `risk_coverage`, `pngwn`, `audit`. Also the PNGs
  `reports/final/reliability_{choice,score,noul,all}.png`.

**Steps**
1. Preconditions. The Milestone 2 report must record the Phase 5 kill rule as PASS. Then run:
   `uv run python -c "import csv;print(round(sum(float(r['usd']) for r in csv.DictReader(open('ledger/spend.csv'))),2))"`
   Expect a number well under 100 (about 5 to 10). At 100 or more, R16 halts paid work: skip
   step 7 and train on gold plus the labels you already have.
2. Write the two workflow YAMLs, then confirm that 5 states per template render:
   `uv run python cli.py build --config configs/data_v2.yaml --decode 5`.
3. Write the failing tests in `tests/test_dataset.py`:
   - `test_heldout_templates_only_in_test_ood`: asserts
     `{r["meta"]["template"] for r in train + cal} & set(cfg["held_out_templates"]) == set()` and
     that every held-out row has `split == "test_ood"`.
   - `test_v2_preserves_v1_splits`: both configs point at the same `split_config`, and for every
     `group_key` in the `data/built/v1/*.jsonl` rows (read the rows; the manifest has no
     per-group map) the v2 split equals the v1 split.
   - `test_state_id_disjoint_v2`: train, cal and test `state_id` sets are pairwise disjoint.
4. Run `uv run pytest tests/test_dataset.py -q` and expect the 3 new tests to fail or error:
   `data/built/v2` does not exist yet. They pass after step 5 builds it. The Phase 4 U1 property
   test (`target_perm[j] == target[perm[j]]`) must still pass on a v2 sample.
5. Build: `uv run python cli.py build --config configs/data_v2.yaml --decode 20`. Expect counts
   within 5% of 10,000 / 2,000 / 3,500, and a `test_ood` count above 0 for each of the 3
   workflows. The build drops train rows whose `norm_hash` is in `configs/eval_only_hashes.txt`
   and prints `dropped_eval_only` per source (R22), as in Phase 4. Rerun step 4's command; expect
   a pass.
6. Read all 20 decoded rows (R6). For each one, check that the option text next to the highest
   target probability is the right answer to its question. One inverted row stops the phase until
   you have found the cause.
7. Label. Run `uv run python cli.py label --config configs/label_v2.yaml --limit 100`. Sum the
   `usage` cost of the new cache rows and multiply by (new decisions / 100). If the projection is
   above 5 USD, cut synthetic states to 1,000 in `configs/data_v2.yaml`. Otherwise run again
   without `--limit`. Append a row to `ledger/spend.csv` and repeat the step 1 command, expecting
   about 3 USD more than before. Rebuild with the step 5 command to attach the soft labels.
8. Below-chance tell (R6), at T = 1, before `eval_final.yaml` exists:
   `uv run python cli.py eval --config configs/eval_m1.yaml --data data/built/v2/cal.jsonl --predictor b0 --calibration none --out-dir reports/final/cal_b0_check`.
   Every gold slice's accuracy must be above 1/n_options for its type. Below chance means
   inverted labels: stop.
9. Audit (R11, A1). Run the Phase 4 audit sampler with `n=200`, seed 1, at least 40 items per
   qtype, and the same oversampling of items where both teachers agree above 0.8. It writes
   `reports/audit/v2_audit.csv`. Record your judgement on every row without looking at the teacher
   labels (about 3 h). `audit_agreement` already exists (Phase 4) and is tested there. Run:
   `uv run python -c "from pathlib import Path;from evaluate import audit_agreement;print(audit_agreement(Path('reports/audit/v2_audit.csv')))"`
   Tripwire: any type below 0.75 goes into `headline_excluded_types` in the pre-registration.
10. Write the failing tests:
    - `tests/test_calibrate_metrics.py`: `test_risk_coverage_toy` (conf `[0.9, 0.8, 0.7, 0.6]`, correct
      `[1, 1, 0, 0]`: accuracy 1.0 at coverage 0.5, 0.5 at coverage 1.0);
      `test_bootstrap_ci_constant` (`np.mean` on all ones gives `point == lo == hi == 1.0`);
      `test_bootstrap_ci_seeded` (two seed-0 calls return equal tuples).
    - `tests/test_evaluate.py::test_length_bucket_edges`: 512 to `le512`; 513 and 2048 to
      `512_2k`; 2049 and 4096 to `2k_4k`; 4097 to `gt4k`.
    Implement, then run `uv run pytest tests/test_calibrate_metrics.py tests/test_evaluate.py -q` and expect
    a pass.
11. Estimate: `uv run python cli.py train --config configs/train_full.yaml --dry-run --tok-s <N>`.
    The design says about 14 h per 10k-decision epoch (DERIVED from synthetic-weight rates,
    `risks.md` R7). P0-8 measured only at 2k, so take `N` = the P0-8 soak median x 232 / 297 (the
    synthetic 4k/2k ratio). R7 names a run longer than one night (take 12 h) as the Modal trigger.
    If the estimate exceeds 12 h, first cut the long synthetic states or `max_seq`, because they
    dominate the token count; cutting gold rows saves minutes, not hours. If it still exceeds
    12 h, apply R7.
12. Launch in the evening:
    `mkdir -p runs/full_v2 && caffeinate -i uv run python cli.py train --config configs/train_full.yaml 2>&1 | tee -a runs/full_v2/train.log`
    Each morning, `tail -5 runs/full_v2/log.jsonl` should show a rising step count and a finite
    loss. After a crash or a sleep, run the same command again: it resumes from the newest
    15-minute checkpoint. Note every failed night in `ledger/hours.csv`.
13. One epoch, no second. A mid-run epoch change would alter `cfg_sha256` (resume raises
    `ResumeMismatch`), restart the cosine schedule, and need a cal-NLL checkpoint selector that is
    on the later list. If a second epoch is ever wanted, it is a new pre-registered run with
    `epochs: 2` from step 0, not an extension.
14. Numerics gate on the adapter:
    `NANOHUNCH_ADAPTER=runs/full_v2/adapter uv run pytest gates/test_numerics_gate.py -q`.
    Expect `4 passed`: per-item trainer-vs-engine NLL parity within 2e-2 on 64 cal decisions, each
    of 5 prompts differing between base and adapter, the CPU fp32 oracle within 1e-3, and overfit-32
    through the engine. A failure blocks steps 15 to 17 (R9).
15. Calibrate both predictors on the same cal split:
    `uv run python cli.py fit-cal --adapter runs/full_v2/adapter --split cal --data data/built/v2/cal.jsonl --n-perms 1 --out runs/full_v2/calibration.json`
    `uv run python cli.py fit-cal --adapter none --split cal --data data/built/v2/cal.jsonl --n-perms 1 --out runs/b0/calibration_v2.json`
    Expect keys `choice:1`, `score:1` and `noul:1`, each with T strictly inside (0.05, 20). A T
    that lands on a search bound is a bug: fix it before continuing (`fit-cal` exits 1 on it).
    Re-check the Phase 3 calibration-grouping rule here: v2 has Choice items with 6 to 8 options.
    If the Phase 3 R13 tripwire fired, also fit `choice:2` for both predictors with `--n-perms 2`.
16. Write `configs/eval_final.yaml`:

    ```yaml
    data: data/built/v2
    split: test                     # test.jsonl, including its split: test_ood rows
    predictors:
      trained: {adapter: runs/full_v2/adapter, calibration: runs/full_v2/calibration.json, n_perms: 1}
      B0:      {adapter: null, calibration: runs/b0/calibration_v2.json, n_perms: 1}
      # only if the Phase 3 R13 tripwire fired, pre-registered here so it is not a second look:
      # trained_p2: {adapter: runs/full_v2/adapter, calibration: runs/full_v2/calibration.json, n_perms: 2}
    baseline: B0
    slices:
      gold:             {label_origin: gold}                  # PRIMARY (a)
      heldout_template: {split: test_ood, label_origin: spec} # PRIMARY (b), gold accuracy
      heldout_judgement: {split: test_ood, label_origin: teacher}
    flip_suite: [reverse, perm_seed_1, perm_seed_2, perm_seed_3]   # Choice items, >= 3 options
    length_edges: [512, 2048, 4096]                                 # le512, 512_2k, 2k_4k, gt4k
    coverages: [1.0, 0.9, 0.8, 0.7, 0.5]
    bootstrap: {n: 10000, seed: 0, groups: state_id}
    ece_bins: 15
    external:
      pngwn_test: {path: data/raw/pngwn/typed-decisions-v2, split: test, cal_split: cal, publish: aggregate_only}
    out_dir: reports/final
    ```

    16a. (3 h) External trained baseline. `hf download Mapika/decider-2b --revision <sha printed by hf>`; record the sha in `configs/eval_final.yaml` as `external.decider_2b: {model_dir: ..., revision: ..., calibration: "own"}`. Write `tools/decider_predictor.py` (glue). Smoke test on 20 test items: every returned distribution sums to 1 within 1e-5 and has the right length. decider uses its own temperature (reported T = 1.30 for v10); report it as shipped **and** refit on our cal split, and say which is which. Add `decider_2b` under `predictors:` with `baseline: B0`. Also add `external.jevbench_public` and `external.semif_authored` (the Phase 3 converted files) so every predictor is scored on them in the same run.

    Then write `reports/final/preregistration.md`. **PRIMARY:** trained minus B0 gold accuracy on
    (a) the `gold` slice and (b) the `heldout_template` slice, paired bootstrap 95% CI resampled by
    `state_id`; success is CI lo > 0 on both, +3 pts is the S1 target. If (a) fails and (b) passes,
    the report says so and the release is still the adapter, framed as a held-out-template gain
    only. **Earlier looks at test:** M1 and M2, listed with their commits. **SECONDARY:** consensus agreement, ECE (width and mass, CI
    via `bootstrap_ci`), NLL, Brier, flip rate and top-1 agreement on `flip_suite`, length-bucket
    accuracy vs B0, risk-coverage. **EXTERNAL:** pngwn test pass vs arm B; decider-2b on our test set, SemIf authored144 and JevBench public items (self-run, not official); the untrained-Qwen3.5-4B row from M1 as the SemIf-method reference. **Excluded types:**
    from step 9. **Decision rules:** the kill criterion and tripwires below, verbatim. Commit with
    `git add configs/eval_final.yaml reports/final/preregistration.md && git commit -m "phase6: pre-register final eval"`.
17. Run the final eval once, overnight:
    `caffeinate -i uv run python cli.py eval --config configs/eval_final.yaml`.
    It writes `reports/final/metrics.json` with `prereg_commit` set to the output of
    `git log -1 --format=%H -- reports/final/preregistration.md`, plus the tables and PNGs. On the
    pngwn test set it reports two ECE figures: one with our cal T, and one with T refit on pngwn
    cal (arm B's 0.015 used pngwn cal). It stores aggregates only (SEC-2).
18. Fill the `pngwn` block and this table in the report draft:

    | | pngwn arm B (0.6B, published) | ours B0 | ours trained |
    |---|---|---|---|
    | accuracy, pngwn test (5,214 decisions) | 0.752 | from run | from run |
    | ECE after T-scaling | 0.015 | from run | from run |
    | reversed-order flips / top-1 agreement | 37.5% / 0.664 | from run | from run |
    | above 2k tokens vs own base | fine-tune lost to base | `2k_4k` and `gt4k` delta vs B0 | same |

    And the open-replication table (all rows run by us in one harness except where marked):

    | | SemIf method (frozen Qwen3.5-4B, M1) | decider-2b (trained 2B, external) | ours B0 (MiniCPM5-2B) | ours trained |
    |---|---|---|---|---|
    | our test, gold accuracy / ECE | from M1 | from run | from run | from run |
    | our test_ood, gold accuracy | from M1 | from run | from run | from run |
    | SemIf authored144, family balanced acc. | 0.813 published, ours from M1 | from run | from run | from run |
    | JevBench public items, accuracy / hard ECE | from run | from run | from run | from run |
    | Mac W1 latency (1k state, 16 q) | from M1 | from run | from run | from run |
    | core lines of code | n/a | n/a | n/a | `tools/loc.py` total |

19. Apply the tripwires below. Then run
    `uv run ruff format . && uv run ruff check --fix . && uv run pytest -q` and commit with
    `phase6: data v2, full run, final eval`.

**Verification gate**
- `uv run pytest -q` prints `56 passed` (49 before plus 7 new).
  `NANOHUNCH_ADAPTER=runs/full_v2/adapter uv run pytest gates/test_numerics_gate.py -q` prints
  `4 passed`.
- `uv run python tools/loc.py` exits 0 (R23: every gate from Phase 2 on; `bootstrap_ci` and
  `risk_coverage` land in `calibrate.py` here).
- This prints `True`:
  `uv run python -c "import json,subprocess;m=json.load(open('reports/final/metrics.json'));h=subprocess.check_output(['git','log','-1','--format=%H','--','reports/final/preregistration.md'],text=True).strip();print(m['prereg_commit']==h)"`
- `metrics.json` has non-empty `primary.gold` and `primary.heldout_template` (each with `delta`,
  `lo`, `hi`, `n`), 4 length buckets, 5 risk-coverage rows and a `pngwn` block.
- The step 1 spend command prints a number under 100.

**Rollback**
- Nothing public changes in this phase. `git revert` the phase commits. `data/built/v1/` and the
  Phase 5 adapter stay untouched, because v2 writes only to `data/built/v2/`, `runs/full_v2/`,
  `runs/b0/calibration_v2.json` and `reports/final/`.
- Point of no return: step 17 spends the test split. If you change code after it, publish any
  re-run as a second look next to the first number, never in its place.
- Data written during a failed window: the label cache is append-only and a retry reuses it, so
  no paid call repeats. An interrupted training run resumes from its last checkpoint. If
  `runs/full_v2/` is corrupt, delete it and restart with the step 12 command (you lose one night).

**Kill criterion**
- A5: if PRIMARY (b), the `test_ood` delta over B0, has CI lo <= 0 after the full run (one
  epoch, resumed across nights if needed), then the adapter is not the product. Phase 7 releases the B0 path
  instead: the base model plus `runs/b0/calibration_v2.json`, `FORMAT.md` and `cli.py decide`,
  with the report stating the negative result and its numbers.
- R18: two failed overnight runs trigger the Modal later-list item (one function with
  `timeout=`). Calendar (R17): if no model trained on teacher-labelled data is evaluated by the
  end of week 9, publish B0 plus a training write-up and stop.

**Tripwires (pre-committed)**
- Trained top-1 order agreement below 0.90 on `flip_suite`, or flip rate above 15% (A12, R13,
  ADR-0002): implement P = 2 reversed pooling for Choice at inference
  (`fmt.permutations_for(n, 2)` with the reversed perm, pooled by `engine.pool`), fit T for
  `choice:2`, and set `default_perms["choice"] = 2`. This takes about 6 h out of the week-10
  slack. Unless `trained_p2` was pre-registered in step 16, its test numbers are a labelled second
  look, reported next to P = 1, never in its place.
- The `2k_4k` or `gt4k` bucket falls more than 1 pt below B0 (A13, S4): report it as a finding in
  the headline table and make it the first "what next" item. Do not drop the bucket.
- Audit agreement below 0.75 on a type: that type leaves the headline (listed in step 16).
- decider-2b beats ours trained on PRIMARY (a) by more than 3 pts: report it plainly; the
  headline becomes "the minimal, Mac-trainable version", with the accuracy gap and its cost
  (decider: 1.47M examples, GH200 hours, 11.9k lines; ours: about 10k decisions, one Mac,
  at most 1,000 core lines). Do not re-scope the release to chase it.

### Phase 7: Release the model card, final report and artefacts on Hugging Face

**Goal:** public HF repo `${HF_USER}/nanohunch-minicpm5-2b` holds exactly the manifest's files,
`reports/final/README.md` is committed, and tag `v0.1.0` points at the commit that produced both.
**Effort:** 2.0 engineer-days (12 h: `release.py` plus tests 3, card 3, report 3, stage plus
private upload plus clean-download check 2, flip plus tag 1). **Depends on:** Phase 6.
**Parallel with:** report drafting during Phase 6's final-eval night (1 engineer, interleaved).
**Risk:** medium; this is the one-way door, and a public licence or secret mistake cannot be recalled.

**Why this phase exists.** A model nobody can load, or can load but not trust, is not a portfolio
artefact. This phase packages the adapter, the calibration and the format so a stranger can run
`decide` and reproduce the gold-slice numbers without API keys. It also puts every release
decision behind a fail-closed gate, because a public repo cannot be unpublished.

**What you will understand after this phase**
- **What a model card must say about calibration.** Say where T was fitted (cal, never test),
  give T per type, give ECE with its CI under both binning schemes, and include the reliability
  diagrams. Say that calibration holds only in-distribution: under a new template or long inputs
  ECE can differ, and the card shows the per-bucket numbers.
- **What a model card must say about limits.** The tested envelope (26 options, 10 Score levels,
  trained at 4,096 tokens, evaluated in `gt4k`), latency per workload on stated hardware, and what
  the model is not. Every limit you leave out comes back as a user's bug report.
- **Licence flow-down.** The weights carry obligations from the base model and from the teachers.
  From the base (Apache-2.0): a LICENSE copy and a NOTICE of changes (SEC-17). From DeepSeek (ToS
  4.2(3), as in R4 and Q6): published outputs are marked AI-generated. NC data (ANLI, pngwn) appears in no training
  file and no released file, and pngwn appears only as aggregate metrics.

**Changes**
| File | Change |
|---|---|
| `release.py` | New. Owns staging, gates and upload. **You write** `assert_release_rows`; the rest is glue. Secret regexes are assembled by concatenation (`"sk" + "-or-"`, `"hf" + "_"`) so the step 5 `git grep` does not match this file. |
| `configs/release_v0.1.0.yaml` | New. `repo_id: ${HF_USER}/nanohunch-minicpm5-2b` (no "DeepSeek" in the name, SEC-20). `artefact: adapter`, or `b0` if the Phase 6 kill fired. An explicit `files:` list of (source path, path-in-repo) pairs. `training_manifests: [data/built/v2/train.jsonl, data/built/v2/cal.jsonl]`. |
| `release_assets/model_card.tmpl.md` | New. A `string.Template` whose `$name` fields are filled from `reports/final/metrics.json`. Sections in step 4. |
| `release_assets/LICENSE` | New. Full Apache-2.0 text. |
| `release_assets/NOTICE.md` | New. Base repo id and revision sha. Changes: "LoRA adapter trained; restricted label-logit readout; calibration.json added". |
| `release_assets/FORMAT.md` | New. The nanohunch-fmt-v1 spec from ADR-0003 including Amendment 1: prompt layout; label vocab per type (letters A to Z for up to 26 options, Score values 0 to 9 with the amended Score labels, yes/no); the option-permutation rule; `max_context` and truncation behaviour. |
| `cli.py` | Glue: `release build`, `release check` and `release upload`, each taking `--config C`; `upload` also accepts `--dry-run`. |
| `.gitignore` | Add `release/`. |
| `tests/test_release.py` | New. 7 tests in step 1. Fake keys are built by concatenation (`"sk" + "-or-" + "x" * 40`) so the repo grep does not match this file. |
| `reports/final/README.md` | New. The final report (step 6). |

**Produces (interfaces later phases use)**
- `release.assert_release_rows(rows: Iterable[dict]) -> int` returns the number of rows checked.
  For each row it calls `dataset.assert_row_releasable` (licence allowlist, per-teacher
  `teacher_id` present and in `RELEASABLE_TEACHERS` for teacher-origin decisions; gold rows with
  `teachers: []` pass). It adds two checks: `source` in the deny-list `{facebook/anli, pngwn/typed-decisions, pngwn/typed-decisions-v2, pngwn/typed-decisions-causal-experiment, pngwn/system-one-qwen3.5-4b-scorer}`,
  and any state or question whose `norm_hash` is in `configs/eval_only_hashes.txt` (R22). It
  imports the constants from Phase 4's module and does not redefine them.
- `release.scan_secrets(root: Path) -> None` runs the regexes `sk-or-[A-Za-z0-9_-]{20,}`,
  `hf_[A-Za-z0-9]{20,}`, `OPENROUTER_API_KEY=.+` and `Bearer\s+\S+` over every file under `root`,
  and raises `ValueError(path, pattern)` on the first match.
- `release.build(cfg_path: Path) -> Path` copies only the `files:` entries into
  `release/v0.1.0/`, renders `README.md` from the template (with `Template.substitute`, so a
  missing key raises `KeyError`), and writes `SHA256SUMS` over every staged file.
- `release.upload_commands(stage: Path, repo_id: str) -> list[list[str]]` returns one
  `["hf", "upload", repo_id, <file>, <path-in-repo>]` per staged file and never passes a directory.
- HF repo layout (12 files, plus the `.gitattributes` HF adds): `README.md`, `adapter/adapters.safetensors`,
  `adapter/adapter_config.json`, `calibration.json`, `FORMAT.md`, `LICENSE`, `NOTICE.md`,
  `SHA256SUMS`, `plots/reliability_{choice,score,noul,all}.png`. The `b0` artefact omits the 2
  adapter files.

**Steps**
1. Write the failing tests in `tests/test_release.py`:
   - Each of these raises `ValueError`: `test_rejects_denylisted_source` (source `facebook/anli`),
     `test_rejects_missing_teacher_id` (a teacher row without `teacher_id`),
     `test_rejects_unlisted_teacher` (a `teacher_id` naming a closed model).
   - `test_accepts_gold_row` returns 1. `test_scan_secrets_finds_fake_key` raises on a temp file
     holding a concatenated fake key. `test_template_missing_key_raises` raises `KeyError`.
   - `test_upload_commands_are_per_file`: every command has 5 elements, none equal to the stage
     directory or `.`.
   Run `uv run pytest tests/test_release.py -q` and expect 7 failures. Implement, re-run, and
   expect 7 passed.
2. Licence gate over every row that shaped the released weights and calibration:
   `uv run python -c "import json,release;print(release.assert_release_rows(json.loads(l) for p in ['data/built/v2/train.jsonl','data/built/v2/cal.jsonl'] for l in open(p)))"`
   Expect about 12,000 and no traceback. Then read `data/built/v2/manifest.json` and confirm that
   no MMLU-Pro or pngwn source appears in train counts (SEC-14).
3. Optional fused weights, only if the adapter loads in stock mlx_lm:
   `uv run python -c "import yaml;from mlx_lm import load;c=yaml.safe_load(open('configs/train_full.yaml'));load(c['model_path'],adapter_path='runs/full_v2/adapter')"`.
   If that loads cleanly, check flag names with `uv run python -m mlx_lm.fuse --help` and fuse into
   `runs/full_v2/fused/`. If it fails (the Phase 5 hand-written adapter format differs from
   mlx_lm's), skip fusing and say so in the card.
4. Fill `release_assets/model_card.tmpl.md`. Each section is backed by numbers from
   `metrics.json`:
   - **Intended use:** typed decisions (Choice, Score, Noul) over a text state, returning
     probabilities, human in the loop for consequential actions; state text is untrusted input.
   - **Format and limits:** nanohunch-fmt-v1, at most 26 options, 10 Score levels, trained at up to
     4,096 tokens, tested up to the longest `gt4k` state.
   - **Calibration evidence:** T per type fitted on cal, ECE (width and mass) with CIs, the four
     reliability PNGs, the risk-coverage table.
   - **Evaluation:** PRIMARY vs B0 with CIs, length buckets, flip rate, the Phase 6 step 18
     pngwn table, audit agreement, excluded types.
   - **What it is NOT:** not Jev parity. Latency per workload W0/W1/W2 on the M5 with model size
     (Phase 2 measured values; the design's synthetic W1 was 893 ms). Knowledge limited by a 2B base.
   - **Data provenance:** each source with revision and SPDX licence; "labels generated by AI
     models (DeepSeek V4.1 Flash via the DeepSeek API; Qwen3.6-35B-A3B open weights)"; no
     closed-model output used for training, selection, calibration or publication (ADR-0004);
     ANLI and pngwn eval-only, aggregates only.
   - **Licence and integrity:** Apache-2.0 from MiniCPM5-2B-Base with LICENSE and NOTICE; the
     SHA-256 of each safetensors file (SEC-10).
5. Stage and gate:
   `uv run python cli.py release build --config configs/release_v0.1.0.yaml`
   `uv run python cli.py release check --config configs/release_v0.1.0.yaml`
   `check` runs `scan_secrets` on `release/v0.1.0` and then
   `trufflehog filesystem release/v0.1.0 --results=verified,unknown --fail` (confirm the flags with
   `trufflehog filesystem --help`). Any non-zero exit aborts. Then run:
   `git grep -nE "sk-or-v1-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}|OPENROUTER_API_KEY=.+" -- . ':!docs/design'; echo "exit=$?"`
   Expect no match lines, then `exit=1`. This is the AGENTS.md pre-push pattern plus the env-var
   form. The key-shaped tails keep it from matching files that quote it; only the design docs
   are excluded, because they quote the `OPENROUTER_API_KEY=.+` form.
6. Write `reports/final/README.md`, framed as **the minimal open System One model** and measured
   against pngwn arm B, SemIf and decider-2b:
   **TL;DR** (PRIMARY numbers with CIs, 3 lines, plus the core line count and "trains on one
   M5 Mac overnight"); **Methodology** (data v2 composition, frozen
   split, teachers, 4,096-token training, T fit); **Pre-registered metrics** (link to the commit);
   **Results** (B0 vs trained, then arm B on order flips, above-2k buckets, T-scaled ECE, then
   the open-replication table from Phase 6 step 18);
   **Failures** (tripwires fired, second looks, excluded types); **What next** (the "Not doing"
   later list, in trigger order); **Footnote**: Jev is closed and was not run (R14), so there is
   no head-to-head claim.
   Commit with `phase7: model card, final report, release tooling`.
7. Upload to a **private** repo first. Create it private in the HF web UI (New model, visibility
   Private), or with `hf repo create` after checking `hf repo create --help`. Run
   `uv run python cli.py release upload --config configs/release_v0.1.0.yaml --dry-run`, read the
   printed per-file `hf upload` lines, then run the same command without `--dry-run`.
8. Clean-download check (S12). In an empty `/tmp/nanohunch-verify`, with `OPENROUTER_API_KEY` and
   `HF_TOKEN` unset, run `hf download ${HF_USER}/nanohunch-minicpm5-2b --local-dir .` and then
   `shasum -a 256 -c SHA256SUMS`, expecting `OK` on every line. From a fresh clone of the code
   plus a local copy of `data/built/v2/test.jsonl` (gitignored and not released), point
   `configs/eval_final.yaml` at the downloaded adapter **and** the downloaded `calibration.json`,
   and run `uv run python cli.py eval --config configs/eval_final.yaml --split test --predictor trained`.
   Score only the gold and spec-gold decisions (consensus metrics need labels that are not
   released, and the card says so). Expect accuracy and ECE within +/- 0.002 of the same subset
   recomputed from `reports/final/items.jsonl`, in under 2 h.
9. Flip the repo to public (Settings, Change visibility, Public). Run the AGENTS.md pre-push
   check (`git grep -nE "sk-or-v1-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}"`, expect no output). Then run
   `git tag -a v0.1.0 -m "nanohunch v0.1.0: adapter, calibration, format v1" && git push origin v0.1.0`.
10. Optional if hours remain (later list, not in effort): local Gradio demo `demo.py`, not deployed.

**Verification gate**
- `uv run pytest tests/test_release.py -q` reports 7 passed, and `uv run pytest -q` prints
  `63 passed`; `uv run python tools/loc.py` exits 0 (R23).
- The step 5 `git grep` prints only `exit=1`, and `release check` exits 0.
- The HF repo's file list (the web Files tab) matches the manifest exactly: 12 files, or 10 for
  `b0`, plus `.gitattributes`, plus `fused/` only if step 3 ran.
- Step 8 reproduces within +/- 0.002, with every `shasum` line `OK`.

**Rollback**
- Before step 9: delete the private repo in HF Settings, fix, re-run (uploads overwrite by path).
  Point of no return: step 9, the public flip; after it, others may already hold copies.
- After the flip: for a secret, revoke the OpenRouter key and HF token within minutes, delete the
  repo, and re-release as v0.1.1 from a clean stage. For a licence problem, set the repo private,
  remove the artefact, and publish a notice in the report. For a wrong number, push a corrected
  card with an erratum line and tag v0.1.1.
- Data written during a failed window: a partial private upload dies with the deleted repo.

**Kill criterion**
- A8 / Q6: if excluding rows cannot make the licence gate pass within 2 h, do not release the
  adapter. Example: DeepSeek V4.1 Flash turns out not to be covered by ToS 4.2(3), which makes
  every DeepSeek-labelled row non-releasable. In that case release `artefact: b0` (base plus B0
  calibration, `FORMAT.md` and the report), and make "retrain on Qwen3.6-only labels" the first
  "what next" item.
