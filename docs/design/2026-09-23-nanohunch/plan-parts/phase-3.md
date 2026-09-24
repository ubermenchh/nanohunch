### Phase 3: Eval harness, B0 bake-off and external anchors: Milestone 1

**Goal:** `uv run python cli.py eval --config configs/eval_m1.yaml --split test` writes
`reports/<run>/metrics.json` and `README.md` for any predictor, and `reports/m1/README.md` (B0 of
MiniCPM5-2B-Base vs Qwen3-4B-Base, framed against pngwn arm B) is public on GitHub.
**Effort:** 23 h = 3.8 engineer-days (1 engineer, week 4; step 9a adds 3 h for external anchors). **Depends on:** Phase 2. **Parallel
with:** Phase 4 non-core glue (public downloads), interleaved evenings only.
**Risk:** medium, because the split salt becomes a **ONE-WAY DOOR when M1 test numbers are
published**: after that, changing it silently reshuffles the test set and makes every earlier
number incomparable.

**Why this phase exists**
Every later claim ("the fine-tune beats B0") is a difference between two eval runs. If the harness
is wrong, or the split leaks, or gold and teacher agreement are mixed, Phases 5 to 7 measure
noise. Building the harness on B0 first, with no training in the loop, also answers Q4 (which base
to train) and the R13 tripwire (is order sensitivity bad enough to need pooling) before any
teacher money is spent.

**What you will understand after this phase**
- **ECE (expected calibration error).** Sort answers into 15 confidence bins. In each bin compare
  average confidence with actual accuracy. Example: 100 answers with confidence near 0.9, of which
  80 are correct, have a gap of |0.80 - 0.90| = 0.10. ECE is the mean of those gaps weighted by
  bin size: with 1,000 answers, that bin contributes 100/1000 x 0.10 = 0.01. Equal-width bins
  (`width`) split [0, 1] into 15 slices; equal-mass bins (`mass`) put the same number of answers
  in each, which stops a nearly empty bin from dominating. Report both.
- **Temperature scaling.** Divide logits by one number T per question type, fit on `cal`, judged
  on `test`. It never changes the argmax, so accuracy is unchanged and only calibration moves.
- **Paired bootstrap.** Two models answer the same 3,500 items. Resample item indices with
  replacement 10,000 times, and for each resample compute acc(A) - acc(B) **on the same indices
  for both models**. The 2.5th and 97.5th percentiles are the 95% CI of the difference. Pairing
  cancels the "this resample happened to draw easy items" noise. Example: A is right on
  2,730/3,500 (0.780), B on 2,660/3,500 (0.760), delta 0.020; if the CI is [0.008, 0.032] the gap
  is real, if it is [-0.004, 0.044] it is not.
- **Gold vs consensus accuracy.** Gold is the human answer key; consensus is the argmax of the
  teacher models' averaged distribution. If teachers agree with gold 85% of the time, a student
  that copies the teachers perfectly scores 1.00 vs consensus and 0.85 vs gold. Mixing the two
  hides whether the student learned the task or learned the teachers' mistakes, so they are always
  separate columns. In M1 there are no teachers, so consensus is `null` for public data and filled
  only where pngwn rows carry a distribution.
- **Order flip rate.** Show the same Choice options in reversed order; if the top-1 option (mapped
  back to canonical order) changes, that is a flip. pngwn arm B flips 37.5% of the time.

**Read before writing (30 min)**
- `refs/minisystemone/trainer/calibrate_temperature.py`: temperatures per primitive and per candidate count, fitted on a separate split. You start with one T per qtype; step 7 measures whether option count needs its own T.
- `refs/semif/benchmarks/calibrate.py`: group-disjoint cross-validated temperature, and honest reporting of which ECE gains fall outside their interval.

**Changes**
| File | Change |
|---|---|
| `calibrate.py` | Extend (core). **You write** `accuracy`, `ece`, `nll`, `brier`, `flip_rate`, `paired_bootstrap` (registry signatures, algorithms in step 3), appended below the Phase 2 temperature code. |
| `dataset.py` | New, core. **You write** `assign_split` and `write_items(path: Path, items: Sequence[EvalItem]) -> str` (returns sha256). |
| `sources/public.py`, `sources/external.py` | New, outside the line budget (glue, like nanochat's dataset scripts). `public.py`: `build_eval_v1(cfg: dict) -> dict[str, int]`, the four public adapters, HotpotQA padding, stored permutations. `external.py`: `convert_pngwn(row: dict, fields: dict[str, str]) -> EvalItem | None`, plus `convert_semif(row) -> EvalItem | None` and `convert_jevbench(row) -> EvalItem | None` (step 9a). |
| `evaluate.py` | New. Registry `EvalItem`, `Predictor`, `run_eval`. **You write** the per-item scoring loop, flip remapping and bootstrap alignment inside `run_eval`. Glue: `load_items(path: Path) -> list[EvalItem]` (reads both the EvalItem row schema and the Phase 1 `data/skeleton/gold_*.jsonl` schema), `length_bucket(n_tokens: int, edges: Sequence[int]) -> str`, `B0Predictor`, `AdapterPredictor`, `_risk_coverage_table`, reliability PNGs, `README.md` rendering. |
| `cli.py` | Replace the Phase 2 `eval` body with `evaluate.run_eval`; add `fit-cal` and `build` (step 8). |
| `bench/latency_p2.py` | Add `--model` (default unchanged) and `--out` (default `reports/phase2.md`) flags. |
| `configs/split.yaml`, `configs/eval_data_v1.yaml`, `configs/eval_m1.yaml` | New, contents in steps 1, 5 and 8. |
| `tests/test_calibrate_metrics.py`, `tests/test_split.py`, `tests/test_dataset.py`, `tests/test_evaluate.py` | New, 10 tests total. |
| `reports/m1/` | New outputs: one directory per run, `latency.md`, `README.md`. |
| `ledger/hours.csv` | Append this phase's rows. |

**Produces (interfaces later phases use)**
- Registry functions of `calibrate.py`, `evaluate.py` and `dataset.assign_split`, unchanged
  names.
- `evaluate.length_bucket(n, (512, 2048, 4096))` returns `le512` (n <= 512), `512_2k` (<= 2048),
  `2k_4k` (<= 4096), `gt4k`. Phase 6 adds `test_length_bucket_edges` for it.
- `evaluate.B0Predictor(model: str, calibration: Calibration | None)` and
  `evaluate.AdapterPredictor(model: str, adapter: str, calibration: Calibration | None)`, both
  satisfying `Predictor` (`name` is `"b0"` or `"adapter"`), both built on `render`,
  `MLXBranchScorer.score`, `pool`, `calibrate.apply`, `to_answer`.
- `evaluate._risk_coverage_table(conf, correct, coverages) -> list[tuple[float, float, float]]`
  (coverage, accuracy, threshold). Phase 6 promotes it to `calibrate.risk_coverage` with the same
  return type.
- Every run writes `reports/<run>/items.jsonl` (one row per item: `state_id`, `question_id`,
  `qtype`, `source`, `length_bucket`, `gold`, `probs`, `correct_gold`, `correct_consensus`,
  `flip_top1` by suite entry), `metrics.json` (keys `run`, `created_at`, `git_commit`,
  `predictor`, `model`, `model_revision`, `adapter`, `calibration`, `format_version`, `split`,
  `split_salt`, `data_sha256`, `n_items`, `by_type`, `by_source`, `overall`, `flip`,
  `length_buckets`, `risk_coverage`, `baseline`), `README.md`, and `reliability_<qtype>.png` plus
  `reliability_all.png`.
- CLI: `cli.py eval --config C [--split S] [--predictor {b0,adapter}] [--model M] [--adapter A]
  [--calibration F] [--data D] [--out-dir O] [--baseline RUN]`. Flags override config keys.
  `--predictor` is case-insensitive and accepts `trained` as an alias of `adapter` (Phase 6 and 7
  write `B0` and `trained`). `cli.py fit-cal --adapter PATH|none --split cal --n-perms P --out F
  [--model M] [--config C]`. `cli.py build --config C [--decode N]`.
- `configs/split.yaml` (frozen salt) and `data/built/eval_v1/{cal,test}.jsonl` with
  `manifest.json` (`split_salt`, `counts` per split and source, `sha256` per file). Phase 4 reuses
  the salt; public-source groups assigned `cal`/`test` here are never trained on.

**Steps** (hours in brackets; tests first; you write the bodies marked "you write")
1. (0.25) Generate the salt once: `uv run python -c "import
   secrets;print('nanohunch-v1-'+secrets.token_hex(16))"`. Write `configs/split.yaml`:
   `salt: <printed value>` and `fractions: {train: 0.80, cal: 0.08, test: 0.12}` (order matters,
   see step 3). Commit it now so the salt has a timestamp before any test number exists.
2. (0.5) Write `tests/test_split.py`, all with `salt="test-salt"` and `fractions={"train": 0.8,
   "cal": 0.08, "test": 0.12}`:
   - `test_assign_split_deterministic`: 1,000 keys called twice give identical lists, and the pins
     hold: `("g0","boolq")` is `train`, `("g28","boolq")` is `test`, `("g35","boolq")` is `cal`,
     `("g35","arc")` is `test` (the source is part of the hash).
   - `test_no_group_in_two_splits`: 10,000 items over 2,000 group keys (5 items each, varied item
     ids): every group maps to exactly one split.
   - `test_fractions_within_1pct_at_10k`: keys `f"g{i}"` for i < 10,000: each split's share is
     within 0.01 of its fraction.
   Run `uv run pytest tests/test_split.py -q`; expect `ImportError: cannot import name
   'assign_split' from 'dataset'`.
3. (1.0) You write `assign_split`: (1) raise `ValueError(f"split fractions sum to {s}, expected
   1.0")` if `abs(sum - 1) > 1e-9`; (2) `h = sha256(f"{salt}|{source}|{group_key}".encode())`; (3)
   `u = int.from_bytes(h.digest()[:8], "big") / 2**64`; (4) walk `fractions` in insertion order
   accumulating; return the first name with `u < cumulative`; (5) if rounding leaves `u` past the
   end, return the last name. Rerun step 2's command; expect `3 passed`.
4. (3.5) Write `tests/test_calibrate_metrics.py`, then run `uv run pytest tests/test_calibrate_metrics.py -q` and
   expect 5 import errors:
   - `test_ece_calibrated_sampler_near_zero`: `rng = np.random.default_rng(0)`, `conf =
     rng.uniform(0.5, 1, 100_000)`, `correct = rng.random(100_000) < conf`; both `ece(...,
     scheme="width")` and `scheme="mass"` are `< 0.01`.
   - `test_ece_conf_one_in_last_bin`: `ece(np.array([1.0]), np.array([1]))== 0.0` and
     `ece(np.array([1.0]), np.array([0]))==1.0`, no `IndexError`.
   - `test_temperature_recovery_end_to_end`: seed 0, 20,000 items, 4 classes, true logits `z ~
     N(0, 2)`, targets sampled from `softmax(z)`, model logits `2*z`.
     `fit_temperature(log_softmax(2*z), targets)` is in `[1.8, 2.2]`, and ECE (width) of the top-1
     after `log_softmax(2*z/T)` is `< 0.02` and lower than before.
   - `test_bootstrap_self_zero`: `a = np.random.default_rng(1).integers(0, 2, 500)`;
     `paired_bootstrap(a, a) == (0.0, 0.0, 0.0)`.
   - `test_flip_rate_identity_zero`: `flip_rate([0,1,2,1],[0,1,2,1]) == 0.0` and
     `flip_rate([0,1],[1,1]) == 0.5`.
   You write, as numbered algorithms: **ece** (1) width: bin index `min(floor(conf*bins), bins-1)`
   so conf 1.0 lands in the last bin; mass: `np.argsort(conf)` then `np.array_split` into `bins`
   chunks; (2) per non-empty bin `gap = |mean(correct) - mean(conf)|`; (3) return `sum(n_b / N *
   gap)`. **nll**: mean of `-log(max(p[target], 1e-12))`. **brier**: mean over items of `sum_k
   (p_k - onehot_k)^2`. **accuracy**: mean of `argmax(p) == target`. **flip_rate**: share of
   positions where the two top-1 lists differ; raise `ValueError` on unequal lengths.
   **paired_bootstrap**: (1) `delta = a.mean() - b.mean()`; (2) `rng =
   np.random.default_rng(seed)`, draw `idx` of shape `(n, len(a))`; (3) `d = a[idx].mean(1) -
   b[idx].mean(1)`; (4) return `(delta, percentile(d, 2.5), percentile(d, 97.5))` as Python
   floats. Rerun; expect `5 passed`.
5. (3.0) Eval data, glue. Write `configs/eval_data_v1.yaml`: `split_config: configs/split.yaml`,
   `out: data/built/eval_v1`, `reference_tokenizer: openbmb/MiniCPM5-2B-Base`, and per split and
   source quotas: `cal: {boolq: 550, arc: 550, csqa: 550, hotpot: 350}` (2,000) and `test: {boolq:
   900, arc: 900, csqa: 900, hotpot: 350, hotpot_pad: 450}` (3,500). Adapters, all public gold, no
   teachers (download with `hf download <id> --repo-type dataset --local-dir data/raw/<name>`):
   - BoolQ `google/boolq` train+validation: state = passage, `noul` question, options
     `("yes","no")`, gold 0 = yes, `group_key = sha256(passage)[:16]`, `source = "boolq"`.
   - ARC `allenai/ai2_arc` Easy+Challenge: state = question stem, `choice` question "Which option
     answers the question in the STATE?", options = choices, `group_key` = ARC id, `source =
     "arc"`.
   - CommonsenseQA `tau/commonsense_qa` train+validation: same shape as ARC, 5 options, `source =
     "csqa"`.
   - HotpotQA `hotpotqa/hotpot_qa` config `distractor`, train+validation rows with answer `yes` or
     `no`: state = the 10 context paragraphs joined as `"Title\nText\n\n"`, `noul` question,
     `group_key` = HotpotQA `_id`, `source = "hotpot"`.
   - `hotpot_pad`: for 150 test HotpotQA items per target, append paragraphs from other HotpotQA
     rows (`random.Random(f"pad|{_id}|{target}")`) until the reference tokenizer count reaches
     2,048, 4,096 and 8,192. Same `group_key` and `source = "hotpot"` as the base item, so padding
     never crosses splits; `meta["template"] = f"pad{target}"`.
   Every row gets `meta["n_tokens"]` (reference tokenizer, prefix plus branch) and
   `meta["length_bucket"]` from it, so both models are bucketed on identical items. Choice rows
   with >= 3 options get `meta["perm_seed_k"]` for k in 1, 2, 3:
   `random.Random(f"{k}|{question.id}").sample(range(n), n)`, redrawn while equal to identity or
   reverse, stored as `"2,0,3,1"`. Selection: `assign_split(group_key, source, fractions, salt)`,
   keep rows whose split matches, order by the same hash `u`, take the quota. Write
   `tests/test_dataset.py::test_padded_variants_share_group_and_split` (every `pad*` row has
   the base row's `group_key` and split). Run `uv run python cli.py build --config
   configs/eval_data_v1.yaml --decode 3`; expect three decoded rows and counts `cal 2000 test
   3500`.
6. (1.5) pngwn converter, glue, eval only (NC treatment, Q5). `hf download
   pngwn/typed-decisions-v2 --repo-type dataset --local-dir data/raw/pngwn/typed-decisions-v2`,
   then print its splits and features: `uv run python -c "from datasets import load_dataset as
   L;d=L('data/raw/pngwn/typed-decisions-v2');print(d)"`. Record the state, question, options,
   type, gold and distribution field names under `external.pngwn_test.fields` in
   `configs/eval_m1.yaml`. `convert_pngwn` maps its types onto `choice`/`noul`/`score`, returns
   `None` for any other type (counted in the README as skipped), fills `consensus` from the
   distribution field if present. If there is no `cal` split, T comes from `eval_v1` cal and the
   README row says so.
7. (4.0) `evaluate.py`. Write `tests/test_evaluate.py::test_run_eval_toy_predictor`: a fake
   predictor that always puts 0.9 on display position 0 (the first option it is shown); on 10
   3-option Choice items with gold 0 and stored perms, `run_eval` returns
   `overall.acc_gold == 1.0` and `flip.reverse == 1.0` (reversed display shows canonical option 2
   first, so the remapped top-1 is 2, not 0), and writes
   `metrics.json` and `items.jsonl`. You write `run_eval`: (1) for each item call
   `pred.predict(state, [q], n_perms=1)`; (2) correctness vs `gold` and, separately, vs
   `argmax(consensus)` where not `None`; (3) per qtype and overall: `accuracy`, `ece` width and
   mass with 15 bins on T-scaled confidence plus width ECE at T = 1, `nll`, `brier`; (4) for each
   `flip_suite` entry build the permuted `Question` (options reordered so display j is
   `options[perm[j]]`), predict, map top-1 back with `perm[display_top1]`, then `flip_rate`
   against step 1; (5) accuracy per `length_bucket`; (5b) ECE per option-count bucket `2`, `3-5`, `6+` on T-scaled confidence; (6) `_risk_coverage_table` at `coverages`;
   (7) if `baseline` is set, load `out_dir.parent / baseline / "items.jsonl"`, align on
   `(state_id, question_id)`, raise `ValueError(f"baseline {baseline} missing {k} items")` if any
   are absent, and store `paired_bootstrap` on gold correctness. Glue: reliability PNGs
   (matplotlib, 15 equal-width bins), `README.md` tables. Run `uv run pytest -q`; expect `25
   passed` (Phase 2's 15 plus 10 new).
8. (1.0) `cli.py` glue and `configs/eval_m1.yaml`:
   ```yaml
   eval:
     data: {cal: data/built/eval_v1/cal.jsonl, test: data/built/eval_v1/test.jsonl}
     bins: 15
     flip_suite: [reverse, perm_seed_1, perm_seed_2, perm_seed_3]   # Choice items, >= 3 options
     length_edges: [512, 2048, 4096]                                 # le512, 512_2k, 2k_4k, gt4k
     coverages: [1.0, 0.9, 0.8, 0.7, 0.5]
     max_context: 8192
   external:
     pngwn_test: {path: data/raw/pngwn/typed-decisions-v2, split: test, cal_split: cal, publish: aggregate_only, fields: {}}
   ```
   `fit-cal` runs the predictor with T = 1 on `cal`, calls `fit_temperature` per qtype, writes a
   `Calibration` with keys `f"{qtype}:{n_perms}"`, `model_revision` = HF snapshot hash, `fitted_on
   = "eval_v1/cal@" + sha256[:12]`. Crosscheck: `uv run python cli.py eval --predictor b0 --data
   data/skeleton/gold_eval.jsonl --out-dir reports/m1/crosscheck`; per-type accuracy within 0.5 pt
   of `reports/phase2.json` `b0_engine`.
9. (2.0 attended, about 4 h wall clock) Bake-off, test numbers from here on:
   `uv run python cli.py fit-cal --adapter none --model openbmb/MiniCPM5-2B-Base --split cal
   --n-perms 1 --out runs/b0/calibration.json --config configs/eval_m1.yaml`, then
   `caffeinate -i uv run python cli.py eval --config configs/eval_m1.yaml --split test --predictor
   b0 --model openbmb/MiniCPM5-2B-Base --calibration runs/b0/calibration.json --out-dir
   reports/m1/minicpm5_b0` (about 45 min). Repeat both for `Qwen/Qwen3-4B-Base` into
   `runs/b0_qwen3_4b/calibration.json` and `reports/m1/qwen3_4b_b0` with `--baseline minicpm5_b0`
   (about 90 min). **Required** `Qwen/Qwen3.5-4B-Base` row into `reports/m1/qwen35_4b_b0`: this is the SemIf method
   (frozen Qwen3.5-4B, direct letter logits) run in our harness, and Phase 6 compares against it.
   Hybrid DeltaNet layers do not trim, so run it with `score_reencode(r, dtype="bfloat16")` (full
   re-encode per question, no branching, about 40 min on the test split; inference uses the fast
   Metal kernel). Report its latency as re-encode latency, not engine latency, and never train it.
   Also score it on SemIf authored144 (step 9a): it should land within 5 pts of SemIf's 0.813. pngwn pass: `--data` pointed at
   the converted test split, out `reports/m1/minicpm5_b0_pngwn`, aggregates only.
9a. (3.0) External anchors, glue plus one run each. These make M1 comparable with work other people already published.
    - **SemIf authored144** (MIT, `refs/semif/benchmarks/data/authored144.jsonl`). You write `convert_semif` in `sources/external.py` (print the first row's keys first and map them). Score it with SemIf's metric, **mean family balanced accuracy**, which you add to `evaluate.py` as `family_balanced_accuracy(items, correct) -> float` (mean over task families of per-family balanced accuracy). SemIf published native-BF16 values (`refs/semif/docs/RESULTS.md`): MiniCPM5-2B **0.686**, Qwen3.5-4B **0.813**, Qwen3-0.6B 0.440. Run MiniCPM5-2B B0 and write `reports/m1/semif_authored/`.
    - **JevBench public items** (MIT, `refs/jevbench/datasets/public/{easy,original,hard}.jsonl`). You write `convert_jevbench` (print keys first). Report accuracy and hard-tier ECE only, per file, and label them "JevBench public items, self-run, not an official JevBench score": the official score also covers sealed items, speed and cost.
    - **Eval-only rule:** both sets are written to `configs/eval_only_hashes.txt` (normalized-text sha256 of every state and question) so Phase 4 can exclude overlaps from training (JevBench's own README warns its public half can be trained on or selected against).
10. (0.5) Latency: `uv run python -m bench.latency_p2 --model <id> --out reports/m1/latency.md`
    for each model; W0/W1/W2 as defined in Phase 2.
11. (2.0) Write `reports/m1/README.md`: headline table next to pngwn arm B (accuracy 0.752, ECE
    0.015 T-scaled, 37.5% order flips) with the caveat that only the pngwn row is the same data;
    order flips per suite entry; long-input accuracy by bucket; reliability diagrams; the bake-off
    with paired CI; the SemIf authored144 row next to SemIf's published MiniCPM5-2B and Qwen3.5-4B numbers; JevBench public-item rows with the self-run caveat; latency; core line count from `tools/loc.py`; contamination caveat (all four public sets are likely in both bases'
    pretraining); Jev only as a footnote marked vendor-reported, not reproduced. Run `uv run ruff
    format . && uv run ruff check --fix .`, commit `phase3: eval harness and B0 bake-off (M1)`,
    `git tag m1`, `git push origin main --tags`.
12. (1.25) Interruption buffer; log hours in `ledger/hours.csv`.

**Verification gate**
- `uv run pytest -q` passes, 25 tests.
- `uv run python cli.py build --config configs/eval_data_v1.yaml` twice gives identical `sha256`
  values in `data/built/eval_v1/manifest.json`.
- Crosscheck in step 8 within 0.5 pt of Phase 2 per type.
- Below-chance tell (R6): MiniCPM5 B0 test gold accuracy >= 0.55 on `noul` and >= 0.35 on
  `choice`.
- **SemIf sanity check:** MiniCPM5-2B B0 on authored144 is within 5 pts of SemIf's published 0.686 (formats differ, so exact agreement is not expected). More than 5 pts **below** means a readout or format bug: fix it before publishing M1 (bounded 3 h, then publish with the gap explained).
- `uv run python tools/loc.py` exits 0 (core total at most 1,000 lines).
- `reports/m1/qwen3_4b_b0/metrics.json` has `baseline.lo <= baseline.delta <= baseline.hi`.
- `gh repo view --json visibility -q .visibility` prints `PUBLIC`, and `reports/m1/README.md`
  renders on GitHub with its PNGs.

**Rollback**
- Before step 11's push: delete `reports/m1/`, change the salt, rebuild; nothing depends on it.
  `git revert` the Phase 3 commits returns `cli.py eval` to Phase 2 behaviour.
- **Point of no return:** pushing `reports/m1/README.md` with test numbers. After that the salt in
  `configs/split.yaml` never changes.
- Data written during a failed window: `data/built/eval_v1/` and `reports/m1/*` are deterministic
  and regenerable; no teacher spend occurs. A bug found after publishing is fixed by rerunning
  with the same salt and committing `reports/m1/ERRATA.md` with old and new numbers, never by
  reshuffling.

**Kill criterion and decisions**
- **Q4 (base choice):** if Qwen3-4B B0 test gold accuracy exceeds MiniCPM5 B0 by more than 3 pts
  (paired `delta > 0.03`), the "Qwen3-4B training on a rented GPU" later item moves forward and
  Phases 5 to 6 are re-sequenced for it; otherwise MiniCPM5-2B-Base stays the MVP base. Record the
  result in ADR-0001. Prior (published, not ours): on authored144 SemIf measured frozen
  Qwen3.5-4B 12.7 pts above frozen MiniCPM5-2B, so expect this rule to fire; the Modal item is
  already costed in `cost.md` (about 16 USD per extra base model).
- **Calibration grouping:** if ECE on cal differs by more than 0.03 between option-count buckets,
  change `Calibration.temperature` keys to `f"{qtype}:{n_perms}:{bucket}"` (MiniSystemOne and
  poorjev both found a global T does not transfer across option counts).
- **R13 tripwire (A12):** if the chosen base's B0 `flip.reverse` on Choice test items exceeds 30%,
  P = 2 reversed pooling and T per `(qtype, 2)` move from "Not doing" into Phase 6.
- **Kill (ADR-0002):** if MiniCPM5 B0 is below the R6 floors on test after 3 h of readout
  debugging, switch the MVP base to Qwen3-4B-Base (`runs/models/qwen3-4b-base-raw`) and rerun
  steps 9 to 11 with it as the lead.
- **Calendar (R17):** if M1 is not public by the end of week 5, cut to 2 synthetic workflows and a
  100-item audit in Phase 4.
