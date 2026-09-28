### Phase 2: Hand-write the inference core and match the skeleton

**Goal:** `MLXBranchScorer.score` encodes each state once, branches the KV cache per question, and
reproduces the Phase 1 B0 numbers in `reports/skeleton.json` (per type, within 0.5 pt accuracy and
2e-2 per-item probability), with the prompt format `nanohunch-fmt-v1` frozen at the end.
**Effort:** 20 h = 3.3 engineer-days (week 3); the kill budget below can add at most 6 h.
**Depends on:** Phase 1 (`skeleton/fmt_ref.py`, `reports/skeleton/b0_fwd.json`,
`reports/skeleton/b0_rev.json`, `reports/skeleton.json`, `data/skeleton/gold_eval.jsonl`).
**Parallel with:** none (one engineer; this is the critical path).
**Risk:** medium, because the hidden-state and head-scale path in `mlx_lm` is model-specific and a
wrong scale gives plausible but wrong probabilities.

**Why this phase exists**

Phase 1 proved the idea with stock code that re-encodes the whole state for every question. That
is correct but costs one full state encode per question, and it hides the numerics inside
`model(ids)`. Every later phase (eval, teacher labels, training, release) consumes the five modules
written here, so they are written by hand, tested first, and checked against a reference we
already trust. The skeleton numbers are the oracle: if the new engine disagrees with them, the new
engine is wrong until proven otherwise.

**What you will understand after this phase**

1. **KV-cache branching.** A causal model computes position *i* from positions 0..*i* only. So
   the keys and values for the state do not depend on any question that comes after it: encode
   the 1,000-token state once, keep its cache, and for each of 16 questions feed only that
   question's roughly 40 tokens. Cost: 1,000 + 16 x 40 = 1,640 token positions instead of
   16 x 1,040 = 16,640, about 10x less. After each question you trim the cache back to 1,000 so
   the next question sees exactly the state and nothing else. This only works if the prefix is
   **token-identical** in every branch and in training. BPE tokenizers can merge across a string
   join: if the prefix ends in `"\n\n"` and the branch starts with `"###"`, `encode(a + b)` may
   produce a token spanning both, so the "same prefix" silently differs. ADR-0003 rule 2 removes
   this: prefix and branch are tokenized separately and concatenated as id lists, everywhere.
2. **Restricted softmax over label rows.** The LM head is a matrix with one row per vocabulary
   entry (about 120k rows of width H). We only care about three: the rows for `' A'`, `' B'`,
   `' C'`. Gather those rows, multiply by the last hidden state (at the last token of `Answer:`),
   and softmax over just those three numbers. Example: label logits `[2.0, 0.5, -1.0]` give
   `exp = [7.389, 1.649, 0.368]`, sum 9.406, probs `[0.79, 0.18, 0.04]`. Mass the model would put
   on other tokens is ignored by construction, and we never materialize the full vocab row.
3. **Temperature scaling.** Divide logits by one scalar T before the softmax. With T = 2 the
   example becomes `[1.0, 0.25, -0.5]`, probs `[0.59, 0.28, 0.13]`: softer, same order. Because
   dividing by a positive constant never changes which logit is largest, accuracy is unchanged;
   only confidence moves. A model that says 0.9 but is right 70% of the time needs T > 1. One T
   per `f"{qtype}:{n_perms}"` key, fitted by minimizing NLL on held-out items, is enough to fix
   most overconfidence, and it has one parameter, so it cannot overfit 500 items.

**Read before writing (30 min, reading only; see `configs/refs.yaml`)**
- `refs/semif/src/semif_phase1/mlx_backend.py`: prefill the state once in MLX and give each question its own copy of the cache. You will use `trim` instead of copying; compare the two and note in `reports/phase2.md` why trim is cheaper for batch-1 branches.
- `refs/semif/src/semif_phase1/direct.py` (`_slot_ids`): how they assert every label is one token. Your `label_vocab` does the same.
- `refs/decider/decider/model.py` (`slot_logits`): the same restricted head-row gather as your `label_logits`, but for several answer slots in one sequence. Do not copy the multi-slot layout (later list); read it to see that the readout is identical.

**Changes**

| File | Change |
|---|---|
| `pyproject.toml` | Add `[tool.pytest.ini_options] markers = ["slow: needs runs/models/minicpm5-2b-base-raw"]`. |
| `fmt.py` | New, core, user-written (budget 130 lines). Registry dataclasses and three errors, then `FORMAT_VERSION`, `label_vocab`, `permutations_for`, `render`; no `mlx` import and no `skeleton` import. `fmt.py` owns the ADR-0003 template text. `skeleton/fmt_ref.py` stays an independent oracle that the tests compare against. |
| `engine.py` | New, core, user-written (budget 200 lines). `label_logits`, `pool`, `Answer`, `to_answer`, `BranchLogits`, `MLXBranchScorer`. |
| `calibrate.py` | New, core, user-written (Phase 2 share about 50 of its 170 lines). `Calibration`, `fit_temperature`, `apply`. |
| `tests/conftest.py` | New. Session fixtures `tok` (`transformers.AutoTokenizer.from_pretrained(MODEL)`) and `scorer`; `MODEL = "runs/models/minicpm5-2b-base-raw"`; skip `slow` tests if the directory is missing. |
| `tests/test_fmt.py`, `tests/test_engine_readout.py`, `tests/test_calibrate.py`, `tests/test_engine.py` | New. Tests below, written before the modules. |
| `tests/fixtures/fmt_v1_golden.json` | New at freeze (step 12). 20 rendered id lists. |
| `cli.py` | New. `eval` subcommand only in this phase (Phase 3 swaps its body for `evaluate.run_eval`). |
| `bench/compare_skeleton.py`, `bench/latency_p2.py` | New glue scripts, spelled out in steps 10 and 11. |
| `reports/phase2/b0_fwd.json`, `reports/phase2/b0_rev.json`, `reports/phase2.json`, `reports/phase2.md` | New outputs. |

**Produces (interfaces later phases use)**

Exactly the registry signatures for `fmt.py`, `engine.py` and the temperature part of
`calibrate.py`, plus these fixed behaviours that Phases 3 to 7 rely on:
- `render` only permutes `choice` questions; `score` and `noul` always get the single identity
  perm, whatever `n_perms` is (the Noul prompt order is fixed by ADR-0003).
- `label_vocab(tok, "score", 10)` returns the ids of the bare digits `"0"`..`"9"` (ADR-0003
  Amendment 1, option (b)); `render` picks `vocab[v]` for each `v` in `q.values`, and every Score
  branch's `token_ids` end with the tokenizer's space id (`encode("Answer: ")[-1]`, checked to be
  exactly one extra id), so the readout position is that space token.
- `BranchLogits.logits` is `np.float32`, display order, T = 1.
- `Answer.expected`: `sum(values * probs)` for score, `probs[0]` (P(yes)) for noul, `None` for
  choice. `Answer.entropy_norm = -sum(p log p) / log(n)`.
- `uv run python cli.py eval --predictor b0 --data <jsonl> --out-dir <dir>` writes
  `<dir>/b0_fwd.json` and `<dir>/b0_rev.json` in the Phase 1 row format (input row plus `probs`
  in canonical order; noul rows identical in both files).

**Steps** (hours in brackets; tests first, you write the module bodies)

1. (1.0) Write the registry dataclasses and errors in `fmt.py`, the pytest marker, and
   `tests/conftest.py`. Run `uv run pytest -q`; expect `11 passed` (Phases 0 and 1, nothing new yet).
2. (1.5) Write `tests/test_fmt.py` (all model-free except the tokenizer fixture):
   - `test_prefix_identical_across_questions`: `render(tok, s, [q1], ...)` and
     `render(tok, s, [q2, q3], ...)` have equal `prefix_ids`, and both equal
     `tuple(tok.encode(prefix_text(s)))` from `skeleton.fmt_ref`.
   - `test_branch_ids_never_retokenized_join`: for the first 100 rows of
     `data/skeleton/gold_eval.jsonl`, `r.prefix_ids + b.token_ids ==
     tuple(tok.encode(prefix_text(s)) + tok.encode(branch_text(...), add_special_tokens=False))`;
     also print how many rows have `tok.encode(prefix_text(s) + branch_text(...))` different
     (information only, it shows why the rule exists).
   - `test_label_not_single_token_raises`: a `FakeTok` whose `encode(" A", ...)` returns `[5, 6]`;
     `pytest.raises(LabelNotSingleToken, match="' A'")` around `label_vocab(FakeTok(), "choice", 3)`.
   - `test_permutations_identity_then_reversed`: `permutations_for(4, 1) == [(0, 1, 2, 3)]`,
     `permutations_for(4, 2) == [(0, 1, 2, 3), (3, 2, 1, 0)]`,
     `permutations_for(4, 3)[2] == (1, 2, 3, 0)` (cyclic shift: `perm[d] = (d + s) % n`); rendering
     choice options `("x", "y", "z")` with `n_perms=2` gives perms `(0, 1, 2)` and `(2, 1, 0)`,
     and the second branch decodes to text containing `"A. z\nB. y\nC. x\nAnswer:"`. Rendering
     `("w", "x", "y", "z")` with `n_perms=3` gives a third branch containing
     `"A. x\nB. y\nC. z\nD. w\nAnswer:"`. A reversal is its own inverse, so only this
     cyclic case catches `perm` swapped with its inverse (R6).
   - `test_too_many_options_raises`: 27 choice options raise `TooManyOptions`; a score question
     with 11 values raises `TooManyOptions`.
   - `test_state_too_long_raises`: a 5,000-token state with `max_context=1024` raises
     `StateTooLong`. **`head_tail` truncation removed 2026-09-25** (cut by
     `risks-overengineering.md` O8, which `risks.md` makes the scope authority; no phase uses it).
     `render` has no `truncate` parameter; `Rendered.truncated` stays and is always `False`, so
     re-adding truncation from the later list does not change the type.
   Run `uv run pytest tests/test_fmt.py -q`; expect `6 failed` or collection errors with
   `ImportError: cannot import name 'render' from 'fmt'`.
3. (3.0) Write `fmt.py`. Algorithm for `render`:
   1. For each question, validate: choice `<= 26` options, score `<= 10` values in `0..9`
      ascending with `len(values) == len(options)`, noul exactly `("yes", "no")`; raise
      `TooManyOptions` (or `ValueError("noul options must be ('yes','no')")`).
   2. `label_vocab(tok, qtype, n)`: encode each label string of ADR-0003 rule 4 (as amended at
      P0-1; for score, the labels of values `0..n-1`) with `add_special_tokens=False`; if any gives
      a length other than 1, raise `LabelNotSingleToken(f"{s!r} -> {ids}")`. No cache (changed
      2026-09-25: encoding at most 26 short strings is negligible, and an `id(tok)` key can be reused
      by a different tokenizer after garbage collection). Write the strings in `fmt.py`; do not
      import them from `skeleton`.
   3. `permutations_for(n, k)`: identity, then reversed, then cyclic shifts by 1..k-2; raise
      `ValueError` if `k` exceeds the number of distinct perms produced.
   4. Branch text per perm, using your own ADR-0003 branch template, with display position `d`
      showing `q.options[perm[d]]`; `token_ids = tok.encode(text, add_special_tokens=False)`,
      plus the space id appended as an id for Score (Amendment 1); `label_ids` from step 2.
   5. `prefix_ids` = your own ADR-0003 prefix text encoded with BOS, the same way
      `skeleton/b0_reader.py` encodes it. The step 2 tests compare these ids with `fmt_ref`'s.
   6. If `len(prefix_ids) + longest branch > max_context`: `reject` raises
      `StateTooLong(f"{len(prefix_ids)} + {longest} > {max_context}")` (the only mode; see above).
   7. Return `Rendered(FORMAT_VERSION, prefix_ids, branches, truncated)`, branches in question
      order then perm order.
   Rerun step 2's command; expect `6 passed`.
4. (1.5) Write `tests/test_engine_readout.py`, run it (expect import failures), then `engine.py`:
   - `test_pool_maps_to_canonical`: branch A perm `(0, 1, 2)` logits `[2.0, 0.5, -1.0]`, branch B
     perm `(2, 1, 0)` logits `[-1.0, 0.5, 2.0]`; `np.exp(pool([A, B])["q1"])` is close to
     `[0.786, 0.175, 0.039]` at `atol=1e-3`. Branch C perm `(1, 2, 0)` with display logits
     `[0.5, -1.0, 2.0]`: `np.exp(pool([C])["q1"])` is also close to `[0.786, 0.175, 0.039]`.
     An inverted remap gives `[0.039, 0.786, 0.175]` here, and A/B alone cannot catch it.
   - `test_score_expected_value`: score question with values `(0, 1, 2)`, probs
     `[0.2, 0.3, 0.5]`; `to_answer(q, p).expected == pytest.approx(1.3)`.
   - `test_noul_p_yes`: probs `[0.7, 0.3]`; `expected == pytest.approx(0.7)`,
     `confidence == pytest.approx(0.7)`, `0 < entropy_norm < 1`.
   `label_logits` contract: inputs `hidden_last [B, H]`, `head_weight [V, H]`, `label_ids [n]`;
   output `[B, n]` in fp32. Only the `n` label rows are ever touched, never a `[B, V]` product. Raise
   `TypeError` if `head_weight` belongs to a quantized layer (the MVP loads bf16). `pool` contract:
   per branch, log-probabilities in display order; display position `d` belongs to canonical
   option `perm[d]`; average the log-probs of the same question across its branches, then
   renormalize so the result is a log-distribution in canonical order. Expect `3 passed`.
5. (2.0) Write `tests/test_calibrate.py`, run it, then `calibrate.py`:
   - `test_temperature_recovery`: `rng = np.random.default_rng(0)`; 20,000 items, 4 classes,
     `z = rng.normal(0, 1.5, (20000, 4))`; targets sampled from `softmax(z)`; inputs
     `log_softmax(2.5 * z)`; `abs(fit_temperature(x, t) - 2.5) / 2.5 < 0.05`. (Changed 2026-09-25:
     at 5,000 items the fitted T strayed up to 3.7% across 10 seeds, so 2% was a coin flip.)
     Plus `test_mixed_option_counts` (3- and 5-option items in one fit, padded with `-inf`).
   - `test_temperature_bounded`: targets equal to argmax of logits scaled by 50 give a result
     `>= 0.05`; targets drawn uniformly, independent of logits, give a result `<= 20`.
   `fit_temperature`: 200-point grid on `log T` over `[log 0.05, log 20]`, NLL of
   `log_softmax(x / T)`, then golden-section search in the bracket around the best grid point
   (40 iterations), clamp to `[0.05, 20]`. `apply`: raise `ValueError` if
   `cal.format_version != FORMAT_VERSION`; `T = cal.temperature[f"{qtype}:{n_perms}"]` (a missing
   key raises `KeyError`); return `softmax(pooled / T)`. Expect `2 passed`.
6. (1.0) Find the hidden state and head in `mlx_lm` for this model. Run
   `uv run python -c "import json; print(json.load(open('runs/models/minicpm5-2b-base-raw/config.json'))['model_type'])"`,
   then `uv run python -c "import inspect, mlx_lm.models.<type> as m; print(inspect.getsource(m.Model.__call__))"`.
   Verify in the `mlx_lm` source for MiniCPM5 (and `llama` for comparison): `Model.__call__` runs
   `out = self.model(inputs, cache=cache)` (final-normed hidden state `[B, L, H]`) and then either
   `self.model.embed_tokens.as_linear(out)` when `tie_word_embeddings`, or `self.lm_head(out)`.
   The older `minicpm` model type divides `out` by `hidden_size / dim_model_base`, but only when the
   head is untied. MiniCPM5 loads as `llama` (MEASURED 2026-09-24), so expect a scale of 1.0 and let
   the step 7 test confirm it. Write the
   exact expression you find into a comment at the top of `engine.py`; the engine keeps three
   private fields: `_backbone` (`model.model`), `_head_weight`, `_head_scale` (1.0 if none).
7. (1.5) Write `tests/test_engine.py`, every test `@pytest.mark.slow`. State: `prefix_text` of 5
   BoolQ passages from `data/raw/p0_items.jsonl` joined by `"\n\n"` (about 1k tokens), 16
   questions (8 BoolQ, 8 ARC) as in P0-4.
   - `test_label_logits_match_full_head`: fp32 scorer; for a 50-token input,
     `engine.label_logits` on `_backbone` output divided by `_head_scale` matches
     `model(ids)[0, -1][label_ids]` at `atol=1e-4`. This is the guard for step 6.
   - `test_isolation_exact`: bf16; q scored alone vs the same q among the 16:
     `np.abs(a - b).max() == 0.0` on the logits.
   - `test_oracle_fp32`: on the **CPU backend** (`mx.set_default_device(mx.cpu)` before building
     the scorer; decided at P0-4, 2026-09-24), `MLXBranchScorer(MODEL, dtype="float32")` on 8 of
     the 16 questions; softmax of `score(r)` vs softmax of `score_reencode(r)`:
     max abs prob diff `<= 1e-3`. P0-4 measured exactly 0.0 on CPU fp32; the GPU fp32 figure
     (1.08e-3 at a 1k state) is Metal accumulation order across matmul shapes, not branching, so
     the correctness check runs where the arithmetic is deterministic. The engine needs no device
     parameter: the test sets the default device.
   - `test_chunked_prefill_matches_oneshot`: a 3,000-token state, bf16, `prefill_chunk=1024` vs
     `prefill_chunk=10**9`: max abs prob diff `<= 2e-2`.
   Run `uv run pytest tests/test_engine.py -q`; expect import failures.
8. (4.0) Write `engine.py`. The contract below says what must hold, not how to write it. The
   library pieces you need are in `mlx_lm/models/cache.py` (prompt-cache creation, the
   trimmability check, trim) and `mlx_lm.load`.
   - `__init__`: load model and tokenizer (with the adapter if given). For `dtype="float32"`, cast
     every parameter before use. Set the three step 6 fields.
   - `score(r)`: (1) make a fresh cache and refuse to continue if it cannot be trimmed (a rotating
     or sliding cache would silently drop state); (2) prefill `prefix_ids` in chunks of
     `prefill_chunk`, forcing evaluation after each chunk so memory stays bounded; invariant:
     cache offset `P == len(r.prefix_ids)`; (3) for each branch, **batch 1 only** (R9, Q3), run its
     ids on the cache, take the last position, apply the head scale, read `label_logits`, then trim
     the cache back by exactly the branch length; invariant: offset returns to `P` after every
     branch; (4) return `BranchLogits` in fp32 numpy, display order, T = 1.
   - `score_reencode(r)`: no cache; one forward of `prefix_ids + token_ids` per branch; same
     readout at the last position, at the scorer's own dtype (changed 2026-09-25: no `dtype`
     argument, so a second model copy is never loaded; build the scorer with `dtype="float32"`
     for the oracle).
     A second fp32 copy of a 2.5B model (about 10 GB each) would push the fp32 oracle past the
     19.07 GB working set.
   Run `uv run pytest tests/test_engine.py -q`; expect `4 passed` in under 3 minutes.
9. (included in step 8) `uv run pytest -q`; expect `26 passed` (11 from Phases 0 and 1, plus 6
   fmt, 3 readout, 2 calibrate, 4 engine). Then
   `uv run ruff format . && uv run ruff check --fix .`, commit `phase2: hand-written core, tests pass`.
10. (2.0) Write `cli.py eval`: args `--predictor b0` (only choice this phase), `--data`,
    `--out-dir` (default `reports/phase2`), `--model` (default `MODEL`), `--adapter`. Per row `i`:
    `Question(str(i), row["type"], row["question"], tuple(row["options"]))`,
    `render(tok, row["state"], [q], n_perms=2 if choice else 1, max_context=8192)`, `score`;
    fwd = `pool([identity branch])`, rev = `pool([reversed branch])` (noul: rev = fwd); write
    `np.exp(...).tolist()` as `probs`. Write `bench/compare_skeleton.py --ref reports/skeleton
    --new reports/phase2`: per file, rows by position, print `max_abs_prob_diff` and per-type
    accuracy (argmax vs `gold_index`, the Phase 1 field name) for both. Every written row keeps
    `reversed` so `skeleton.tiny_eval` can compute flips. Run:
    `uv run python cli.py eval --predictor b0 --data data/skeleton/gold_eval.jsonl --out-dir reports/phase2`
    (expect about 4 minutes per direction, both written in one pass),
    `uv run python -m skeleton.tiny_eval --name b0_engine --fwd reports/phase2/b0_fwd.json --rev reports/phase2/b0_rev.json --json reports/phase2.json`,
    `uv run python -m bench.compare_skeleton --ref reports/skeleton --new reports/phase2`.
11. (1.0) Write `bench/latency_p2.py`: W0 = 256-token state and 4 questions, W1 = 1,024 and 16,
    W2 = 8,192 and 16; states cut to exact token counts from repeated `p0_items.jsonl` passages;
    time `render + score`, 1 warm-up then median of 5. Run `uv run python -m bench.latency_p2`;
    it prints `W0 <ms> W1 <ms> W2 <ms>` and appends them to `reports/phase2.md`.
12. (1.5) Freeze `nanohunch-fmt-v1`: write `tests/fixtures/fmt_v1_golden.json` (the rendered
    `prefix_ids`, `token_ids`, `label_ids` of the first 20 gold eval rows, choice with
    `n_perms=2`, plus 4 hand-made Score questions with values `0..4`, so the amended Score labels
    are frozen tested) and `tests/test_fmt.py::test_golden_v1_frozen` asserting `render` output
    equals the file and `FORMAT_VERSION == "nanohunch-fmt-v1"`. Write `reports/phase2.md` (skeleton vs
    engine table per type, max prob diff, latency table vs MEASURED synthetic, the step 6 head
    expression). Run `uv run pytest -q` (expect `27 passed`), ruff as step 9, commit
    `phase2: match skeleton, freeze nanohunch-fmt-v1`, `git tag nanohunch-fmt-v1`, log hours.

**Verification gate**
- `uv run python tools/loc.py` exits 0; `fmt.py` + `engine.py` + the Phase 2 part of `calibrate.py` should be near 380 lines. Over budget means the design is growing, not that the budget is wrong: cut before adding.
- `uv run pytest -q`: `27 passed` (11 from Phases 0 and 1, plus 16 new: 6 formatter + golden, 3
  readout, 2 calibrate, 4 engine).
- `tests/fixtures/fmt_v1_golden.json` contains Score entries as well as Noul and Choice.
- `bench.compare_skeleton` against `reports/skeleton`: per type (`noul`, `choice`)
  |delta accuracy| `<= 0.5` pt, and max abs prob diff `<= 3e-2` per item in both files (bf16
  branched vs bf16 full re-encode). **Changed 2026-09-25 from 2e-2** (author's decision): P0-4
  measured 2.9e-2 in bf16 on real weights; the 2e-2 came from the synthetic-weight 1.3e-2. The
  engine measured 2.72e-2 on 1,000 items, and 0.0 to 2.8e-6 on the worst items in CPU fp32, so the
  CPU fp32 oracle (1e-3) stays the correctness check.
- `reports/phase2.json` `b0_engine.choice.flip` within 1 pt of `reports/skeleton.json`
  `b0.choice.flip`.
- `reports/phase2.md` records W0, W1, W2 against the MEASURED synthetic 318 / 893 / about 6,600 ms
  for MiniCPM5-2B. Above 1.5x any of these is not a stop; note it and check the prefill chunk and
  `mx.eval` placement before Phase 3.

**Rollback**
- Everything is new files: `git revert` the phase commits, or `git checkout` the Phase 1 tag.
  Phase 1's `skeleton/` path stays working throughout and remains the fallback reader.
- Point of no return: none inside the phase. The format freeze (`git tag nanohunch-fmt-v1`) is a soft
  one-way door (ADR-0003): nothing is trained or logit-labelled on it until Phase 4 and 5, so a v2
  before then costs only regenerating the golden file. After Phase 5 it costs relabelling and
  retraining.
- Data written during a failed window: `reports/phase2/*.json` and `reports/phase2.json` are
  written at the end of a run and overwritten by rerunning step 10; nothing downstream reads them
  until Phase 3.

**Kill criterion**
- If `test_oracle_fp32` (CPU fp32) cannot get under 1e-3 after 6 h of debugging (step 6 head expression
  checked, trim offsets asserted, chunk size 10**9 tried), stop. The design assumed branched and
  re-encoded logits agree (P0-4 measured about 4.0e-4, ADR-0003 verification). File the exact
  repro (model revision, `mlx` and `mlx_lm` versions, a 1k-token state, the diff) as a
  `mlx_lm` issue, then make `score` call `score_reencode` on a bf16 scorer (full re-encode, no branching)
  for all later phases, rerun the gate with that path, and report branching as future work in
  the release. Latency then scales with question count (W1 about 16x the prefix cost), which
  Phase 3 and Phase 6 eval sizes can absorb on the Mac.
