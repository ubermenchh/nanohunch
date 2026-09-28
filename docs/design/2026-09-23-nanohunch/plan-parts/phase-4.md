### Phase 4: Data v1 and the teacher labeller, audit before bulk

**Goal:** `data/built/v1/{train,cal,test}.jsonl` exist with 3,000 train decisions, cal of at least
2,000 and test of at least 3,500 (the `eval_v1` items plus synthetic rows; `test.jsonl` also holds
the held-out-template rows, marked `split: "test_ood"`), all keyed by option id. Every teacher label
is cached and costed, and a 100-item blind human audit is scored per question type **before** the
3k-decision labelling run. (R3's "no bulk labelling before the 1k/3k curve" refers to the Phase 6 v2
scale-up; Phase 4 labels only what the curve and the eval splits need.)
**Effort:** 24 h = 4.0 engineer-days, weeks 5 to 6 (includes 2 h interruption buffer).
**Depends on:** Phase 0 P0-7 (`configs/teachers_p0.yaml`, `bench/teacher_probe.py`), Phase 3
(the frozen salt in `configs/split.yaml`, `assign_split`, `sources/external.norm_hash`). **Parallel with:** Phase 3 for the
public downloads and adapters only (one engineer, so interleaved evenings, not saved calendar time).
**Risk:** medium, because teacher label quality (A1) is unmeasured until the audit in step 12, and
Q6 (DeepSeek terms) decides whether one teacher is releasable.

**Why this phase exists**

The model learns whatever the labels say, including their mistakes. pngwn's v1 shipped a label
keyed by option *index*; one shuffle later it was inverted, and nobody noticed until accuracy came
out below chance. This phase builds the data with labels keyed by option *id*, measures two
teachers against gold we generate for free (spec facts), and makes you judge 100 items by hand
while a bad teacher still costs cents instead of a training run.

**What you will understand after this phase**

- *Soft labels from `top_logprobs`.* The teacher answers with one token and OpenRouter returns its
  20 most likely tokens with log-probabilities. For a 3-option Choice it might return `" A"` -0.22,
  `" B"` -1.9, `" C"` -3.5, plus tokens like `"The"` that are not labels. exp() gives 0.80, 0.15,
  0.03. Their sum, 0.98, is `candidate_mass`: how much of the teacher's belief landed on a legal
  answer. Renormalize over the candidates: 0.80/0.98, 0.15/0.98, 0.03/0.98 = 0.816, 0.153, 0.031.
  (0.817, 0.152, 0.031 before rounding the inputs.) That vector, not the argmax, is the training target. Mass below 0.9 means the teacher was trying
  to say something else, so the label is untrustworthy: retry once, then drop and log.
- *Log-linear pooling of two orders.* Teachers prefer some positions. Suppose a 2-option Choice
  whose true logits are o1 = 0.2, o2 = 0.0 (unbiased p(o1) = 0.550), and the teacher adds +0.5 to
  whatever sits in position A. Canonical order (o1 in A): p(o1) = 0.668. Reversed (o2 in A):
  p(o1) = 0.426. Map each result back to option ids, average the *log*-probs per id
  (o1: (-0.403 + -0.854)/2 = -0.629; o2: (-1.103 + -0.554)/2 = -0.829), renormalize: the gap is
  exactly 0.200 again, so p(o1) = 0.550. The bias is additive in log space, so it subtracts out; an
  arithmetic mean (0.547) only approximates. With 3 or more options, reversal cancels any bias that
  is linear in position exactly and halves a bias concentrated on one position.
- *Why option ids, not indices.* `canonical_order` is a seeded shuffle and every later phase
  permutes again (augmentation in Phase 5, flip rate in Phase 3). A label stored as "index 0"
  silently means a different option after each shuffle. A label stored as `{"opt_refund": 0.8}`
  cannot be misread. The below-chance tell (a gold slice scoring under 1/n) is the alarm for this.

**Changes**

| File | Change |
|---|---|
| `dataset.py` | Core. Constants `LICENSE_ALLOWLIST`, `RELEASABLE_TEACHERS`, `DENY_SOURCES`, `EVAL_ONLY_SOURCES`. **You write:** `to_display`, `build_row`, `assert_row_releasable`, `renormalize`, `pool_orders`, `consensus`, `leaks_answer_verbatim`, `find_lookup_questions`, `teachers_disagree`, `route_split`, `sample_audit`. Glue kept thin: `__main__` with `--config`, `--stage {ingest,generate,build}`, `--decode N`, manifest writer. |
| `sources/public.py`, `sources/triage.py` | Outside the line budget. Train-side public adapters, pngwn / SemIf / JevBench overlap exclusion (step 3), and the triage generator: `sample_spec`, `spec_questions` (you write these two; they define ground truth), generator driver. |
| `label.py` | Outside the line budget (HTTP glue). `TeacherSpec` (registry), errors `CreditExhausted(RuntimeError)`, `TeacherUnavailable(RuntimeError)`, `cache_key`, `request_body`, `label_decision` (calls `dataset.renormalize`), cache IO, `spend.csv` rows. |
| `evaluate.py` | **You write:** `audit_agreement`. Phase 6 step 9 reuses it unchanged. |
| `cli.py` | Glue: `label --config C [--limit N] [--max-usd X] [--stage pilot|bulk]`; `audit sample|judge`. |
| `configs/data_v1.yaml` | New. `split_config: configs/split.yaml` (the salt is read from there, never copied), `targets`, `templates`, `held_out_templates: [triage_webform]`, `generator`, `teachers: configs/teachers_p0.yaml`, `min_candidate_mass: 0.9`, `disagree_jsd: 0.3`, `headline_excluded_types: []`. |
| `prompts/triage_render.txt`, `prompts/triage_judgement.txt` | New. Generator prompts (ticket rendering; judgement question writing with JSON output). |
| `tests/test_dataset.py`, `tests/test_label.py`, `tests/test_evaluate.py` | 8 new tests, step 1. `test_label.py` is new; the other two already hold 1 Phase 3 test each. |
| `data/raw/`, `data/labels/cache.jsonl`, `data/labels/dropped.jsonl`, `data/audit/audit_v1.csv`, `data/built/v1/` | Gitignored outputs. |
| `ledger/spend.csv`, `ledger/hours.csv` | Append this phase's rows (`phase=4`). |

**Produces (interfaces later phases use)**

- Row schema `nanohunch.data.v1`, one JSONL line per state:
  ```json
  {"schema_version": "nanohunch.data.v1", "state_id": "tri-000412", "group_key": "tri-spec-000412",
   "split": "train", "source": "nanohunch/synthetic-triage", "source_revision": "gen-v1",
   "source_license": "apache-2.0", "state": "Subject: charged twice ...",
   "meta": {"template": "triage_email", "workflow": "triage", "length_bucket": "le512"},
   "decisions": [{"question_id": "q_severity", "qtype": "score", "text": "How severe ...?",
     "options": [{"id": "q_severity:0", "text": "cosmetic"}, {"id": "q_severity:1", "text": "minor"},
                 {"id": "q_severity:2", "text": "degraded"}, {"id": "q_severity:3", "text": "major"},
                 {"id": "q_severity:4", "text": "outage"}],
     "canonical_order": ["q_severity:0", "q_severity:1", "q_severity:2", "q_severity:3", "q_severity:4"],
     "values": [0, 1, 2, 3, 4], "gold_option_id": "q_severity:3", "label_origin": "spec",
     "teachers": [{"teacher_id": "qwen3.6-35b-a3b", "model_version": "<response model field>",
       "host": "<provider tag>", "date": "2026-10-26", "format_version": "nanohunch-fmt-v1",
       "perm": [0, 1, 2, 3, 4],
       "probs_by_option_id": {"q_severity:0": 0.01, "q_severity:1": 0.04, "q_severity:2": 0.20,
                              "q_severity:3": 0.71, "q_severity:4": 0.04}, "candidate_mass": 0.97}],
     "consensus_by_option_id": {"q_severity:0": 0.02, "q_severity:1": 0.05, "q_severity:2": 0.21,
                                "q_severity:3": 0.68, "q_severity:4": 0.04}}]}
  ```
  This example is the R6 contract, so it must stay self-consistent. Option text carries no value
  prefix (`render` adds `{value}: `), the perm has one entry per option, and gold is one of the
  option ids. `length_bucket` uses the Phase 3 names (`le512`, `512_2k`, `2k_4k`, `gt4k`).
  `gold_option_id` is null and `label_origin` is `"teacher"` for judgement questions; public rows
  have `label_origin: "gold"` and `teachers: []`, `consensus_by_option_id: null`.
- `dataset.LICENSE_ALLOWLIST: frozenset[str] = frozenset({"cc-by-sa-3.0", "cc-by-sa-4.0", "mit", "apache-2.0"})`.
- `dataset.RELEASABLE_TEACHERS: frozenset[str] = frozenset({"deepseek-v4.1-flash", "qwen3.6-35b-a3b"})`
  (DeepSeek is removed in step 9 if Q6 comes back negative). Phase 7 imports both; never redefine.
- `dataset.assert_row_releasable(row: dict) -> None`, raises `ValueError(f"{state_id}: {field}")`.
- `dataset.to_display(probs_by_option_id: dict[str, float], canonical_order: Sequence[str], perm: Perm) -> np.ndarray`:
  with `target` the canonical-order vector, returns `t` where `t[j] == target[perm[j]]`. Phase 5 imports it;
  it is the project's **only** option-id-to-display remap (a second copy is how the pngwn inversion happens).
- `dataset.route_split(row: dict, held_out_templates: Sequence[str], fractions: dict[str, float], salt: str) -> str`:
  `"test_ood"` for held-out templates, else `assign_split(row["group_key"], row["source"], fractions, salt)`.
- `dataset.sample_audit(rows: Sequence[dict], *, n: int, seed: int, min_per_qtype: int, agree_thr: float = 0.8) -> list[dict]`.
- `label.label_decision(spec, state, q, perm, *, cache: Path, option_ids: tuple[str, ...], client: httpx.Client | None = None) -> dict`
  returning the `teachers[]` entry shape above. `option_ids[i]` names `q.options[i]` (canonical order).
- `dataset.renormalize(top: list[dict], labels: list[str], qtype: QType) -> tuple[float, list[float]]` (mass, probs in display order).
- `dataset.pool_orders(a: dict[str, float], b: dict[str, float]) -> dict[str, float]`; `dataset.consensus(per_teacher: Sequence[dict[str, float]]) -> dict[str, float]`.
- `evaluate.audit_agreement(csv_path: Path, *, against: str = "consensus_top1") -> dict[str, tuple[float, int]]`.
- `data/built/v1/manifest.json`: `dataset_version` (sha256 of the manifest JSON without that key,
  `sort_keys=True`), `split_salt`, `counts` per split and source, `sha256` per JSONL file.

**Steps**

1. (2 h) Tests first. Create `tests/test_label.py` and add to the two existing files, then run
   `uv run pytest tests/test_dataset.py tests/test_label.py tests/test_evaluate.py -q` and expect
   8 failures (`ImportError` on the new names) and the 2 Phase 3 tests still passing:
   - `test_dataset.py::test_perm_remap_property`: for 200 random `(n in 2..8, perm, probs)`,
     `to_display(...)[j] == target[perm[j]]` for every j, and the output sums to 1 within 1e-9.
   - `test_dataset.py::test_releasable_assert_rejects_nc`: a row with `source_license: "cc-by-nc-4.0"`
     raises `ValueError` whose message contains the `state_id` and `source_license`; a row with a
     teacher entry `teacher_id: "gpt-x"` raises naming `teacher_id`; sources `pngwn/typed-decisions`
     and `pngwn/typed-decisions-v2` both raise; an allowlisted row returns None.
   - `test_dataset.py::test_lookup_filter_drops_team_leak`: 40 rows where question "Which team
     owns this?" is a deterministic function of spec field `product` return that question text from
     `find_lookup_questions`; a question with 2 answers for one product value is not returned.
   - `test_dataset.py::test_below_chance_tell`: 400 Noul gold items and a predictor that is right
     80% of the time score 0.80 by `calibrate.accuracy` when targets come from `gold_option_id`; the
     same rows with targets stored by index and then a yes/no swap score 0.20 (< 0.5, the tell); 400
     4-option Choice items with index targets after a random reshuffle score in [0.18, 0.32].
   - `test_label.py::test_renormalize_candidates`: the top list `[" A" -0.22, " B" -1.9, " C" -3.5,
     "The" -2.0]` with labels `[" A", " B", " C"]` returns mass 0.982 (abs 0.005) and probs
     [0.817, 0.152, 0.031] (abs 0.002); it equals `bench.teacher_probe.candidate_mass` on 20 rows of
     `reports/phase0/teacher_probe.csv` within 1e-9. (Phase 0 must therefore store the raw
     `top_logprobs` JSON in a `top_logprobs` column of that CSV.)
   - `test_label.py::test_pool_two_orders_cancels_position_bias`: a synthetic teacher adds +0.5 to the
     logit in display position A. n = 2, true logits (0.2, 0.0): `pool_orders` equals the unbiased
     softmax within 1e-9. n = 4, true logits (0.0, 0.3, 0.0, 0.0): each single order has the wrong
     argmax (the option shown in A), the pooled argmax is the second canonical option (logit 0.3).
   - `test_label.py::test_cache_key_idempotent`: with `httpx.MockTransport` returning a canned
     completion and counting requests, two identical `label_decision` calls return equal dicts, the
     counter is 1, `cache.jsonl` has 1 line, and that line has no key named `headers` or `authorization`.
   - `test_evaluate.py::test_audit_agreement_per_type`: a 6-row CSV (4 `choice` rows, 3 matching
     `consensus_top1`; 2 `noul` rows, 2 matching) returns `{"choice": (0.75, 4), "noul": (1.0, 2)}`.
2. (0.5 h) Write the constants and `assert_row_releasable`: required fields `source_license`, and
   `teacher_id` on every teacher entry; reject `source in DENY_SOURCES` (the Phase 7 deny-list),
   `split == "train" and source in EVAL_ONLY_SOURCES` (`{"TIGER-Lab/MMLU-Pro"}`), license not in
   `LICENSE_ALLOWLIST`, teacher-origin decisions whose teachers are not all in `RELEASABLE_TEACHERS`.
   `DENY_SOURCES` includes both `pngwn/typed-decisions` and `pngwn/typed-decisions-v2`.
3. (0.5 h) Write `to_display` and `build_row` (option ids assigned at ingest as
   `f"{question_id}:{k}"`; `canonical_order` = option ids shuffled by
   `random.Random(sha256(salt|state_id|question_id))` for Choice, fixed `["yes","no"]` for Noul,
   ascending `values` for Score). Run the step 1 command; expect 3 of 8 passing (perm, releasable,
   below-chance).
4. (2.5 h) Public adapters (glue), train-side splits only, `uv add datasets` once. Sources and
   licenses: `google/boolq` (Noul, `cc-by-sa-3.0`), `hotpotqa/hotpot_qa` config `distractor` with
   answer in {yes, no} (Noul, `cc-by-sa-4.0`), `allenai/ai2_arc` `ARC-Challenge` and `ARC-Easy`
   (Choice, `cc-by-sa-4.0`), `tau/commonsense_qa` (Choice 5, `mit`). `source_revision` =
   `huggingface_hub.HfApi().dataset_info(id).sha`. SEC-14: load the `test` split of
   `data/raw/pngwn/typed-decisions-v2` (the copy Phase 3 downloaded), take `norm_hash` of every
   state and question, and drop any public item whose state or question hash is in that set or in
   `configs/eval_only_hashes.txt` (SemIf authored144 and JevBench public items, from Phase 3 step
   9a). Print the dropped count per source. Run
   `uv run python dataset.py --config configs/data_v1.yaml --stage ingest`; expect one line per
   source `<source> kept=<k> dropped_pngwn_overlap=<d> dropped_eval_only=<e>` and `data/raw/public_v1.jsonl`.
5. (3 h) Synthetic workflow 1, support-ticket triage, 4 templates: `triage_email`, `triage_chat`,
   `triage_handoff`, `triage_webform` (held out, routed to `test_ood`). You write `sample_spec(template,
   rng) -> dict` drawing: `product` (6 values), `severity` 0..4, `customer_tier` {free, pro,
   enterprise}, `sentiment` {negative, neutral, positive}, `days_since_purchase` 0..60, `prior_tickets`
   0..5, `channel`, `refund_eligible` (derived: `days_since_purchase <= 30 and customer_tier != "free"`).
   You write `spec_questions(spec, template)` returning spec-fact decisions with `label_origin: "spec"`
   and `gold_option_id` from the spec. Glue: the generator (`generator.model` = the DeepSeek V4.1
   Flash id from `configs/teachers_p0.yaml`, `temperature: 0.8`, `max_tokens: 700`) renders the ticket
   from `prompts/triage_render.txt` with the spec as JSON; a second call with
   `prompts/triage_judgement.txt` writes 3 judgement questions as JSON (one Noul, one Choice with 3 to 8
   options, one Score with anchored levels 0..4, the level names as option text without the number).
   Reject malformed JSON and regenerate once, then skip.
6. (2 h) Filters. You write `leaks_answer_verbatim(state, decision) -> bool` (a judgement question
   whose correct-looking option text appears verbatim in the state), `find_lookup_questions(rows) ->
   set[str]` (**judgement questions only**: conditional entropy of the answer given any single spec
   field is 0 over at least 20 rows, the pngwn `team` leak; spec-fact questions are lookups by
   design and are never passed in), and `teachers_disagree(decision, jsd_thr) -> bool` (JSD between
   the two teachers' pooled distributions above `disagree_jsd`, flagged in `meta`, kept for the
   audit pool). `--stage generate` runs the first filter; `find_lookup_questions` runs at `--stage
   build`, once judgement questions have consensus labels, and so does `teachers_disagree`. Run
   `uv run python dataset.py --config configs/data_v1.yaml --stage generate`; it prints
   `states=<n> dropped_verbatim=<a>` per template. Run the step 1 command; expect 4 of 8.
7. (3 h) `label.py`. Glue given here; you write `renormalize` (step 1 contract), `pool_orders` (log of
   each id's prob, mean per id, exp, renormalize) and `consensus` (arithmetic mean per id across
   teachers, renormalized). Request body with the prompt text
   `prefix_text(state) + branch_text(qtype, text, displayed_options, values)` and
   `label_strings(qtype, n, values)` from `skeleton/fmt_ref.py` (Score as amended at P0-1). The
   system line depends on qtype, because Phase 0's `SYSTEM` allowed only a letter or yes/no, which
   would collapse Score candidate mass: Choice "a capital letter", Noul "yes or no", Score "the
   number of one level" (or "a capital letter" if Amendment 1 chose letters for Score).
   `renormalize` strips the leading space before matching, so a teacher answering `"7"` or `" 7"`
   counts either way:
   ```python
   def request_body(spec: TeacherSpec, prompt: str, system: str) -> dict:
       return {"model": spec.model, "max_tokens": 1, "temperature": 0, "logprobs": True,
               "top_logprobs": spec.top_logprobs, "reasoning": {"enabled": False},
               "usage": {"include": True},
               "provider": {"order": list(spec.provider_order), "allow_fallbacks": False,
                            "require_parameters": True},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}]}
   def cache_key(teacher_id, model_version, format_version, state_id, question_id, perm, prompt) -> str:
       raw = "|".join([teacher_id, model_version, format_version, state_id, question_id,
                       ",".join(map(str, perm)), hashlib.sha256(prompt.encode()).hexdigest()])
       return hashlib.sha256(raw.encode()).hexdigest()
   ```
   The prompt hash is in the key because step 9 can regenerate a state under the same `state_id`;
   without it the cache would serve the old text's labels. `TeacherSpec(**row)` takes only its
   registry fields, so drop `quantization` when loading `configs/teachers_p0.yaml` and log it per
   row instead.
   `model_version` in the key is the pinned `spec.model` before the call; the response `model`
   field is stored in the entry. `label_decision`: look up the key in the in-memory index of
   `data/labels/cache.jsonl` (lines failing `json.loads` are skipped and counted); on a miss, POST with
   `httpx.Client(timeout=60)`; on `httpx.TimeoutException`, HTTP 429 or 5xx retry up to 3 times,
   sleeping `2 * 2**k * random.uniform(0.5, 1.5)` s; HTTP 402 raises `CreditExhausted`; HTTP 404 whose
   `error.message` starts with `No endpoints found` raises `TeacherUnavailable`; null
   `choices[0].logprobs` gives mass 0. If mass < `min_candidate_mass`, retry the call once; if still
   low, append `{key, teacher_id, question_id, mass, reason: "low_candidate_mass"}` to
   `data/labels/dropped.jsonl` and return None. Append to the cache only the fields `key, teacher_id,
   model_version, host, date, format_version, perm, labels, top_logprobs, candidate_mass,
   prompt_tokens, completion_tokens, usd`, then `flush()` and `os.fsync()`. Append a `ledger/spend.csv`
   row (`phase=4`, `item=label`, host from the response `provider`, `usd` = `usage.cost`). Choice is
   labelled at perm `tuple(range(n))` and `tuple(reversed(range(n)))`, then `pool_orders` per teacher;
   Noul and Score at the identity perm only. Run the step 1 command; expect 7 of 8.
8. (1 h) Decode check (U1): `uv run python dataset.py --config configs/data_v1.yaml --decode 20`
   prints 20 rows as the teacher would see them (state head, question, options in display order with
   ids, gold id). Read all 20. Any gold id pointing at the wrong option text stops the phase until fixed.
9. (1 h) Q6 and pilot. Read the current DeepSeek terms page; record the answer in `reports/phase4.md`.
   If V4.1 Flash is not covered: remove it from `RELEASABLE_TEACHERS`, set `generator.model` to the
   Qwen3.6 id, rerun step 5, and continue single-teacher. Pilot:
   `uv run python cli.py label --config configs/data_v1.yaml --stage pilot --limit 600 --max-usd 1`
   labels 600 train-side decisions with both teachers. Expect a final line
   `labelled=<k> dropped_low_mass=<d> usd=<x>` with `d/600 < 0.05` and x under 0.40, plus
   `spec_agreement choice=<a> noul=<b> score=<c>` (teacher consensus top-1 against spec gold: the free
   continuous audit) and `mean_candidate_mass choice=<a> noul=<b> score=<c>`. P0-7 probed only BoolQ
   and ARC, so this is the first Score measurement. If Score mass is below 0.9 for a teacher, that
   teacher's Score labels are dropped and Score trains on spec gold only.
10. (0.5 h) You write `sample_audit`: from `label_origin == "teacher"` pilot decisions, at least
    `min_per_qtype` per qtype, remaining slots weighted 2:1 toward decisions where both teachers put
    more than `agree_thr` on the same option (confident errors are the expensive kind), seeded.
    `uv run python cli.py audit sample --config configs/data_v1.yaml --n 100 --seed 0 --min-per-qtype 25 --out data/audit/audit_v1.csv`
    writes columns `item_id, qtype, teacher_top1, consensus_top1, human_option_id, notes`.
11. (2.5 h) Blind judging: `uv run python cli.py audit judge --csv data/audit/audit_v1.csv` shows the
    state, question and options with ids in canonical order, never the teacher columns; it accepts
    only a listed option id, `s` (skip, leaves `human_option_id` empty) or `n` (notes), and saves
    after every row so it resumes.
12. (0.5 h) Implement `audit_agreement` (skipped rows excluded from n). Run
    `uv run python -c "from pathlib import Path;from evaluate import audit_agreement as a;p=Path('data/audit/audit_v1.csv');print(a(p));print(a(p,against='teacher_top1'))"`.
    Tripwire per type: consensus below 0.75 moves the type into `headline_excluded_types`. Fallback for
    that type: if `teacher_top1` (the DeepSeek teacher, or the only teacher) is at or above 0.75, train
    it single-teacher plus gold; otherwise train it on gold and spec labels only. Run the step 1
    command; expect `8 passed`.
13. (1.5 h) Label to target and build. `uv run python cli.py label --config configs/data_v1.yaml --stage bulk --max-usd 6`
    ("bulk" is the stage name; it labels only the 3k train target plus synthetic cal/test/test_ood
    rows), then `uv run python dataset.py --config configs/data_v1.yaml --stage build`. Build
    converts every Phase 3 `eval_v1` cal and test item into a `nanohunch.data.v1` row with the same
    `state_id`, question id, gold and `source_license`, so `data/built/eval_v1/` stays untouched and
    M1 remains reproducible. It then appends new rows via `route_split` and fills to `targets`:
    train 3,000 decisions (public gold plus teacher-labelled synthetic); cal = `eval_v1` cal plus
    synthetic cal rows; `test.jsonl` = `eval_v1` test plus synthetic `test` rows plus `test_ood`
    rows (held-out template, marked by `split`). It drops any train row whose state or question
    `norm_hash` is in `configs/eval_only_hashes.txt` and prints `dropped_eval_only` (R22), then
    runs `assert_row_releasable` on every train and cal row, writes to `data/built/v1.tmp/` and
    renames to `data/built/v1/`. Expected spend about 3 USD (about 0.00022 USD per decision for
    both teachers).
14. (0.5 h) `uv run ruff format . && uv run ruff check .`, append hours, commit
    `phase4: data v1, teacher labeller, audit v1`.

**Verification gate**

- `uv run pytest tests/test_dataset.py tests/test_label.py tests/test_evaluate.py -q` prints `10 passed`
  (8 new plus 2 from Phase 3); `uv run pytest -q` prints `45 passed`.
- `uv run python -c "import json;m=json.load(open('data/built/v1/manifest.json'));print(m['dataset_version'][:12],m['counts']['decisions'])"`
  prints train 3000 (within 2%), cal at least 2000, test at least 3500, and `test_ood` at least 300
  decisions **with spec gold** (the Phase 5 kill rule is measured on these; under 300 gives a CI
  too wide to decide anything, so generate more `triage_webform` states before Phase 5). The
  manifest's `counts` has a `decisions` block per split alongside the row counts.
- `uv run python -c "import json,dataset as b;rows=[json.loads(l) for s in ['train','cal'] for l in open(f'data/built/v1/{s}.jsonl')];[b.assert_row_releasable(r) for r in rows];print(len(rows),'ok')"` prints `<n> ok`.
- Audit: every qtype has n at least 25 and either agreement at or above 0.75 or an entry in
  `headline_excluded_types` with its fallback written in `reports/phase4.md`.
- `uv run python -c "import csv;print(round(sum(float(r['usd']) for r in csv.DictReader(open('ledger/spend.csv')) if r['phase']=='4'),2))"`
  prints below `6.0`. `data/labels/dropped.jsonl` holds under 5% of attempted decisions.

**Rollback**

- Undo: `rm -rf data/built/v1` and revert the phase commit; Phase 3 cal and test rows are unchanged
  because build copies them, so Phase 3 reports stay valid. No adapter or public artefact depends on v1.
- Point of no return: none. Money spent (under 6 USD) is sunk but not lost: every call is in the cache.
- Data written during a failed window: `cache.jsonl` is append-only with fsync per line; a torn last
  line is skipped on load and that call repeats (one duplicate charge at most). Rerunning `label`
  resumes from the cache with zero repeat requests for completed keys. `v1.tmp/` is deleted on rerun.

**Kill criterion**

- Assumption A1 (teacher consensus is a usable label). If consensus audit agreement is below 0.75 on
  two or more of the three qtypes after one bounded repair (4 h: prompt fix in
  `prompts/triage_judgement.txt`, relabel the pilot, 50 fresh audit items), stop teacher labelling.
  Train Phase 5 on `label_origin in {"gold", "spec"}` only and restrict the headline to gold slices.
- Q1 (P0-7). If a teacher's `dropped_low_mass` exceeds 10% of pilot calls after one host switch from
  the P0-7 endpoint list, drop that teacher and run single teacher plus gold.
