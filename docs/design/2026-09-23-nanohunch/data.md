# Data and state: nanohunch

Author: sd-data. Mode: GREENFIELD. Depth: standard. Date: 2026-09-23.

Scope: the **training, calibration and eval data layer**. There is no database
and no request-path state (`architecture.md:427-429`, D14 at
`architecture.md:844`); persistent state is a paid-call cache, versioned
dataset builds, eval reports and an audit file. The repo has zero commits
(`_brief/system-map.md:3-5`), so every schema claim cites a design file.

Labels: `MEASURED`, `DERIVED` (arithmetic shown), `ASSUMPTION` (with range).
Prefixes: `REQ:`/`RN:` (`_brief/`), `ARCH:`, `SEC:`, `COST:`, `CAP:`, `REL:`
(the sibling `.md` files), `ADR3:` (`adr/0003-prompt-serialization-format-v1.md`).

Inherited, not reopened: nanohunch-fmt-v1 and split tokenization (`ADR3:19-47`);
restricted label-logit readout, 26 options, 10 levels (`ARCH:297-320`);
licence allowlist and NC deny-list (`SEC:223-291`); teacher allowlist
(`SEC:342-351`); cache record schema (`SEC:516-519`); bulk soft labels from
DeepSeek V4.1 Flash plus Qwen3.6-35B-A3B via API logprobs, closed strong
models as eval reference only (`COST:84-85`, `COST:182-184`, `SEC:61-68`).

---

## 1. Access patterns (priority order)

| # | Pattern | Frequency | Selectivity | Shape it forces |
|---|---|---|---|---|
| P1 | Trainer streams train split, state-grouped, reshuffled and re-permuted per epoch | every step; 2 epochs x 6 to 9 runs (`COST:186-188`) | full scan | immutable, content-hashed split files, row = state (`ARCH:517-567`) |
| P2 | Labeller point lookup by cache key, paid call on miss | 58k calls at MVP (DERIVED 2.1); reruns must hit 100% (`REL:593`) | one row | primary key computed before the call, single writer |
| P3 | Build: join questions with cache rows by (question_id, teacher, perm), pool, split by group, dedup | 10 to 20 builds (ASSUMPTION) | full scan | deterministic pure function of (sources, cache snapshot, config) |
| P4 | Eval: dev per checkpoint, cal to fit T, test at most 3 reads per release (`REL:719-720`), sliced by type x length x source | per checkpoint | one split, group-by | stored eval permutations, bucket and source columns |
| P5 | Generation: seeded latent specs, LLM states, LLM questions | ~7.1k calls at MVP (DERIVED 2.1) | append | same cache, `kind = generate` |

**Why:** everything except the paid-call cache is a whole-split scan of
immutable data, so files plus one SQLite table serve every pattern.

---

## 2. Sources and counts

### 2.1 MVP (train 15k, `REQ:280`; 40% public / 60% teacher-labelled, `REQ:282`)

| Source (licence, `SEC:227-238`) | Conversion | Train dec. | d | States | State tokens (ASSUMPTION) |
|---|---|---|---|---|---|
| HotpotQA (CC-BY-SA-4.0) | yes/no items -> Noul; span answers -> Choice over context entities (4 to 8) | 2,000 | 2 | 1,000 | 1,200; half padded to 2k to 4k |
| BoolQ (CC-BY-SA-3.0) | Noul | 1,000 | 1 | 1,000 | 150 |
| SQuAD v2 (CC-BY-SA-4.0) | "answerable?" Noul + answer Choice | 800 | 2 | 400 | 200 |
| MNLI `genre != fiction` (`SEC:231`) | Choice, 3 labels | 800 | 1 | 800 | 60 |
| CommonsenseQA (MIT) | Choice, 5 | 700 | 1 | 700 | 50 |
| ARC (CC-BY-SA-4.0) | Choice, 3 to 5 | 700 | 1 | 700 | 80 |
| Synthetic, 8 workflows (3) | 5 kept of ~10 generated | 9,000 | 5 | 1,800 | 350 to 3,000 |
| Real text, GitHub PRs/issues (pointers, `SEC:744-747`) | 5 per state | 0 (1,000 in test_ood) | 5 | 200 | 100 to 16k (`RN:138`) |
| **Total** | | **15,000** | **2.3** (15,000 / 6,400) | **6,400** | |

MMLU-Pro (MIT) is **eval-only at MVP** (1,000-item gold slice, `REQ:249`);
target may add 1,500 train items after 6.4 is proven. **Why:** it is nearly
all test split and overlaps pngwn's test (`SEC:284-291`), while CSQA and ARC
teach the same knowledge-Choice format with no leakage risk.

Eval splits: **dev 1,000** (checkpoint selection, `REL:443`); **cal 2,000**
(stratified on type x length bucket: 12 strata x 167, DERIVED, above the 100
floor in `REL:530`); **test_id 4,000** (>= 3,000, `REQ:285`); **test_ood 2,000**
(one held-out workflow 600, one held-out template per other workflow 400, real
text 1,000); **mmlu_pro_gold 1,000**; **pngwn_anchor** 5,214 (MEASURED,
`RN:97`), never stored in our dataset; **audit 250** from eval splits only.

```
teacher-labelled decisions per teacher (DERIVED)
  = synth train 9,000 + public-train check subset 1,000 + dev 1,000 + cal 2,000
  + test_id 4,000 + test_ood 2,000 + mmlu_pro_gold 1,000            = 20,000
calls per teacher = 20,000 x (0.55 x 1 + 0.45 Choice share x 2 perms) = 29,000
DS V4.1 Flash 29,000 x 0.00008 = 2.3 USD ; Qwen3.6-35B 29,000 x 0.00014 = 4.1 USD
labels 6.4 USD (4.4 to 9.3)          per-call prices COST:290-291
generation: synth decisions all splits 9,000 + 0.6 x 7,000 + 1,000 = 14,200
  / 5 = 2,840 kept / 0.8 keep rate = 3,550 states generated
  3,550 x 0.0008 (state, COST:181) + 3,550 x (1,200 x 0.10e-6 + 800 x 0.50e-6) = 2.8 + 1.8 = 4.6 USD
```

Choice share 45% and keep rate 0.8 are ASSUMPTION (35 to 55%, 0.7 to 0.9).

**Target** (train 40k, `COST:147`): public 14k, synthetic 23k, real 3k; dev
1.5k, cal 3k, test_id 5k, test_ood 3k. Teacher decisions 23k + 3k + 2k + 1.5k
+ 3k + 5k + 3k + 1k = 41.5k; calls x 1.45 = 60k; x 0.00022 = **13.2 USD
cumulative** (DERIVED).

### 2.2 Decisions per state

**Decision:** synthetic d = 5 (cap 8 per state in train), public 1 to 2.
**Why:** higher d is cheaper because the state is a cached API prefix
(`COST:290`) and a shared prefix in packed training (`CAP:262-278`), and the
cap keeps one state's correlated targets from dominating a batch.

```
naive tokens per epoch (state + 40-token tail), DERIVED from 2.1:
  <=128   2,200 x 105                                  = 0.23M
  128-512 1,000 x 190 + 800 x 240 + 1,500 x 390        = 0.97M
  512-2k  1,000 x 1,240 + 4,500 x 1,140                = 6.37M
  2k-4k   1,000 x 3,040 + 3,000 x 3,040                = 12.16M
  total ~19.7M/epoch (2 epochs 39M); shared-state packing ~6.0M (3.3x less)
MVP run naive, Qwen3-4B at 5k tok/s (ASSUMPTION, COST:168): 39M / 5,000 = 2.2 h = 3.5 USD
target naive: 40k x 2,032 = 81M/epoch, 163M / 5,000 = 9.0 A100-h > 8 h cap (REQ:340)
```

Per-decision expansion (`ARCH:560-563`) is fine at MVP; **tree packing is
required at target**, meeting the revisit trigger in `ARCH:821`.

---

## 3. Question generation

**States.** Code samples a latent spec (workflow, template_id,
entity_bundle_id, facts like `severity=3`, target length bucket) with a seeded
RNG; DeepSeek V4.1 Flash renders a consistent state. Workflows: support
tickets, security alerts, invoices, agent traces, pull requests, email
threads, incident timelines, purchase orders; ~12 templates each
(ASSUMPTION); no workflow above 20% of synthetic decisions. States above 4k
tokens are assembled from LLM segments plus code-templated log lines. All
states pass the SEC-9 scrub (`SEC:733-748`). **Why:** latent facts give a
known answer for some questions (an inversion check, `REL:405-407`), and
DeepSeek's terms allow publishing outputs (`SEC:309`).

**Questions.** A second call writes ~10 candidates (about 4 Noul, 4 Choice,
2 Score): one predicate each, answerable from the state alone, no reference to
letters or positions, no "all/none of the above". Choice options come from a
versioned per-workflow **option bank** plus distractors; counts 2 to 4 (50%),
5 to 8 (35%), 9 to 16 (10%), 17 to 26 (5%) (ASSUMPTION); options <= 40 tokens.
Score picks a `scale_id` from a versioned library (severity 0..4, likelihood
0..4, quality 0..9 for 10% of Score) with an anchor sentence per level,
rendered per `ADR3:41-42`. An optional `intended_option_id` is kept as a weak
signal. **Why:** a fixed scale library makes "level 3" mean the same thing
across states, which an ordinal calibrated readout needs.

**Degenerate filters.**

| Filter | Rule | Stage |
|---|---|---|
| Compound / positional | regex for joined predicates in Noul; any letter or "first option" reference | pre-label, reject |
| Duplicate options | normalized equal or MinHash >= 0.9 within a question | pre-label, reject |
| Verbatim answer | exactly one option appears verbatim in state: tag `lookup_candidate`, cap 15% per workflow (ASSUMPTION) | pre-label |
| 1:1 field lookup (pngwn `team`, `RN:100`) | per template, >= 20 states: H(consensus argmax given one `key: value` field) < 0.05 bits -> drop template | post-label |
| Shallow leak | bag-of-words LR >= 0.98 or majority >= 0.95 per template -> flag, report with and without (`REL:437-440`) | post-label |
| State not needed | 10% per template labelled without state; agreement >= 0.9 -> drop template | post-label |
| Unanswerable | both teachers normalized entropy > 0.95, or generator marked "needs outside info" | post-label, drop |
| Spec infidelity | spec answer vs consensus < 80% for a template -> drop template | post-label |

---

## 4. Soft labels

**Per call:** nanohunch-fmt-v1 text for one question in one option order,
thinking off, `temperature=0`, `max_tokens=1`, `logprobs=true`,
`top_logprobs=20` (maximum ASSUMPTION; verify in P0-7, `CAP:734`). Reading:
(1) strip whitespace from returned tokens, map to candidate labels, merge
variants (`A`, ` A`) by logsumexp; (2) `candidate_mass` >= 0.9 accept, 0.5 to
0.9 accept and flag, < 0.5 retry once with a strict prompt variant then drop
this teacher record (the other teacher stays at half weight, flag
`single_teacher`); (3) each candidate missing from top-k (certain at n > 20)
gets `min(p_kth, residual / n_missing)`, counted in `tail_imputed`; (4)
renormalize over candidates, store **by option id**. **Why:** first-token
logprobs give a full distribution in one call (`COST:396`), and the k-th
probability is a hard upper bound on anything outside the top k.

**Permutations:** Choice with n >= 3 gets two calls, canonical and reversed;
Score and Noul one call (never permuted, `ARCH:303-316`). The two orders are
pooled **log-linearly** per teacher, as in the engine (`ARCH:322-331`); each
teacher's flip rate is reported (`REL:514-516`). **Why:** reversal moves every
option, cancelling first/last position bias for about +2.0 USD at MVP
(DERIVED: 0.45 x 20,000 extra calls x (0.00008 + 0.00014) = 1.98).

**Consensus:** arithmetic mean of the two pooled teacher distributions
(`REQ:222-225`, `ARCH:554`). Disagreement by Jensen-Shannon divergence (base 2):
JSD <= 0.1 weight 1.0; 0.1 to 0.3 weight 0.5; > 0.3 (`REL:562`) weight 0.25,
reported and oversampled into the audit; confident conflict (both max p >= 0.8,
different argmax) weight 0, excluded from headline metrics until audited.
Expected disagreement 10 to 20% (ASSUMPTION). Gold items train on
`lambda x onehot(gold) + (1 - lambda) x consensus` or one-hot gold
(`ARCH:557-558`). **Why:** the mean of probabilities is the published
reference definition, so accuracy stays comparable to Jev; log-linear is used
only within one teacher, where the goal is cancelling bias.

---

## 5. Schemas and storage

### 5.1 Engine per data class

| Class | Engine | Rejected, and why |
|---|---|---|
| Paid-call cache | **SQLite, WAL, single writer, `PRIMARY KEY(cache_key)`** (`REL:594-595`) | JSONL append: interleaved writes, no key; Postgres: a server for one writer |
| Canonical dataset | **JSONL, sorted keys, fixed float format, zstd**, one row per state | Parquet canonical: bytes vary across pyarrow versions, so the content hash drifts (`REL:679`) |
| Published export | **Parquet**, derived | JSONL on Hub: fine, weaker viewer and types |
| Eval results | JSONL per (checkpoint, split) + JSON report | metrics DB: one user; DuckDB over files for ad hoc (two-way) |
| Queue | **none**: work = planned keys minus `ok` rows | a queue: one process at 4 to 8 concurrency (`REL:147-151`) |
| Dedup index | in-memory MinHash LSH per build | vector DB: ~12k states |
| Closed-model reference, B1 outputs | separate `data/baselines/`, never imported by `train.py` (`SEC:349-351`) | shared table: one bad join violates SEC-1 |

**Why:** the only concurrent writer is one labeller, so SQLite gives key-based
idempotency and crash-safe transactions with nothing to operate.

### 5.2 Cache table `calls`

`cache_key` = sha256(canonical_json({kind, teacher_id, provider,
prompt_version, format_version, params, prompt})) PK; `kind` (`label`,
`generate`, `state_free_probe`); `teacher_id` (requested string); `provider`
(pinned, `SEC:352-354`); `model_returned` (must equal the session pin,
`REL:128-137`); `request_params` (JSON, no headers); `prompt_sha256`;
`prompt`; `response_body` (raw JSON body); `usage`; `status` (`in_flight`,
`ok`, `error_retryable`, `error_final`); `created_at`, `completed_at` (UTC).
**Forbidden** (`SEC:516-519`): headers, `Authorization`, keys, org/project ids,
pickled SDK objects, `os.environ`; a write-time check rejects values matching
`SEC:525-526` regexes. Rows are written `in_flight` before the call
(`REL:592`); `in_flight` older than 10 min is re-issued, so a crash re-pays at
most 4 to 8 calls (~0.001 USD, DERIVED). Backup to a private HF dataset every
1,000 rows or 24 h (`REL:201`).

### 5.3 Canonical row `nanohunch.train.v1` (amended, see 12 D1)

```json
{"schema_version": "nanohunch.train.v1", "format_version": "nanohunch-fmt-v1",
 "state_id": "sha256(normalized state)", "group_key": "synth:invoices:tpl07:eb0213",
 "split": "train", "state_text": "...", "state_len_chars": 5210,
 "state_len_tokens": {"qwen3@906bfd4b": 1284}, "length_bucket": "512-2k",
 "source": {"dataset": "synthetic", "source_revision": "gen-spec-v1@<git sha>",
   "source_license": "cc-by-sa-4.0", "workflow": "invoices", "template_id": "tpl07",
   "generator": "deepseek/deepseek-v4.1-flash", "redaction_counts": {"email": 1}},
 "decisions": [{
   "question_id": "sha256(state_id, q_type, text, sorted option ids)",
   "q_type": "choice", "question_text": "Which cost center should this be charged to?",
   "options": [{"id": "o_3fa9c2d1", "text": "Marketing"}, {"id": "o_91b0e7aa", "text": "R&D"}],
   "canonical_order": ["o_91b0e7aa", "o_3fa9c2d1"], "scale_id": null,
   "teachers": [{"teacher_id": "deepseek/deepseek-v4.1-flash", "provider": "deepseek",
     "model_returned": "...", "date": "2026-10-02", "prompt_version": "label-v1",
     "method": "logprobs_top_k", "top_k": 20,
     "perms": [{"order": ["o_91b0e7aa", "o_3fa9c2d1"], "cache_key": "...",
       "probs": {"o_91b0e7aa": 0.83, "o_3fa9c2d1": 0.17}, "candidate_mass": 0.97, "tail_imputed": 0}],
     "pooled": {"o_91b0e7aa": 0.82, "o_3fa9c2d1": 0.18}}],
   "consensus_probs": {"o_91b0e7aa": 0.80, "o_3fa9c2d1": 0.20}, "jsd": 0.02,
   "weight": 1.0, "flags": [], "gold_option_id": null, "gold_kind": null,
   "label_origin": "teacher", "eval_perms": null}]}
```

Rules: option id = `o_` + 8 hex of sha256(normalized text), unique per
question (asserted); Score ids `v0..v9`, Noul `yes`/`no`. `canonical_order` is
a seeded shuffle at ingest (seed = question_id), so a source's fixed gold
position never becomes canonical (`REL:508-512`). `gold_kind` in {`dataset`,
`spec`, `human`}; `label_origin` stays {`gold`, `teacher`} so the SEC-1 assert
is unchanged (`SEC:346-348`). Index arrays exist only inside the loader.
`state_len_tokens` is keyed by tokenizer revision. **Why:** a label stored as
an id survives any reordering, which is the bug pngwn shipped (`REL:385-392`).

### 5.4 Layout

`data/` (gitignored): `cache/calls.sqlite`; `gen/specs.jsonl`;
`builds/<version>/` with `manifest.json` (file sha256s, counts, seeds, cache
snapshot), `{train,dev,cal,test_id,test_ood,mmlu_pro_gold}.jsonl.zst`,
`reports/{lengths,balance,dedup,agreement,groups}.json`; `audit/labels.jsonl`;
`baselines/`; `erasures.jsonl`. `release/<artifact>/` is staging, written only
by `release/build.py` (`SEC:522-527`). Hub: `<user>/nanohunch-labelcache`
(private), `<user>/nanohunch-decisions` (public, configs `permissive` and
`share_alike`, `SEC:279-283`). Size: ~12k states x 1,300 tokens x 4 chars =
~62 MB text per build (DERIVED), ~15 MB zstd (ASSUMPTION 4x); cache ~250 to
300 MB (`REQ:288`).

---

## 6. Splits (the one-way door)

**Split key = `group_key`, assigned by `hash(group_key, split_seed)` into
fixed fractions within each source stratum;** `state_id` disjointness is also
asserted, so `REQ:359` holds as a corollary.

| Source | group_key | Cardinality (MVP) | Skew |
|---|---|---|---|
| HotpotQA | union-find over item `_id` and paragraph titles | ~1,000 items, components 1 to 3 (ASSUMPTION) | popular titles merge items; components > 20 items dropped |
| BoolQ / SQuAD v2 | page title / article title | ~1,000 / ~200 | at most 4 questions per SQuAD article |
| MNLI | `promptID` | ~800 | one hypothesis per premise |
| CSQA, ARC, MMLU-Pro | question id | 1 each | none |
| Synthetic | `(workflow, template_id, entity_bundle_id)` | ~2,800 groups of 1 to 2 states | none by construction |
| Real text | repository | 10 to 20 (ASSUMPTION) | large repos dominate; all in test_ood at MVP |

Padded HotpotQA states use paragraphs **only from items already in the same
split**. **Why this key, and why one-way:** models memorize paragraphs,
templates and entity bundles, not state hashes, so a state-only split leaks
(`REL:416-423`); hashing each group independently means new data never moves
an existing group, and changing the key later reshuffles every split and
invalidates every past eval (`ARCH:835`).

**Hot key analog:** one HotpotQA component, SQuAD article or template holding
a large share of a split, so the split measures one topic. The build fails if
any group > 1% of a split's decisions (test_id: 40, where a synthetic group
holds <= 16) or any template > 5% of test_id (96 templates average ~1%)
(thresholds ASSUMPTION); `reports/groups.json` lists the top 20 groups per
split. Split-fraction noise with ~8k groups: sd = sqrt(0.18 x 0.82 / 8,000) =
0.43 pts for test_id (DERIVED).

**Held-out domain:** test_ood = one whole workflow (seeded draw, recorded) +
one whole template per other workflow + all real text; none of their groups,
templates or option banks appear in train. **Why:** test_id shares templates by
design, so only a held-out workflow shows whether the model reads states or
learned our generator.

**Leakage scans (T1, `REL:376`):** (1) exact `state_id` and `question_id`
disjoint; (2) MinHash, 5-gram word shingles, 128 perms, LSH, Jaccard >= 0.8
(`REL:432-434`), fail if > 0.5% of test states have a train near-duplicate;
before the first paid run offending groups are reassigned, after it the test
item is dropped; 0.6 to 0.8 synthetic pairs reported; (3) **SEC-14:** drop from
train, dev and `mmlu_pro_gold` every MMLU-Pro `question_id` in pngwn
`test_meta.jsonl` / `cal_meta.jsonl` (`SEC:289-290`) plus MinHash >= 0.8 on
question text; (4) **the same for HotpotQA**, pngwn's Noul source
(`RN:98-99`); (5) pngwn anchor states vs our train: any hit >= 0.8 fails.

---

## 7. Length distribution

Train by decisions (DERIVED, 2.1): `<=128` 14.7%, `128-512` 22%, `512-2k`
36.7%, `2k-4k` 26.7%, `4k-8k` 0% (MVP trains at 4,096, `REQ:270`), `8k+` eval
only. Target (8,192): 10 / 20 / 35 / 20 / 15%, with `4k-8k` from long traces,
timelines and real PRs. Fill: MNLI, CSQA, ARC -> `<=128`; BoolQ, SQuAD, short
tickets -> `128-512`; HotpotQA, most synthetic -> `512-2k`; padded HotpotQA,
traces, timelines -> `2k-4k`+; real PRs span all. Each eval split has >= 150
decisions per bucket above 2k (ASSUMPTION) so S4 gets a CI. Build fails if
labelled p90 per type < 0.9 x requested p90 (`REL:230-232`); truncation removes
state tokens only (`REL:577-579`). **Why:** pngwn's fine-tune lost to its base
above 2k because it trained at 384 tokens (`RN:140-142`), so length is a
controlled input here.

---

## 8. Option order

Storage: canonical order is the seeded ingest shuffle (5.3). Train: each epoch
a Choice question gets `perm` from `seed(epoch_seed, question_id)`;
`display_ids[j] = canonical_order[perm[j]]`, `target[j] =
consensus_probs[display_ids[j]]` looked up by id, then converted to an index
array; Score and Noul are not permuted. Eval: every eval Choice row stores
`eval_perms` explicitly (identity, reversed, 3 seeded random orders as id
lists, `REQ:234-237`), so an RNG library change cannot alter the test. Checks:
property test `target_perm[j] == target[perm[j]]` (`REL:401-404`); consensus
argmax position histogram per type, chi-square vs uniform, fail at p < 0.01
(`REL:512-514`). **Why:** augmentation is how Candidate B reaches the order
target (`ARCH:365`) and the likeliest place to invert a label, so the remap
goes through ids and is property-tested.

---

## 9. Quality

**Human audit (S6, `REQ:417`).** 250 decisions from cal, test_id, test_ood,
stratified by q_type (3) x source class (public, synthetic, real) x agreement
(agree, JSD > 0.3), disagreement oversampled 2x and reweighted by inverse
sampling probability. The author sees state, question and options in a random
order, no teacher output, and writes `{question_id, auditor, chosen_option_id,
confidence 1..3, flags (ambiguous, unanswerable, leaked, bad_options),
shown_order, ts}` to `audit/labels.jsonl`; 50 items are re-shown a week later
for self-agreement. At 85% agreement the 95% CI is 1.96 x sqrt(0.85 x 0.15 /
250) = +/- 4.4 pts overall, +/- 7.7 pts per type at n = 83 (DERIVED). ~4 h
(`COST:185`).

| Check (thresholds ASSUMPTION unless cited) | Target | If missed |
|---|---|---|
| Teacher argmax agreement overall / Noul / Choice | >= 80% (`REQ:449`) / 85% / 80% | review templates; drop < 70% |
| Score exact / within 1 level | >= 60% / >= 90% | revise scale anchors |
| Each teacher vs gold (BoolQ, HotpotQA, MNLI, MMLU-Pro) | within 10 pts of the other | down-weight that teacher for that source |
| Consensus vs audit | >= 80% MVP (`REQ:417`) | < 75%: teacher mix changes (`ARCH:406`) |
| Balance: Noul yes-rate; Score levels; q_type mix | 35 to 65% per workflow; no level > 50%, each >= 5%; Choice 45 / Noul 35 / Score 20 (+/- 5) | resample or adjust quotas |
| Canary drift | >= 95% on 50 fixed prompts per session (`REL:134-137`) | stop session |

**Why:** two LLMs can be confidently wrong together, and only gold and human
labels sit outside their label path (`REL:239-243`).

---

## 10. Versioning, release, retention, consistency

**Version = sha256 of the canonical manifest** (sorted `(split, file sha256)`,
schema and format versions, seeds, cache snapshot `(max rowid, count)`); the
builder git sha is recorded, not hashed. Tags `v1` (MVP), `v2` (target).
Rebuild with the same inputs gives the same hash (no-op); publish pushes only
if the Hub tag's hash differs.

**Published:** train, dev, cal, test_id and synthetic test_ood as Parquet in
configs `permissive` / `share_alike` with per-row licence, the AI-generated
notice (`SEC:359-361`) and a canary GUID for contamination detection. Real
text as pointers `(repo, number, commit_sha)` + labels + hydration script
(`SEC:744-747`); `mmlu_pro_gold` as ids + labels. **Eval-only, never
published:** pngwn anchor (aggregates only, `SEC:270`), deny-listed ids
(`SEC:263-266`, build fails), closed-model reference and B1 outputs (`SEC:40`,
`SEC:315`), the call cache (private backup).

**Growth bounds.** Cache grows only with paid calls, so the budget bounds it:
~60k to 120k rows, 0.3 to 0.6 GB (DERIVED, 2.1 and `REQ:288`), kept forever
(`REQ:292`). Builds: last 3 plus tagged. Eval reports ~1 MB per checkpoint.

**Erasure** (`SEC:750-752`): append `state_id` to `erasures.jsonl` (build fails
if an erased id appears); set `prompt` and `response_body` to NULL on its cache
rows but keep the key, so it is never re-bought; rebuild; squash Hub history.
Done within one build.

**Consistency.** A build writes to `builds/.tmp-<rand>/`, writes
`manifest.json` last, then renames; readers ignore directories without one.
The builder refuses while any `in_flight` row exists (read-after-write). The
Hub cache backup is eventually consistent, lagging <= 1,000 rows or 24 h; a
restore sees pushed rows only and re-buys the rest (<= 1,000 x 0.0001 = 0.1
USD, DERIVED). Jobs refuse a manifest hash mismatch (`REL:441-442`).

**Idempotency keys:** calls `cache_key`; specs `(spec_seed, index)`; build
version hash; publish tag; training `config_hash` (`REL:681`); audit
`(question_id, auditor, pass)`; erasure `state_id`. Dedupe window for paid
calls: **forever**, since the retry window is any rerun in the project's life
and the store is small. Migration (item 9): not applicable in GREENFIELD; v1
to v2 is additive and existing groups never move.

---

## 11. Failure modes and automated checks

| Failure | Mechanism here | Check (stage) |
|---|---|---|
| Label inversion | options shuffled, target not; letter parsed against wrong order | id-keyed labels, text round trip, perm property test, spec Bayes > 1/n + 3 SE, slice below chance fails (`REL:398-409`) (T0, T1, T4) |
| Leakage | shared paragraph, template, bundle; pngwn / MMLU-Pro / HotpotQA overlap | group disjointness, MinHash, SEC-14 extended (T1) |
| Hot group | one component or template dominates a split | > 1% group, > 5% template fails (T1) |
| Lost update / double pay | crash between response and write | `in_flight` first, PK, weekly ledger vs provider usage (`REL:592`) |
| Cache miss by construction | unseeded shuffle or timestamp in prompt | render-twice hash test (T0), dry-run hit rate >= 99% (T2) |
| Teacher drift / logprobs gone | alias repointed, reasoning on | `model_returned` pin, canary, `method` constant per type (`REL:143-145`) |
| Off-format output | mass on non-label tokens | `candidate_mass` rule; fail if > 5% of a teacher's records dropped (ASSUMPTION) |
| Length bias from retries | long states time out | labelled vs requested p90 (T1) |
| Non-releasable data or teacher | ANLI row, closed-model label | deny-list and allowlist, fail closed on missing fields (`SEC:762-766`) |
| Secrets / PII | SDK object serialized, pasted keys | allowlisted schema, write-time regex, scrub drops on error, trufflehog at release (`SEC:516-527`, `SEC:771`) |
| Silent corruption | truncated or edited file | per-file sha256 in manifest, checked by every job (T2) |
| Unbounded growth | cache or builds pile up | budget-bounded cache, keep-3 builds, stale sweep |
| Irreversible change | split key or id scheme changed after a paid run | frozen by `schema_version`; change means new version plus full re-eval |

---

## 12. Disagreements

- **D1.** `ARCH:531-543` stores options as a list and targets as index arrays
  (`"gold": 2`). Here every distribution and gold is keyed by option id with an
  explicit `canonical_order`, because `REL:398-400` forbids bare indices and
  pngwn's bug was an index meaning the wrong option. No row exists yet, so the
  name `nanohunch.train.v1` is kept; `ARCH` 6.3 should be updated.
- **D2.** D5 "split key = `state_id`" (`ARCH:835`, `ARCH:555-556`) becomes
  `group_key`, with `state_id` disjointness still asserted, as `REL:428-431`
  already requires.
- **D3.** SEC-14 (`SEC:284-291`) is extended to HotpotQA (`RN:98-99`).
- **D4.** d = 2.3 overall (`CAP:47` d = 2, `COST:151` d = 3); 3.55k MVP states generated, not 5.5k.
- **D5.** The length mix gives ~19.7M naive tokens per epoch, not 10M
  (`COST:155`): MVP still fits (3.5 USD per run), target needs tree packing.

## 13. Open questions

1. DeepSeek **V4.1** Flash under the V4 Flash terms (`SEC:309`), in `RELEASABLE_TEACHERS` (`SEC:343`)? UNVERIFIED.
2. Qwen3.6-35B-A3B via OpenRouter: host, precision, host terms (UNVERIFIED, `SEC:311`)? Fallback: vLLM self-host (`CAP:484-490`).
3. Provider `top_logprobs` maximum (ASSUMPTION 20) with thinking disabled, for both teachers.
4. May scrubbed real text go to API teachers under `SEC:729-732`, or wait for a self-hosted teacher?
5. HotpotQA title-component sizes in our sample (ASSUMPTION 1 to 3).
