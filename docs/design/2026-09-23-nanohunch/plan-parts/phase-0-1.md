### Phase 0: Set up the repo, pin references, measure the five unknowns

**Goal:** a committed repo with secrets ignored from commit 1, both base models on disk, and `reports/phase0.md` holding a measured value, a pass/fail and a decision for P0-1, P0-4, P0-6, P0-7 and P0-8.
**Effort:** 14 h (2.3 engineer-days). **Depends on:** the plan-level prerequisites. **Parallel with:** nothing (solo); inside the phase, the unattended 2 h soak (step 12) overlaps with the teacher probe (step 11), which uses only the network.
**Risk:** medium, because two results (P0-7, P0-8) can move Phase 4 and Phase 5 onto their fallbacks.

**Why this phase exists.** Five numbers the design leans on are ASSUMPTION or were measured on synthetic weights: label tokens being single ids, the branch oracle on real weights, an unexplained 0.12 fp32 gap, teacher logprob stability through OpenRouter, and Mac LoRA speed on real weights. Each one, if wrong, changes a later phase. Measuring them costs about a week and under 0.20 USD; discovering them in week 6 costs a redesign (R8, R9, R10, R18). Planning also found a sixth, already MEASURED fact: stock `mlx_lm.lora` wraps every `{"prompt","completion"}` row in the tokenizer's chat template (`mlx_lm/tuner/datasets.py:107-127`, mlx-lm 0.31.3), which would silently break the raw nanohunch-fmt-v1 format. Step 4 neutralises it.

**What you will understand after this phase**
- **Restricted label readout.** The answer is not generated text: you take the logits at the last token of `Answer:`, keep only the rows for ` A`, ` B`, ` yes` and so on, and softmax those. This only works if every label is exactly one token, which is why P0-1 comes first.
- **KV-cache branching and its oracle.** Encoding the state once and appending each question to a copy of its KV cache must give the same probabilities as encoding state plus question from scratch. In bf16 the two differ by rounding (1.3e-2 MEASURED), so correctness is checked in fp32 (4.0e-4 MEASURED on random tokens), and "isolation" checks that batching branches never lets one row influence another.
- **Logprob repeatability.** A teacher's top-20 logprobs are a soft label only if the same request gives the same distribution twice. Jensen-Shannon divergence (JSD, base 2, range 0 to 1) is the run-to-run distance; candidate mass is how much probability lands on valid labels at all.
- **Sustained versus burst throughput.** A 30-iteration benchmark measures the chip cold; a 2 h soak measures thermals, allocator growth and sleep. Only the soak predicts an overnight run.

**Changes**
| File | Change |
|---|---|
| `.gitignore` | **Done 2026-09-24.** GitHub Python template (initial commit, already ignores `.env` and `.venv`) plus a nanohunch block: `/data/`, `/runs/`, `/refs/`, `*.safetensors`, `.DS_Store`. |
| `.env.example` | **Done 2026-09-24.** Two lines: `OPENROUTER_API_KEY=` and `HF_TOKEN=`. |
| `AGENTS.md` | **Done 2026-09-24.** Learn-by-writing rule, reference-repo rule, line budget, pre-push check. |
| `ledger/hours.csv` | **Done 2026-09-24.** Header `week_start,phase,hours,note`. |
| `ledger/spend.csv` | **Done 2026-09-24.** Header `date,phase,item,model,host,quantization,prompt_tokens,completion_tokens,usd,note`. |
| `pyproject.toml`, `uv.lock`, `.python-version` | New, from `uv init --bare` and `uv python pin 3.12` (`README.md` already exists and is kept). |
| `bench/__init__.py`, `skeleton/__init__.py` | New, empty, so scripts run as `uv run python -m bench.<name>`. |
| `bench/equiv.py`, `bench/synth_bench.py`, `bench/branch_bench.py` | Copied unchanged from `docs/design/2026-09-23-nanohunch/capacity-bench/`. |
| `bench/check_labels.py` | New (P0-1). Code in step 5. |
| `bench/make_raw_model.py` | New. Builds a model dir with a passthrough chat template. Code in step 4. |
| `skeleton/fmt_ref.py` | New. Reference nanohunch-fmt-v1 text fill (ADR-0003), shared by bench and skeleton. Code in step 6. |
| `tests/test_fmt_ref.py` | New. 3 tests (step 6). |
| `bench/sample_items.py` | New. 25 BoolQ + 25 ARC-Challenge items to `data/raw/p0_items.jsonl`. |
| `bench/equiv_real.py` | New (P0-4), adapted from `bench/equiv.py`. |
| `bench/anomaly.py` | New (P0-6), hypothesis toggles on `bench/equiv.py` line 20. |
| `bench/teacher_probe.py` | New (P0-7). Glue, core parts in step 11. |
| `configs/teachers_p0.yaml` | New. One entry per teacher: `teacher_id`, `model`, `provider_order`, `quantization`. |
| `bench/make_synth_lora.py`, `bench/soak_parse.py`, `configs/p0_lora.yaml` | New (P0-8). |
| `reports/phase0.md`, `reports/phase0/teacher_probe.csv` | New. Results table and per-call probe rows. |
| `tools/loc.py` | New, glue (code in step 14). Counts core lines against the 1,000-line budget. |
| `configs/refs.yaml` | New. Pinned reference repos (step 14). `refs/` is gitignored. |

**Produces (interfaces later phases use)**
- `skeleton/fmt_ref.py`: `PREAMBLE: str`, `prefix_text(state: str) -> str`, `branch_text(qtype: str, question: str, options: Sequence[str], values: Sequence[int] = ()) -> str`, `label_strings(qtype: str, n: int, values: Sequence[int] = ()) -> list[str]`. Phase 2 `fmt.render` must produce the same token ids as `tok.encode(prefix_text(s)) + tok.encode(branch_text(...), add_special_tokens=False)`.
- `runs/models/minicpm5-2b-base-raw/` and `runs/models/qwen3-4b-base-raw/`: symlinked weights, passthrough chat template. Every stock `mlx_lm` command in Phase 1 uses these paths.
- `configs/teachers_p0.yaml`: fields named exactly as `label.TeacherSpec` (`teacher_id`, `model`, `provider_order`) plus `quantization`. Phase 4 loads it.
- `bench/teacher_probe.py`: `candidate_mass(top: list[dict], labels: list[str], qtype: str) -> tuple[float, list[float]]` and `jsd(p: list[float], q: list[float]) -> float`, the reference Phase 4 `label.py` is tested against.
- `reports/phase0.md`: the values Phase 2 (oracle tolerance, batch policy), Phase 4 (teacher set, hosts) and Phase 5 (training location) read.

**Decision table (what each measurement changes)**
| ID | Pass rule | If it fails | Phases that change |
|---|---|---|---|
| P0-1 | all 38 labels single-token in both tokenizers, alone and after `Answer:` | amend the ADR-0003 label set (for example ` Yes`/` No`, or letters for Score) and rerun, before any other step | 1, 2 |
| P0-4 | fp32 unbatched max abs prob diff <= 1e-3 and isolation diff == 0.0 | 1e-3 to 1e-2: S10 tolerance becomes 2x measured, noted in `reports/phase0.md`; above 1e-2: a cache-copy bug, fix before Phase 2; isolation != 0: `MLXBranchScorer.score` runs branches one at a time | 2 |
| P0-6 | cause found within 2 h | accepted (R9): batch 1 plus gradient accumulation in training, no batched re-encode anywhere, NLL parity test in Phase 5 | 2, 5 |
| P0-7 | per teacher: mean candidate_mass >= 0.9 and mean JSD < 0.01 | one passes: single teacher plus gold labels (R10); none passes: gold-only training and generator-derived gold for synthetic spec-fact questions | 4 |
| P0-8 | median tok/s after minute 30 >= 200 at 2k, no OOM, drop <= 20%, peak memory growth <= 0.5 GB | the Modal later item moves forward to Phase 5 (R7, R18); the Mac does eval only | 5, 6 |

**Steps**
1. (1.5 h) Repo hygiene. **Already done on 2026-09-24** in `/Users/umangkaushik/fun/nanohunch` (remote `ubermenchh/nanohunch`, public): `.gitignore`, `.env.example`, `AGENTS.md`, both ledger headers, README, and these design docs under `docs/design/2026-09-23-nanohunch/`. Your part: read `AGENTS.md` and confirm it states the rules you want (it is the contract any AI pair reads first). Then `cp .env.example .env`, fill both keys, and run `git check-ignore .env data/a runs/a refs/a w.safetensors`; expect 5 lines echoed back. Pre-push check, run before every push: `git grep -nE "sk-or-v1-|hf_[A-Za-z0-9]{30,}"`; expect no output. Use the saved time to read `docs/design/2026-09-23-nanohunch/README.md` and `risks.md` once.
2. (0.5 h) `uv init --bare --python 3.12 --name nanohunch && uv python pin 3.12` (`--bare` writes only `pyproject.toml`, so the existing README and `.gitignore` are untouched), then `uv add mlx "mlx-lm>=0.31.3" numpy pyyaml httpx python-dotenv datasets matplotlib` and `uv add --dev pytest ruff`. Run `uv run python -c "import mlx.core as mx, mlx_lm; print(mlx_lm.__version__, mx.default_device())"`. Expect `0.31.3 Device(gpu, 0)` or a newer version. Copy the three capacity-bench scripts into `bench/`, add the empty `__init__.py` files, commit `chore: uv project and bench scripts`.
3. (0.5 h) `hf --version` (MEASURED 1.30.0 at `~/.local/bin/hf`; if missing use `uv run huggingface-cli download`). Run `hf download openbmb/MiniCPM5-2B-Base` and `hf download Qwen/Qwen3-4B-Base`; each prints its snapshot path. Then `uv run python -c "from mlx_lm import load; load('openbmb/MiniCPM5-2B-Base'); print('ok')"`. Expect `ok`. If it raises `ValueError: Model type ... not supported`, stop and record it as a BLOCKING row in `reports/phase0.md`: ADR-0001 then promotes Qwen3-4B-Base and every later MiniCPM path switches to it. Record `uv run python -c "import mlx.core as mx; print(mx.device_info())"` (older mlx: `mx.metal.device_info()`); expect `max_recommended_working_set_size` near 19.07e9 (A15, MEASURED).
4. (0.5 h) Write `bench/make_raw_model.py`:
   ```python
   """Model dir with a passthrough chat template. Stock mlx_lm.lora applies the chat template to
   every prompt/completion row (mlx_lm/tuner/datasets.py:107-127); this makes it prompt + completion."""
   import json, sys
   from pathlib import Path
   from huggingface_hub import snapshot_download
   repo, out, bos = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "bos"
   src = Path(snapshot_download(repo, local_files_only=True))
   out.mkdir(parents=True, exist_ok=True)
   for f in src.iterdir():
       if f.name not in ("tokenizer_config.json", "chat_template.jinja", "chat_template.json"):
           (out / f.name).unlink(missing_ok=True)
           (out / f.name).symlink_to(f.resolve())
   cfg = json.loads((src / "tokenizer_config.json").read_text())
   cfg["chat_template"] = ("{{ bos_token }}" if bos else "") + "{% for m in messages %}{{ m['content'] }}{% endfor %}"
   (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
   print(f"{out}: passthrough template, bos={bos}")
   ```
   Run it after step 5 tells you `adds_bos` per model: `uv run python -m bench.make_raw_model openbmb/MiniCPM5-2B-Base runs/models/minicpm5-2b-base-raw <bos|nobos>` and the same for `Qwen/Qwen3-4B-Base` into `runs/models/qwen3-4b-base-raw`. Verify: `uv run python -c "from mlx_lm import load; _,t=load('runs/models/minicpm5-2b-base-raw'); print(repr(t.apply_chat_template([{'role':'user','content':'X'},{'role':'assistant','content':' B'}], tokenize=False)))"`. Expect `'X B'`, preceded by the BOS string only if `bos` was passed.
5. (1.0 h) P0-1. Write `bench/check_labels.py`:
   ```python
   import sys
   from huggingface_hub import snapshot_download
   from transformers import AutoTokenizer
   LABELS = [f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + [f" {d}" for d in range(10)] + [" yes", " no"]
   bad = 0
   for repo in sys.argv[1:]:
       tok = AutoTokenizer.from_pretrained(snapshot_download(repo, local_files_only=True))
       ctx = tok.encode("Answer:", add_special_tokens=False)
       adds_bos = tok.bos_token_id is not None and tok.encode("x")[0] == tok.bos_token_id
       for s in LABELS:
           ids = tok.encode(s, add_special_tokens=False)
           joint = tok.encode("Answer:" + s, add_special_tokens=False)
           if len(ids) != 1 or joint != ctx + ids:
               bad += 1
               print(f"FAIL {repo} {s!r} alone={ids} after_answer={joint[len(ctx):]}")
       print(f"{repo}: adds_bos={adds_bos} labels={len(LABELS)}")
   print("ALL SINGLE-TOKEN" if bad == 0 else f"{bad} FAILURES")
   sys.exit(1 if bad else 0)
   ```
   Run `uv run python -m bench.check_labels openbmb/MiniCPM5-2B-Base Qwen/Qwen3-4B-Base`. Expect two `adds_bos=` lines, last line `ALL SINGLE-TOKEN`, exit code 0. If a tokenizer asks for `trust_remote_code`, read the `.py` file in its snapshot before passing `trust_remote_code=True`. Any `FAIL` line: amend ADR-0003 rule 4 per the decision table, rerun until exit 0.
6. (1.0 h) Tests first: write `tests/test_fmt_ref.py` with `test_prefix_matches_adr` (asserts `prefix_text("S") == PREAMBLE + "### STATE\nS\n### END STATE\n\n"` and `PREAMBLE == "You will answer questions about the STATE. Answer with the label of exactly one option.\n\n"`), `test_choice_branch` (asserts `branch_text("choice", "Q?", ("x", "y")) == "### QUESTION\nQ?\nA. x\nB. y\nAnswer:"`), `test_noul_branch_and_labels` (asserts `branch_text("noul", "Q?", ("yes", "no")) == "### QUESTION\nQ?\n(yes/no)\nAnswer:"`, `label_strings("noul", 2) == [" yes", " no"]`, `label_strings("choice", 3) == [" A", " B", " C"]`). Run `uv run pytest tests/test_fmt_ref.py -q`; expect `ModuleNotFoundError: No module named 'skeleton.fmt_ref'`. Then write `skeleton/fmt_ref.py`:
   ```python
   PREAMBLE = "You will answer questions about the STATE. Answer with the label of exactly one option.\n\n"
   def prefix_text(state):
       return f"{PREAMBLE}### STATE\n{state}\n### END STATE\n\n"
   def option_lines(qtype, options, values=()):
       if qtype == "choice":
           return "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
       if qtype == "score":
           return "\n".join(f"{v}: {o}" for v, o in zip(values, options))
       return "(yes/no)"
   def branch_text(qtype, question, options, values=()):
       return f"### QUESTION\n{question}\n{option_lines(qtype, options, values)}\nAnswer:"
   def label_strings(qtype, n, values=()):
       if qtype == "choice":
           return [f" {chr(65 + i)}" for i in range(n)]
       return [f" {v}" for v in values] if qtype == "score" else [" yes", " no"]
   ```
   Rerun; expect `3 passed`. Write `bench/sample_items.py`: `load_dataset("google/boolq", revision="35b264d0", split="validation")` and `load_dataset("allenai/ai2_arc", "ARC-Challenge", revision="210d026f", split="test")`, `random.Random(0).sample` 25 of each, rows in the Phase 1 schema (see Phase 1 step 1). Run `uv run python -m bench.sample_items`; expect `wrote 50 rows to data/raw/p0_items.jsonl`. Commit.
7. (2.0 h) P0-4. Write `bench/equiv_real.py`, adapting `bench/equiv.py`: load `runs/models/minicpm5-2b-base-raw`; cast parameters with `tree_map(lambda p: p.astype(mx.float32), ...)` as `equiv.py:9`; state = `prefix_text` of 5 BoolQ passages from `p0_items.jsonl` joined by `"\n\n"` (about 1k real tokens); 16 branches = `branch_text` of 8 BoolQ and 8 ARC items, each with its own label ids. (a) Branched: prefill the prefix ids once with `make_prompt_cache`, then for each branch alone copy every layer's `(k, v)` into a fresh `KVCache` (`equiv.py:16-17` with B = 1), run the branch ids, softmax over its label rows at the last position. (b) Oracle: one forward of prefix ids + branch ids, same readout. Print `fp32 branch_vs_reencode=<max abs diff>`. (c) Isolation: crop the 16 branches to the shortest length and compare batched-16 against one-at-a-time as `equiv.py:22-28`; print `isolation=<diff>`. Rerun without the cast and print the bf16 value for the record. Run `uv run python -m bench.equiv_real`. Pass: fp32 <= 1e-3 (expect near 4.0e-4) and `isolation=0.00e+00`. Apply the decision table.
8. (2.0 h, hard time box) P0-6. Reproduce first: `uv run python bench/equiv.py Qwen3-0.6B 1024 16 fp32`; expect `batched_naive_vs_isolated` near `1.2e-01`. Write `bench/anomaly.py` taking the same arguments plus toggles, testing in this order: (1) `--B 2,4,16` (does it appear at B = 2?); (2) `--cpu` via `mx.set_default_device(mx.cpu)` (if CPU matches isolated, the Metal kernel is at fault); (3) `--naive-sdpa`, which replaces the module attribute `mlx_lm.models.qwen3.scaled_dot_product_attention` with an fp32 reference (repeat keys and values for GQA, add a `-inf` upper-triangular mask when `mask == "causal"`, softmax, matmul); (4) `--same-rows`, a batch of 16 identical rows compared against row 0 alone. Stop at 2 h. Outcome A, explained: write the cause and the passing configuration into `reports/phase0.md` and mark R9 for re-review. Outcome B, not explained: write "accepted: batch 1 everywhere (R9)", and file a minimal repro on `ml-explore/mlx` only if outcome (2) isolated it to Metal.
9. (0.5 h) P0-7 setup. List slugs at run time: `curl -s https://openrouter.ai/api/v1/models | uv run python -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data'] if 'deepseek' in m['id'] or 'qwen3.6' in m['id']]"`. Pick the DeepSeek V4.1 Flash and Qwen3.6-35B-A3B ids. For each: `curl -s https://openrouter.ai/api/v1/models/<the id you picked>/endpoints | uv run python -c "import json,sys; [print(e.get('tag'), e['provider_name'], e.get('quantization'), 'top_logprobs' in e['supported_parameters'], e['pricing']['prompt']) for e in json.load(sys.stdin)['data']['endpoints']]"`. Keep endpoints printing `True`; prefer bf16 or fp8 over int4. Write `configs/teachers_p0.yaml` with, per teacher, `teacher_id` (`deepseek-v4.1-flash`, `qwen3.6-35b-a3b`), `model` (the id), `provider_order` (a one-element list, the endpoint `tag`) and `quantization`. If no endpoint of a model prints `True`, that teacher fails P0-7 now.
10. (0.5 h) Write `bench/teacher_probe.py`. Loop: for each teacher, each of the 50 items, 2 runs; POST to `https://openrouter.ai/api/v1/chat/completions` with `httpx`, key from `.env` via `python-dotenv`. Errors: HTTP 404 whose `error.message` starts with `No endpoints found`: retry once with `provider_name` in place of `tag`, then mark the teacher failed; HTTP 429: sleep 10 s, retry once; HTTP 402: stop the run (credit limit hit); `choices[0].logprobs` null: row with `note=no_logprobs`, candidate_mass 0. Per call, append to `ledger/spend.csv` (`phase=0`, `item=P0-7`, host from the response's top-level `provider`, quantization from the config, tokens and `usage.cost`) and to `reports/phase0/teacher_probe.csv`. Core parts:
    ```python
    SYSTEM = "Reply with only the label of one option: a capital letter, or yes, or no."
    def body(model, host, prompt):
        return {"model": model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "reasoning": {"enabled": False}, "usage": {"include": True},
                "provider": {"order": [host], "allow_fallbacks": False, "require_parameters": True},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    def candidate_mass(top, labels, qtype):   # top = choices[0].logprobs.content[0].top_logprobs
        mass = [0.0] * len(labels)             # variants "B", " B", "B." of one label are summed
        for t in top:
            s = t["token"].strip().rstrip(".")
            for j, lab in enumerate(labels):
                a, b = (s.lower(), lab.strip()) if qtype == "noul" else (s, lab.strip())
                mass[j] += math.exp(t["logprob"]) if a == b else 0.0
        total = sum(mass)
        return total, ([m / total for m in mass] if total > 0 else [1 / len(labels)] * len(labels))
    def jsd(p, q):                             # base 2, in [0, 1]
        m = [(a + b) / 2 for a, b in zip(p, q)]
        kl = lambda x, y: sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0)
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    ```
    The prompt is `prefix_text(state) + branch_text(...)` from `skeleton/fmt_ref.py`.
11. (1.0 h) Run `uv run python -m bench.teacher_probe --config configs/teachers_p0.yaml --items data/raw/p0_items.jsonl --runs 2`. Expect one line per teacher: `<teacher_id> n=50 mean_candidate_mass=<x> mean_jsd=<y> no_logprobs=<k> usd=<z>` and a total under 0.20 USD (DERIVED: 200 calls at about 500 tokens). If a teacher fails on its first host, try one more host from step 9 (bounded: 1 extra run). Record host, quantization and both means in `reports/phase0.md`; Q1 and Q7 close here.
12. (1.5 h active, 2 h unattended) P0-8. `configs/p0_lora.yaml` holds `lora_parameters: {rank: 16, scale: 20.0, dropout: 0.0}` and `seed: 0` (rank is not a CLI flag). `bench/make_synth_lora.py` writes `data/p0_synth/train.jsonl` (200 rows) and `valid.jsonl` (5 rows) of `{"prompt": <decoded random token ids in [1000, 30000)>, "completion": " A"}`, trimmed until the templated length is 2,000 to 2,047 tokens. Short run: `mkdir -p runs/p0_soak && uv run mlx_lm.lora --model runs/models/minicpm5-2b-base-raw --train --data data/p0_synth --iters 30 --batch-size 1 --num-layers 16 --max-seq-length 2048 --grad-checkpoint --mask-prompt --steps-per-report 10 --steps-per-eval 100000 --val-batches 1 --adapter-path runs/p0_soak -c configs/p0_lora.yaml`. Expect `Iter 30: ... Tokens/sec <t> ... Peak mem <m> GB` with t near 297 (MEASURED synthetic at 2k). Soak: iters = ceil(7200 x t / 2048), then `caffeinate -i uv run mlx_lm.lora <same flags, --iters <that number>> 2>&1 | tee runs/p0_soak/train.log`. Then `uv run python -m bench.soak_parse runs/p0_soak/train.log`, which parses each `Iter` line, accumulates wall time from `It/sec`, and prints `median_tok_s_after_30min=<a> min_tok_s=<b> drop_pct=<c> peak_mem_30min=<d> peak_mem_end=<e> iters=<n>`. Do not run steps 7 or 8 during the soak (GPU contention).
13. (1.0 h) Write `reports/phase0.md`: one table row per ID, columns `| ID | Measurement | Value | Pass/fail | Decision taken |`, rows starting `| P0-1 |`, `| P0-4 |`, `| P0-6 |`, `| P0-7 |`, `| P0-8 |`, plus setup rows for the model load, `adds_bos` per model and the working set. Log the week in `ledger/hours.csv`. `uv run ruff format . && uv run ruff check --fix .`, commit `phase0: measurements and decisions`.
14. (0.5 h, taken from the buffer) Reference repos and the line budget. Append `refs/` to `.gitignore`. Write `configs/refs.yaml` with these pinned commits (checked 2026-09-24) and clone each shallowly: `git clone --filter=blob:none https://github.com/<repo> refs/<name> && git -C refs/<name> checkout <sha>`.
    | name | repo | sha | licence | what you read it for |
    |---|---|---|---|---|
    | semif | TheoLeeCJ/SemIf-OpenJev | 23cf1f39fc95 | MIT | zero-shot baseline, MLX cache copy, authored144 eval data |
    | decider | Mapika/decider | b44b4c9880a6 | Apache-2.0 | trained letter-slot readout; external trained baseline |
    | reflex | kshetrajna12/reflex | 231f896d818a | MIT | proper-score LoRA loss on restricted logits |
    | von | wfzyx/von | 1d86116dcc76 | Apache-2.0 | order-invariant option masks (later list only) |
    | minisystemone | Colvin0315/MiniSystemOne | 016c6dd0c36a | Apache-2.0 | per-group temperature fitting |
    | jevbench | fstandhartinger/jevbench | 2fa63fa3226c | MIT | public eval items, scoring conventions |
    Rule (AGENTS.md): refs are for **reading**, never imports. If a line of your core is materially adapted from one, add a one-line `# adapted from <repo>@<sha>:<path>` comment and keep the licence notice (all six allow it). Then write `tools/loc.py`:
    ```python
    import sys, pathlib
    BUDGET = {"fmt.py": 130, "engine.py": 200, "calibrate.py": 170, "train.py": 220, "dataset.py": 150, "evaluate.py": 130}
    TOTAL = 1000
    def loc(p):
        return sum(1 for l in p.read_text().splitlines() if l.strip() and not l.strip().startswith("#"))
    rows = {f: (loc(pathlib.Path(f)) if pathlib.Path(f).exists() else 0) for f in BUDGET}
    for f, n in rows.items():
        print(f"{f:14s} {n:5d} / {BUDGET[f]:4d}{'  OVER (soft)' if n > BUDGET[f] else ''}")
    total = sum(rows.values()); print(f"{'core total':14s} {total:5d} / {TOTAL}")
    sys.exit(1 if total > TOTAL else 0)
    ```
    Run `uv run python tools/loc.py`; expect every file at `0` and exit 0. Per-file budgets are soft (a warning); the 1,000-line total is a hard gate from Phase 2 on.

**Verification gate**
- `uv run pytest -q` prints `3 passed`; `uv run ruff check .` prints `All checks passed!`.
- `git log --reverse --stat --format=%s | head -20` shows `.gitignore` in the first commit and `.env.example` in the second, both before any code; `git check-ignore .env` prints `.env`.
- `uv run python -m bench.check_labels openbmb/MiniCPM5-2B-Base Qwen/Qwen3-4B-Base` exits 0.
- `grep -cE "^\| P0-(1|4|6|7|8) \|" reports/phase0.md` prints `5`, and every such row has a non-empty "Decision taken" cell.
- `uv run python -c "import csv; print(round(sum(float(r['usd']) for r in csv.DictReader(open('ledger/spend.csv'))), 4))"` prints a value below `0.2`.
- `ls refs` lists 6 directories; `uv run python tools/loc.py` exits 0.
- The phase passes when every measurement has a decision. A failed measurement is a valid result that selects a fallback, not a failed phase.

**Rollback**
- Undo: `git revert` the phase commits and `rm -rf runs/models runs/p0_soak data/p0_synth data/raw/p0_items.jsonl`; the Hugging Face cache stays and is harmless.
- Point of no return: the first `git push` to a public remote. Do not push in this phase; if you do, run the pre-push check first.
- Data written during a failed window: `ledger/spend.csv` rows are real spend and stay; a half-written `reports/phase0/teacher_probe.csv` is deleted and the probe rerun (about 0.05 USD per full run).

**Kill criteria**
- A9 / A10: if the median tok/s after minute 30 of the soak is below 200 at 2k (or it OOMs) after one retry with `--num-layers 8` (bounded 1 h), then Phase 5 and Phase 6 train on one Modal A100 function (the "later" item, `timeout=` set, cost per `cost.md`) and the Mac does eval only.
- R10: if both teachers show mean candidate_mass < 0.9 or mean JSD >= 0.01 after trying 2 hosts each (bounded 1 h, 0.20 USD), then Phase 4 drops teacher soft labels: gold-only training on public sets plus generator-derived gold for synthetic spec-fact questions. If exactly one passes, that teacher plus gold labels.

### Phase 1: Walking skeleton: B0 and a trained number with stock tools

**Goal:** by the end of week 2, `reports/skeleton.md` shows zero-shot B0 and a stock-LoRA MiniCPM5-2B side by side (accuracy, 15-bin ECE, reversed-order flip rate) on a 1,000-item gold eval.
**Effort:** 16 h (2.7 engineer-days). **Depends on:** Phase 0 (P0-1 passed, raw model dir built). **Parallel with:** nothing (solo); the unattended LoRA run (step 7) overlaps with writing `skeleton/tiny_eval.py`.
**Risk:** low, because every component is stock `mlx_lm` and nothing here is frozen.

**Why this phase exists.** The pre-mortem's most likely failure (R2) is weeks of hand-written infrastructure with no end-to-end number. A walking skeleton is the thinnest version of the whole pipeline that runs from data to a metric: data in, model read, model trained, number out, each piece as crude as allowed. We accept stock tools here because their job is to be the reference: from Phase 2 on, each hand-written module (R20) replaces a stock piece and must reproduce these numbers before you move on. A bug then shows up as a disagreement with a known number instead of as a plausible-looking wrong result.

**What you will understand after this phase**
- **Walking skeleton.** An end-to-end path first, depth later. It turns "is my engine right?" into "does my engine match `reports/skeleton.json`?".
- **Expected calibration error (ECE).** Sort predictions into 15 equal-width bins by their top probability; in each bin compare mean confidence with the fraction correct; ECE is the item-weighted mean gap. A model that says 0.9 and is right 90% of the time scores 0.
- **Order flip rate.** Reverse the option order and ask again; if the top-1 option changes, the model answered the letter position, not the content. pngwn's arm B flipped 37.5%; this is the number the project exists to lower (R14).
- **Prompt-masked fine-tuning.** `--mask-prompt` puts loss only on the answer token, so 950 examples teach "which letter", not "continue this passage". Phase 5 replaces this full-vocab hard-label loss with restricted soft cross-entropy.

**Changes**
| File | Change |
|---|---|
| `skeleton/prep_gold.py` | New. BoolQ and ARC to `data/skeleton/gold_train.jsonl` and `gold_eval.jsonl`; exposes `split_of`. |
| `skeleton/b0_reader.py` | New. Stock load, one sequence per question, restricted readout. Code in step 3. |
| `skeleton/tiny_eval.py` | New. Accuracy, 15-bin ECE, flip rate per type; writes `reports/skeleton.json`. |
| `skeleton/to_lora_jsonl.py` | New. Gold rows to `data/skeleton/train.jsonl` and `valid.jsonl` in completions format; exposes `make_example`. |
| `configs/skeleton_lora.yaml` | New. `lora_parameters: {rank: 16, scale: 20.0, dropout: 0.0}`, `seed: 0`. |
| `tests/test_skeleton.py` | New. 6 tests (steps 2, 4, 5). |
| `reports/skeleton/*.json`, `reports/skeleton.json`, `reports/skeleton.md` | New. Per-item probabilities (about 80 KB each), metrics, write-up. |

**Produces (interfaces later phases use)**
- Row schema of `data/skeleton/gold_*.jsonl`: `{"id": str, "group_key": str, "type": "noul" | "choice", "state": str, "question": str, "options": list[str], "gold_index": int}`. Ids are `boolq:<row index>`, `arc-easy:<ARC id>`, `arc-challenge:<ARC id>`. Noul options are `["yes", "no"]`, gold 0 = yes.
- `skeleton.prep_gold.split_of(source: str, group_key: str, salt: str) -> Literal["train", "eval"]`. Salt `nanohunch-skeleton-v0`. This is NOT the Phase 3 frozen salt; skeleton numbers are never published as test results.
- `skeleton.b0_reader.read(model, tok, row: dict, reverse: bool = False) -> list[float]`, probabilities in canonical (dataset) option order.
- `skeleton.tiny_eval.accuracy`, `ece15(conf, correct) -> float`, `flip_rate(fwd_top1, rev_top1) -> float`: Phase 3 `calibrate.accuracy`, `calibrate.ece(..., bins=15, scheme="width")` and `calibrate.flip_rate` must return identical values on the same inputs.
- `reports/skeleton.json`: `{"b0": {"noul": {"n", "acc", "ece15"}, "choice": {"n", "acc", "ece15", "flip"}, "all": {...}}, "lora": {same}}`. Phase 2 reproduces the `b0` block with `MLXBranchScorer`; Phase 5 reproduces the `lora` block with `train.train` on the same 950 items and hard labels.

**Steps**
1. (3.0 h) `skeleton/prep_gold.py`. BoolQ: `load_dataset("google/boolq", revision="35b264d0")`, train and validation pooled; print `column_names`. `group_key` = the `title` column if present, else `"passage:" + sha256(passage)[:16]`; write which one into `reports/skeleton.md`. State = the passage; question = the BoolQ question with the first letter capitalised and `?` appended; options `["yes", "no"]`; gold 0 if `answer` is true. ARC: `load_dataset("allenai/ai2_arc", c, revision="210d026f")` for `c` in `ARC-Easy`, `ARC-Challenge`, all splits pooled; keep rows with 3 to 5 choices; `gold_index = choices["label"].index(answerKey)` (labels are `A` to `E` or `1` to `5`); `group_key` = the ARC `id`. ARC has no passage, so state = `"Science exam question. There is no passage; answer from general knowledge.\n\n" + stem` and question = `"Which option correctly answers the question in the STATE?"`. Say honestly in `reports/skeleton.md` that the ARC slice tests knowledge, not reading. Split: `split_of` computes `int(hashlib.sha256(f"{salt}|{source}|{group_key}".encode()).hexdigest()[:8], 16) / 2**32` and returns `"train"` below 0.5; then `random.Random(0).sample` per split: 500 BoolQ, 250 ARC-Easy, 250 ARC-Challenge. `--show 20` prints 20 fully rendered rows with the gold option text; read them (R6). Run `uv run python -m skeleton.prep_gold --show 20`. Expect `train=1000 eval=1000 shared_group_keys=0` and 20 rows whose gold option text is correct.
2. (included above) Tests in `tests/test_skeleton.py`; import the module under test inside each test function, so later tests can be written before their modules exist: `test_split_deterministic` (`split_of("boolq", "Tie", "nanohunch-skeleton-v0")` returns the same value on 3 calls and is in `{"train", "eval"}`) and `test_groups_disjoint` (reads both built files, asserts the two `group_key` sets are disjoint). Run `uv run pytest tests/test_skeleton.py -q`; expect `2 passed`.
3. (2.5 h) `skeleton/b0_reader.py` (stock code, the numeric reference):
   ```python
   """B0 / stock-LoRA reader: one full sequence per question, no branching."""
   import argparse, json
   import mlx.core as mx
   from mlx_lm import load
   from skeleton.fmt_ref import prefix_text, branch_text, label_strings

   def encode_split(tok, prefix, branch):   # ADR-0003 rule 2: tokenized separately, ids concatenated
       return tok.encode(prefix) + tok.encode(branch, add_special_tokens=False)

   def read(model, tok, row, reverse=False):
       n = len(row["options"])
       order = list(range(n))[::-1] if reverse and row["type"] == "choice" else list(range(n))
       shown = [row["options"][i] for i in order]
       label_ids = [tok.encode(s, add_special_tokens=False)[0] for s in label_strings(row["type"], n)]
       ids = encode_split(tok, prefix_text(row["state"]), branch_text(row["type"], row["question"], shown))
       last = model(mx.array(ids)[None])[0, -1]                     # logits at the last token of "Answer:"
       p = mx.softmax(last[mx.array(label_ids)].astype(mx.float32)).tolist()
       canon = [0.0] * n
       for disp, c in enumerate(order):                             # display position -> canonical option
           canon[c] = p[disp]
       return canon

   def main():
       ap = argparse.ArgumentParser()
       ap.add_argument("--model", required=True)
       ap.add_argument("--adapter", default=None)
       ap.add_argument("--data", required=True)
       ap.add_argument("--out", required=True)
       ap.add_argument("--reverse", action="store_true")
       ap.add_argument("--limit", type=int, default=0)
       a = ap.parse_args()
       model, tok = load(a.model, adapter_path=a.adapter)
       rows = [json.loads(line) for line in open(a.data)]
       rows = rows[: a.limit] if a.limit else rows
       out = []
       for i, r in enumerate(rows):
           out.append({"id": r["id"], "type": r["type"], "gold_index": r["gold_index"],
                       "reversed": a.reverse, "probs": read(model, tok, r, a.reverse)})
           if i % 100 == 0:
               print(f"{i}/{len(rows)}", flush=True)
       json.dump(out, open(a.out, "w"))
       print(f"wrote {len(out)} rows to {a.out}")

   if __name__ == "__main__":
       main()
   ```
   Smoke: `uv run python -m skeleton.b0_reader --model runs/models/minicpm5-2b-base-raw --data data/skeleton/gold_eval.jsonl --out /tmp/b0_smoke.json --limit 5`. Expect `wrote 5 rows`, each `probs` summing to 1 within 1e-5.
4. (1.5 h) `skeleton/tiny_eval.py`. Tests first in `tests/test_skeleton.py`: `test_ece_perfect` (`ece15([1.0, 1.0], [1, 1]) == 0.0`), `test_ece_known` (ten predictions at 0.9 with 8 correct give `abs(ece15(...) - 0.1) < 1e-9`), `test_flip_rate` (`flip_rate([0, 1, 2], [0, 2, 2]) == 1/3`). Run `uv run pytest tests/test_skeleton.py -q`, expect `3 failed, 2 passed` with `ModuleNotFoundError: No module named 'skeleton.tiny_eval'`. Then:
   ```python
   def accuracy(probs, gold):
       return float(np.mean([int(np.argmax(p)) == g for p, g in zip(probs, gold)]))
   def ece15(conf, correct):
       conf, correct = np.asarray(conf, float), np.asarray(correct, float)
       idx = np.minimum((conf * 15).astype(int), 14)                 # bin 14 includes conf == 1.0
       return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                        for b in range(15) if (idx == b).any()))
   def flip_rate(fwd_top1, rev_top1):
       return float(np.mean([a != b for a, b in zip(fwd_top1, rev_top1)]))
   ```
   The CLI `--name <b0|lora> --fwd <file> --rev <file> --json reports/skeleton.json` prints `name type n acc ece15 flip` rows for `noul`, `choice`, `all` (flip only for choice, from rows whose `reversed` is true) and merges its block into the JSON. Expect `5 passed` for the file.
5. (1.0 h) B0 runs: `mkdir -p reports/skeleton`, then `uv run python -m skeleton.b0_reader --model runs/models/minicpm5-2b-base-raw --data data/skeleton/gold_eval.jsonl --out reports/skeleton/b0_fwd.json` and the same with `--reverse --out reports/skeleton/b0_rev.json`. Expect about 4 minutes each (DERIVED: 1,000 sequences of about 300 tokens at 1,300 tok/s prefill, MEASURED). Then `uv run python -m skeleton.tiny_eval --name b0 --fwd reports/skeleton/b0_fwd.json --rev reports/skeleton/b0_rev.json --json reports/skeleton.json`. Determinism: rerun with `--limit 50 --out /tmp/b0_50.json` and compare to the first 50 rows; expect a max abs diff of 0.0 (accept <= 1e-5).
6. (2.0 h) `skeleton/to_lora_jsonl.py`. `make_example(row, rng) -> {"prompt", "completion", "perm"}`: for Choice draw `perm = rng.sample(range(n), n)` (`perm[display_pos] = canonical index`, the registry's `Perm` convention), show `options[perm[j]]` at position j, completion = the label at the display position `j` where `perm[j] == gold_index`; Noul is never permuted. Prompt = `prefix_text + branch_text`, completion like `" B"`. Test `test_target_remap`: for 200 rows, the option line for the completion letter contains `options[gold_index]`. Main: `random.Random(0)`, first 950 train rows to `data/skeleton/train.jsonl`, last 50 to `valid.jsonl` (stock `mlx_lm.lora` needs a validation file; 950 is what is trained on). It loads the raw tokenizer and prints `max_len=<L> split_mismatch=<K> of 950`, where K counts rows whose `apply_chat_template` ids differ from `encode_split(prefix, branch) + [label_id]`. Expect L < 2048 (rows over it are dropped and counted) and K = 0; if K > 9 (1%), recheck step 4 of Phase 0 before training. Run `uv run pytest tests/test_skeleton.py -q`, expect `6 passed`.
7. (1.5 h) Check flags: `uv run mlx_lm.lora --help | grep -cE -- "--(train|mask-prompt|grad-checkpoint|num-layers|max-seq-length|adapter-path|iters|batch-size)"`; expect at least 8 (all present in mlx-lm 0.31.3 on this Mac, MEASURED). Train: `caffeinate -i uv run mlx_lm.lora --model runs/models/minicpm5-2b-base-raw --train --data data/skeleton --iters 1900 --batch-size 1 --num-layers 16 --max-seq-length 2048 --grad-checkpoint --mask-prompt --learning-rate 2e-5 --steps-per-report 50 --steps-per-eval 200 --val-batches -1 --save-every 500 --adapter-path runs/skeleton -c configs/skeleton_lora.yaml 2>&1 | tee runs/skeleton_train.log`. 1,900 iterations = 2 epochs at batch 1 (R9). Expect `Saved final weights to runs/skeleton/adapters.safetensors`, validation loss below its iteration-0 value, and 20 to 45 minutes of wall time (DERIVED from 232 to 347 tok/s MEASURED).
8. (1.0 h) LoRA eval: `uv run python -m skeleton.b0_reader --model runs/models/minicpm5-2b-base-raw --adapter runs/skeleton --data data/skeleton/gold_eval.jsonl --out reports/skeleton/lora_fwd.json`, the same with `--reverse --out reports/skeleton/lora_rev.json`, then `uv run python -m skeleton.tiny_eval --name lora --fwd reports/skeleton/lora_fwd.json --rev reports/skeleton/lora_rev.json --json reports/skeleton.json`. Adapter-applied check (R3): `uv run python -c "import json; a=json.load(open('reports/skeleton/b0_fwd.json'))[:5]; b=json.load(open('reports/skeleton/lora_fwd.json'))[:5]; print(max(abs(x-y) for r,s in zip(a,b) for x,y in zip(r['probs'],s['probs'])))"`; expect a value above `0.001`.
9. (2.0 h) `reports/skeleton.md`: what a walking skeleton is (two sentences), the B0 and LoRA table per type (acc, ece15, flip), the per-type delta with counts of items LoRA-right-B0-wrong and the reverse, and a notes block: salt `nanohunch-skeleton-v0`, BoolQ `group_key` choice, the ARC honesty note, 950/50 split, `split_mismatch`, iterations, learning rate, wall time and peak memory from the log. State that these are the reference numbers for Phase 2 and Phase 5. `uv run ruff format . && uv run ruff check --fix .`, commit `phase1: walking skeleton, B0 and stock LoRA reference numbers`, log hours.
10. (1.5 h) Buffer for interruptions and one rerun.

**Verification gate**
- `uv run pytest -q` prints `9 passed` (3 from Phase 0, 6 here).
- `uv run python -m skeleton.tiny_eval --name b0 --fwd reports/skeleton/b0_fwd.json --rev reports/skeleton/b0_rev.json` prints rows `noul 500`, `choice 500`, `all 1000`.
- Below-chance tell (R6): B0 `noul` acc >= 0.55 and `choice` acc >= 0.35 in `reports/skeleton.json` (chance is 0.50 and about 0.25). Below that, the readout or remapping is broken.
- `reports/skeleton.json` has both `b0` and `lora` blocks; `reports/skeleton.md` exists and is committed by end of week 2.
- Sanity: LoRA acc is not more than 3 pts below B0 on either type. If it is, treat it as a bug (template, `split_mismatch`, learning rate), not a result.

**Rollback**
- Undo: `git revert` the phase commit; `rm -rf data/skeleton runs/skeleton runs/skeleton_train.log`. No other module depends on the skeleton yet.
- Point of no return: none. The skeleton salt is throwaway, the adapter is local and gitignored, and nothing is published.
- Data written during a failed window: a partial adapter in `runs/skeleton` is overwritten by rerunning step 7 (`--resume-adapter-file runs/skeleton/0001500_adapters.safetensors` resumes from the last save); partial `reports/skeleton/*.json` files are only written at the end of a run, so they are complete or absent.

**Kill criterion**
- ADR-0002 (letter-logit readout): if B0 accuracy is at or below chance on either type after 2 h of readout debugging (P0-1 having passed), then MiniCPM5-2B-Base is not readable this way: rerun this phase on `runs/models/qwen3-4b-base-raw` and, if Qwen passes, it leads the Phase 3 bake-off and the Modal later item moves forward.
- A7 (hours): if `reports/skeleton.md` is not committed by the end of week 3, the R17 calendar rules apply one week early: Milestone 1 is cut to 2 synthetic workflows and a 100-item audit.
