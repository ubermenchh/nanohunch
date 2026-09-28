### Phase 5: Hand-write the trainer, pass the numerics gate, learning curve: Milestone 2

**Goal:** by the end of week 7, `reports/m2/README.md` shows a learning curve from your own `train.py` at 1k and 3k
decisions, each point with a paired-bootstrap delta over calibrated B0 and a flip rate, plus the pre-registered kill decision.
**Effort:** 24 h = 4.0 engineer-days, weeks 6 to 7 (about 11 h of it unattended training, not counted).
**Depends on:** Phase 3 (frozen split, `run_eval`, B0 calibration), Phase 4 (`data/built/v1`). **Parallel with:** nothing
(one person, 1 engineer; training nights overlap with report writing).
**Risk:** high: a trainer that disagrees with the engine by a scale factor or an off-by-one position trains the wrong thing
silently for hours.
**ONE-WAY DOOR:** nanohunch-fmt-v1 is baked into every adapter from the first curve run on. After that, changing `fmt.py`
means retraining every adapter.

**Why this phase exists**

Stock `mlx_lm.lora` (Phase 1) trains full-vocabulary cross-entropy on one hard token. The product needs soft targets over the
label rows only, and permutation augmentation, and neither is a flag. Writing the trainer yourself is also the second half of
the learning goal. But a hand trainer is the easiest place in the project to be wrong without noticing, so this phase puts a
**numerics gate** (trainer and engine agree on the same adapter) before any overnight run, and a **pre-registered kill rule**
before the run that decides whether scaling data (Phase 6) is worth it.

**What you will understand after this phase**

- **LoRA.** A frozen layer computes `W x`. LoRA adds a trainable low-rank path: `y = W x + (alpha / r) * B (A x)`, with `A` of
  shape `[r, in]` and `B` of shape `[out, r]`. `A` starts small and random, `B` starts at zero, so at step 0 the extra term is
  exactly zero and the model equals the base; the zero test proves this. On a 2048 x 2048 layer, full fine-tuning trains 2048 x
  2048 = 4,194,304 numbers; LoRA with r = 16 trains 16 x 2048 + 2048 x 16 = 65,536, 1.6% as many. `alpha / r` is a fixed gain:
  alpha 32, r 16 gives 2.0. In the `mlx_lm` adapter file that gain is stored as `scale`, so you write `scale = alpha / rank`,
  never `alpha`.
- **Restricted soft cross-entropy.** Take the last-position hidden state, multiply it only by the head rows of the label tokens
  (`engine.label_logits`), softmax over those n numbers, and score against a target distribution: `loss = -sum_i t_i log p_i`.
  Example: target `[0.7, 0.2, 0.1]`, model `[0.5, 0.3, 0.2]`:
  `-(0.7 ln 0.5 + 0.2 ln 0.3 + 0.1 ln 0.2) = 0.485 + 0.241 + 0.161 = 0.887`. The loss is minimised at `p = t`, and the minimum
  is the target's entropy (0.802 here), not zero; this is why the overfit test uses one-hot targets. We never take the full
  vocabulary: the head is about 73k rows x 2048, so full logits at every step cost memory we do not need, and inference only
  ever reads the label rows. Training on them and reading them is one consistent contract.
- **Permutation augmentation.** Each epoch, every Choice question gets a new random option order, and its target is moved with
  its options by **option id**: if option `o3` (target 0.7) is shown at position B, then `B` gets 0.7. The model can no longer
  learn "the answer is usually A", so position cannot be a shortcut. Score and Noul are never permuted: their labels are ordinal
  or fixed yes/no.
- **Why a pre-registered kill criterion.** After 30 hours of work, any positive number looks like a reason to continue. Writing
  the threshold and the alternative down and committing it before the decisive run is what makes the decision honest; the git
  timestamp is the evidence.

**Read before writing (45 min)**
- `refs/reflex/src/reflex/train/calibrate.py`: a LoRA loss (NLL or Brier) on label-restricted logits, almost exactly your `restricted_soft_ce`. Note how they mask invalid options.
- `refs/reflex/docs/results/frozen-vs-trained.md` and the `lora-mix*` reports: every adapter they trained was rejected because it hurt general judgement. This is the strongest outside evidence for the kill criterion below; read it before you write `reports/m2/prereg.md`.
- `refs/decider/decider/train.py`: full fine-tuning with CE plus optional Brier, which did beat frozen Qwen at 1.47M examples. Your 3k-decision curve is roughly 500x smaller, so a null result is plausible and still publishable.

**Changes**

| File | Change |
|---|---|
| `train.py` | **You write.** `LoRALinear`, `apply_lora`, `restricted_soft_ce`, `permute_question`, `make_example`, `run_loop`, checkpoint save/find/load, `train`, `load_adapter_model`. The target remap is `dataset.to_display` (Phase 4), imported, never re-implemented. Glue (concrete, and if `train.py` nears its budget, move it to `config.py`): `load_config`, `TrainConfig` dataclasses, JSONL logging. |
| `cli.py` | Glue. `train --config C [--dry-run] [--tok-s N]`: dry run prints `decisions=<n> dropped=<k> tokens=<t> steps=<s> est_hours=<h>` with `h = t * epochs / N / 3600`, `N` default 297 (P0-8, replace with the soak median in `reports/phase0.md`). Without `--dry-run` it calls `train.train(Path(C))` and prints the adapter dir. |
| `skeleton/to_train_rows.py` | New, glue. Converts the first 950 rows of the Phase 1 gold train file (the 1,000-row file `skeleton/prep_gold.py` wrote next to `data/skeleton/gold_eval.jsonl`; `ls data/skeleton` shows it) into the Phase 4 row schema at `data/skeleton/rows_950.jsonl`. Same 950 rows stock LoRA trained on, and each row's `canonical_order` is the option order stock showed (the `perm` stored by `skeleton/to_lora_jsonl.py`), so with `perm_augment: false` every item is displayed exactly as stock displayed it. |
| `configs/skeleton_repro.yaml`, `configs/curve_1k.yaml`, `configs/curve_3k.yaml`, `configs/overfit_32.yaml`, `configs/eval_m2.yaml` | New. Contents below. |
| `tests/test_train.py` | New. 4 fast tests below; builds a tiny model with `mlx_lm.models.llama.Model(ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=512))`. |
| `gates/test_overfit32.py`, `gates/test_numerics_gate.py` | New. 1 and 4 tests. They live outside `tests/` so plain `uv run pytest -q` never starts a 20-minute training run or fails for lack of `NANOHUNCH_ADAPTER`; you run them by explicit path. |
| `pyproject.toml` | Add `testpaths = ["tests"]` to `[tool.pytest.ini_options]`. |
| `reports/m2/prereg.md`, `reports/m2/README.md`, `reports/m2/*.json`, `reports/m2/curve.png` | New, committed. |

**Produces (interfaces later phases use)**

```python
# train.py
@dataclass(frozen=True) class LoRAConfig: rank: int; alpha: float; dropout: float; num_layers: int; targets: tuple[str, ...]
@dataclass(frozen=True) class OptimConfig: lr: float; weight_decay: float; betas: tuple[float, float]; warmup_frac: float
                                           schedule: Literal["cosine", "constant"]; min_lr_frac: float; clip_grad_norm: float | None
@dataclass(frozen=True) class TrainConfig: model_path: str; data_path: str; out_dir: str; n_decisions: int | None; gold_only: bool
    lambda_gold: float; perm_augment: bool; lora: LoRAConfig; optim: OptimConfig; max_seq: int; batch_size: int; grad_accum: int
    epochs: int; max_steps: int | None; grad_checkpoint: bool; ckpt_minutes: float; keep_ckpts: int; max_drop_frac: float
    select_by: Literal["last"]; seed: int
class ConfigError(ValueError): ...          # message names the unknown or missing key
class LoRATargetNotFound(KeyError): ...     # message names the key and the layer index
class ResumeMismatch(RuntimeError): ...     # checkpoint cfg_sha256 differs from the current config
@dataclass(frozen=True, slots=True) class TrainExample: token_ids: tuple[int, ...]; label_ids: tuple[int, ...]; target: np.ndarray  # [n] display order
# One example = one (state row, decision) pair. Phase 4 rows are states with a decisions[] list; train() flattens
# them, and n_decisions, max_drop_frac and gold_only all count decisions, never rows.
ExampleFn = Callable[[int, int], TrainExample]          # (epoch, index) -> example; pure, deterministic
@dataclass(frozen=True) class LoopResult: steps: int; losses: list[float]; adapter_dir: Path
def load_config(path: Path) -> TrainConfig
class LoRALinear(nn.Module):                             # params lora_a [in, r], lora_b [r, out], same names as mlx_lm
    @staticmethod
    def from_base(linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> "LoRALinear"
def apply_lora(model, cfg: LoRAConfig) -> int            # freezes base, returns trainable parameter count
def restricted_soft_ce(label_logits, target_probs, valid_mask) -> mx.array   # [B,N],[B,N],[B,N] bool; SUM over decisions
def permute_question(q: Question, perm: Perm) -> Question
def make_example(tokenizer, row: dict, decision: dict, perm: Perm, lambda_gold: float) -> TrainExample   # raises StateTooLong; target via dataset.to_display
def run_loop(model, n_examples: int, example_fn: ExampleFn, cfg: TrainConfig, out_dir: Path, *,
             stop_after: int | None = None, force_ckpt_at: int | None = None) -> LoopResult
def train(cfg_path: Path) -> Path                        # returns <out_dir>/adapter; resumes automatically
def load_adapter_model(adapter_dir: Path) -> tuple[nn.Module, object]   # trainer's own LoRALinear path, for the gate
```

- **Adapter dir** `<out_dir>/adapter/`: `adapters.safetensors`, `adapter_config.json` (`fine_tune_type: "lora"`, `num_layers`,
  `lora_parameters: {rank, scale: alpha/rank, dropout, keys}`), loadable by `mlx_lm.load(model_path, adapter_path=...)` and
  `MLXBranchScorer(adapter_path=...)`; plus `train_meta.json` (`model_path`, `format_version`, `cfg_sha256`, `step`,
  `data_sha256`). `<out_dir>/train_items.jsonl` lists the `(state_id, question_id)` of every example trained on, so the
  gate can score exactly those items.
- **Checkpoint dir** `<out_dir>/ckpt-<step>/`: `adapters.safetensors`, `optimizer.safetensors` (flattened `optimizer.state`,
  including its `step`), `state.json` (`step`, `epoch`, `cursor`, `cfg_sha256`, `python_random_state`, `numpy_bitgen_state`,
  `mx_key`).
- **Log** `<out_dir>/log.jsonl`, one line per optimizer step:
  `ts, step, epoch, loss, entropy, grad_norm, lr, tok_s, peak_mem_gb`.
- **Config key names are the contract.** Phase 6 `configs/train_full.yaml` must say `data_path` (not `data`) and `ckpt_minutes`
  (not `checkpoint_every_min`); `load_config` raises `ConfigError` on both, so the mismatch fails on the dry run, not overnight.

**Configs** (`configs/curve_3k.yaml`; `curve_1k.yaml` differs only in `out_dir: runs/curve_1k`, `n_decisions: 1000`):

```yaml
model_path: runs/models/minicpm5-2b-base-raw
data_path: data/built/v1/train.jsonl
out_dir: runs/curve_3k
n_decisions: 3000          # seeded shuffle then prefix, so the 1k set is inside the 3k set
gold_only: false
lambda_gold: 0.0           # target = (1-l)*consensus + l*onehot(gold); consensus alone if no gold; onehot if no consensus
perm_augment: true
lora: {rank: 16, alpha: 32, dropout: 0.0, num_layers: 16,
       targets: [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj]}
optim: {lr: 1.0e-4, weight_decay: 0.01, betas: [0.9, 0.999], warmup_frac: 0.03, schedule: cosine, min_lr_frac: 0.1, clip_grad_norm: 1.0}
max_seq: 4096
batch_size: 1              # R9: batch 1 until the P0-6 anomaly is explained
grad_accum: 16
epochs: 2
max_steps: null
grad_checkpoint: true
ckpt_minutes: 15
keep_ckpts: 3
max_drop_frac: 0.02
select_by: last
seed: 0
```

`configs/skeleton_repro.yaml` copies `curve_3k.yaml` and matches the stock run instead:
`data_path: data/skeleton/rows_950.jsonl`, `out_dir: runs/skeleton_repro`, `n_decisions: 950`, `gold_only: true`,
`lambda_gold: 1.0`, `lora.alpha: 320` (scale 20.0, as `configs/skeleton_lora.yaml`), `lora.targets` = the `keys` in
`runs/skeleton/adapter_config.json`, `max_seq: 2048`, `grad_accum: 1`, `epochs: 1` (950 steps, as stock since 2026-09-25),
`optim: {lr: 5.0e-6, weight_decay: 0.0, warmup_frac: 0.0, schedule: constant, clip_grad_norm: null}`, `perm_augment: false`
(stock drew one permutation per row, now baked into `canonical_order` by `to_train_rows`). If stock used its default LoRA
targets, `keys` may be missing from `runs/skeleton/adapter_config.json`; then read the defaults from
`mlx_lm/tuner/utils.py` for this model type and write them in explicitly. `configs/overfit_32.yaml`:
`out_dir: runs/overfit32`, `n_decisions: 32`, `gold_only: true`, `lambda_gold: 1.0`, `perm_augment: false`, `max_seq: 1024`,
`max_drop_frac: 1.0`, `grad_accum: 1`, `epochs: 10`, `max_steps: 300`, `optim.lr: 1.0e-3`, `warmup_frac: 0.03` (32 examples x
10 epochs = 320 micro-steps, so `max_steps: 300` is the binding stop; with the inherited `epochs: 2` the run would end at 64).

`configs/eval_m2.yaml`:

```yaml
data: data/built/v1
split: test                  # test.jsonl, including its split: test_ood rows; second look at v1 test (M1 was the first)
perm_suite: [1, 2]           # flip rate = canonical vs the second perm of permutations_for (the reversal)
predictors:
  B0:        {adapter: null,                 calibration: runs/b0/calibration.json,        n_perms: 1}
  curve_1k:  {adapter: runs/curve_1k/adapter, calibration: runs/curve_1k/calibration.json, n_perms: 1}
  curve_3k:  {adapter: runs/curve_3k/adapter, calibration: runs/curve_3k/calibration.json, n_perms: 1}
baseline: B0
slices:
  gold:             {label_origin: gold}                            # public gold, whatever the source id spelling
  heldout_template: {split: test_ood, label_origin: spec}           # spec-fact decisions have gold; the kill metric
  heldout_judgement: {split: test_ood, label_origin: teacher}       # consensus top-1 only; reported, not decisive
bootstrap: {n: 10000, seed: 0, groups: state_id}
out_dir: reports/m2
```

**Steps** (tests first; `uv run pytest tests/test_train.py -q` after each)

1. (0.5 h) Look before writing.
   `uv run python -c "import json;print(json.loads(open('data/built/v1/train.jsonl').readline()))"`: confirm the row fields
   `make_example` reads (state, question with option texts, option ids, gold option id, consensus by option id); if Phase 4
   named them differently, only `make_example` changes.
   `uv run python -c "from mlx_lm import load;m,_=load('runs/models/minicpm5-2b-base-raw');print([n for n,_ in m.layers[0].named_modules()])"`:
   confirm the 7 target names. Read the model class `__call__` in `mlx_lm/models/` for MiniCPM: note any pre-head scale
   (`hidden / (hidden_size / dim_model_base)`) and whether the head is tied to `embed_tokens`. Use exactly the hidden-to-head
   path the Phase 2 engine uses.
2. (1.0 h) Write `test_lora_zero_init_equals_base`: tiny model, logits on a fixed 16-token input before and after
   `apply_lora(model, LoRAConfig(16, 32, 0.0, 2, targets))`; assert `max|diff| == 0.0`, every `lora_b` is all zeros, some
   `lora_a` is nonzero, the returned count equals `sum(r * (in + out))` over targeted linears, and
   `tree_flatten(model.trainable_parameters())` names end only in `lora_a` or `lora_b`. A target name that does not exist raises
   `LoRATargetNotFound`. Run: expect 1 failure (`ImportError: cannot import name 'apply_lora'`). Implement `LoRALinear` and
   `apply_lora` to this contract: the base is frozen; only the target linears of the last `num_layers` blocks are wrapped;
   `lora_a` is small uniform (bound `1/sqrt(in)`), `lora_b` is zero, the gain is `alpha / rank`; the forward is the
   "What you will understand" formula with the mlx_lm shapes (`lora_a [in, r]`, `lora_b [r, out]`, so `x` goes through
   `lora_a` first). Run: expect pass.
3. (1.5 h) Write `test_restricted_soft_ce_matches_masked_full_ce`: (a)
   `restricted_soft_ce(log([[0.5,0.3,0.2]]), [[0.7,0.2,0.1]], all true)` is `0.8869` within `1e-4`; (b) random `hidden [4,8]`,
   `head [50,8]`, `label_ids` of length 3 to 5: `label_logits(hidden, head, ids)` equals `(hidden @ head.T)[:, ids]` within
   `1e-6`, and the restricted loss equals full-vocab CE with non-label logits set to `-inf` within `1e-5`; (c) padding row to N
   = 6 with `valid_mask` false leaves the loss unchanged within `1e-6`, even with a large logit in the padded slot. Implement
   it so that masked slots get no probability mass and contribute exactly 0. The trap test (c) catches: a masked slot with
   target 0 and log-prob `-inf` gives `0 * -inf = nan`. Run: expect pass.
4. (1.0 h) Write `test_perm_remap_property`: 500 cases from `np.random.default_rng(0)`, `n` in 2 to 8, random perm and random
   target by option id. Assert for every display position j: `permute_question(q, perm).options[j] == q.options[perm[j]]`,
   `dataset.to_display(t, ids, perm)[j] == t[ids[perm[j]]]`, the sum is 1 within `1e-9`, and the text at the display argmax
   of the permuted question equals the text of the canonical argmax. This tests that `permute_question` and `to_display`
   agree, which is where an inversion would hide. Implement `permute_question` (pure). `make_example` builds the display
   question for one decision, renders it with `n_perms=1` and `max_context=cfg.max_seq`, and returns `prefix_ids +
   branch.token_ids`, `branch.label_ids`, and the `to_display` target. Run: expect pass.
5. (4.5 h) Write `test_resume_bitwise` (tiny model, 40 synthetic `TrainExample`s, `grad_accum: 4`, `tmp_path`): run A,
   `run_loop(..., stop_after=6)` uninterrupted, losses `LA`; run B, fresh model with the same seed,
   `stop_after=3, force_ckpt_at=3`; then a fresh model and a fresh `run_loop(..., stop_after=6)` on the same `out_dir` resumes
   from `ckpt-3` and returns losses `LB` for steps 4 to 6 only (`LoopResult.losses` covers steps run in that call). Assert
   `LB == LA[3:6]` with `==` (bitwise), and that `ckpt-3/state.json` has `cursor == 12`. Implement `run_loop` to this
   contract. The order and the code are yours; the relevant APIs are in `mlx.optimizers` (schedules, AdamW, gradient
   clipping), `mlx.nn.value_and_grad` and `mx.checkpoint`.
   - **Determinism.** Every RNG (mx, python, numpy) is seeded from `seed` before anything random happens. Epoch order is a
     function of `(seed, epoch)` only, and each example's perm is a function of `(seed, epoch, index)` only, so neither
     depends on how many steps ran before a resume.
   - **Length and schedule.** The run stops at `total = max_steps or ceil(n_examples * epochs / grad_accum)` optimizer
     steps, even if epochs remain. Linear warmup over `warmup_frac * total` steps, then cosine decay to
     `lr * min_lr_frac`, or a constant `lr`.
   - **Accumulation.** Each micro-step's loss is scaled by `1 / grad_accum`; the optimizer steps once per `grad_accum`
     micro-steps on the summed gradients, clipped first if `clip_grad_norm` is set; each optimizer step writes one log line.
   - **Resume.** A checkpoint holds everything needed to continue bitwise: adapter weights, the full optimizer state
     including its step, all RNG states, `step`, `epoch`, `cursor`. On resume, take the newest complete `ckpt-<int>`
     (ignore `*.tmp`), raise `ResumeMismatch` if `cfg_sha256` **or** `data_sha256` differs, and truncate `log.jsonl` to the
     resumed step.
   - **Durability.** Checkpoint every `ckpt_minutes` (and at `force_ckpt_at`), written as `.tmp` then renamed, keeping the
     newest `keep_ckpts`. The final adapter is written the same way, so `adapter/` exists only after a finished run.
   - `grad_checkpoint: true` trades compute for memory per block. Run: expect `4 passed`.
6. (1.0 h) Glue: `load_config` (reject unknown and missing keys with `ConfigError`), `train` (load config, seed, `mlx_lm.load`,
   `apply_lora` (after seeding, so `lora_a` depends on `cfg.seed`), read rows, **flatten to `(row, decision)` pairs**, drop
   decisions that raise `StateTooLong` at `max_seq` and raise `ConfigError` if the dropped fraction of decisions exceeds
   `max_drop_frac`, keep decisions with `label_origin in {"gold", "spec"}` if `gold_only`, seeded shuffle, prefix
   `n_decisions` decisions, write `train_items.jsonl`, call `run_loop`), `load_adapter_model`, and `cli.py train`. `uv run python cli.py train --config configs/curve_1k.yaml --dry-run`; expect
   `decisions=1000` and `est_hours` between 0.9 and 2.0 (DERIVED design value 1.3 h). Outside that band, record the number and
   rescale the week-7 schedule before step 11.
7. (1.5 h) Write `gates/test_overfit32.py::test_overfit_32`: `shutil.rmtree("runs/overfit32", ignore_errors=True)`,
   `train(Path("configs/overfit_32.yaml"))`; assert the log has 300 lines and the mean loss of the last 10 is `< 0.05`.
   `caffeinate -i uv run pytest gates/test_overfit32.py -q`; expect `1 passed` in about 15 to 20 minutes (DERIVED: 300 steps
   under 1,024 tokens at about 300 tok/s). Then run the resume drill on the real model once: start the same config, Ctrl-C
   after the first checkpoint (set `ckpt_minutes: 2` for the drill), rerun, and check that `log.jsonl` continues without a gap
   or duplicate step. R18 depends on resume working at 2B scale, not only on the tiny model.
8. (1.5 h, plus 3.0 h reserved for gate debugging) Write `gates/test_numerics_gate.py`. Module fixture reads `os.environ["NANOHUNCH_ADAPTER"]` (missing:
   `pytest.fail("set NANOHUNCH_ADAPTER")`, never skip) and `model_path` from its `train_meta.json`. Items: the 64 gold
   decisions of `data/built/v1/cal.jsonl` (public gold and spec gold, at least 16 of them Score) with the smallest
   `state_id` that fit 4,096 tokens.
   - `test_trainer_engine_nll_parity` (R9): trainer side `load_adapter_model(adapter)`,
     `make_example(..., perm=identity, lambda_gold=1.0)`, `-log softmax(label_logits)[gold]`; engine side
     `MLXBranchScorer(model_path, adapter_path=adapter).score(render(...))`, `-log softmax(logits)[gold]`. Assert
     **per item** `|NLL_trainer - NLL_engine| <= 2e-2` (R9 says per-example; a mean can hide one badly wrong item).
   - `test_adapter_applied` (R3): 5 prompts, engine with and without the adapter; assert **each** prompt's max abs logit
     diff is `> 1e-3`.
   - `test_oracle_fp32_with_adapter` (R8): 8 of the items on the CPU backend (as Phase 2 `test_oracle_fp32`, P0-4
     decision), `MLXBranchScorer(model_path, adapter_path=adapter, dtype="float32")`; softmax of `score` vs
     `score_reencode(r)` on that fp32 scorer, max abs prob diff `<= 1e-3`.
   - `test_overfit32_through_engine`: engine with `runs/overfit32/adapter` (fail with "run: uv run pytest
     gates/test_overfit32.py" if absent) on exactly the items in `runs/overfit32/train_items.jsonl`; assert 32/32 top-1
     equal gold and mean NLL `< 0.1`.
   Run `NANOHUNCH_ADAPTER=runs/overfit32/adapter uv run pytest gates/test_numerics_gate.py -q`; expect `4 passed`. On failure, check
   in this order: head rows (tied vs `lm_head`), the pre-head scale from step 1, `scale` written as `alpha` instead of
   `alpha / rank` (diffs near a constant factor), last position off by one, a BOS token added on one side only. Start the 8 h
   kill clock at the first failure.
9. (0.5 h active, about 45 min unattended) Reproduce the skeleton. `uv run python -m skeleton.to_train_rows` (expect
   `rows=950`), then
   `caffeinate -i uv run python cli.py train --config configs/skeleton_repro.yaml 2>&1 | tee runs/skeleton_repro.log`. Gate it:
   `NANOHUNCH_ADAPTER=runs/skeleton_repro/adapter uv run pytest gates/test_numerics_gate.py -q`, expect `4 passed`. Evaluate with
   the Phase 1 reader for an apples-to-apples number:
   `uv run python -m skeleton.b0_reader --model runs/models/minicpm5-2b-base-raw --adapter runs/skeleton_repro/adapter --data data/skeleton/gold_eval.jsonl --out reports/skeleton/repro_fwd.json`,
   the same with `--reverse --out reports/skeleton/repro_rev.json`, then
   `uv run python -m skeleton.tiny_eval --name repro --fwd reports/skeleton/repro_fwd.json --rev reports/skeleton/repro_rev.json --json reports/skeleton.json`.
   Expect `all` accuracy within 0.02 of the `lora` block. The losses differ on purpose (restricted soft CE vs full-vocab CE),
   so 0.01 on 1,000 items would sit inside run-to-run noise.
10. (1.0 h) Pre-register. Write `reports/m2/prereg.md` containing, verbatim, both kill criteria below; the metric (gold
    accuracy on the `heldout_template` slice of `configs/eval_m2.yaml`: `test_ood` spec-fact decisions, paired bootstrap
    `n=10_000, seed=0` resampled by `state_id`, vs calibrated B0); the slice's decision and state counts; the CI half-width
    those counts imply at B0's accuracy (the minimum detectable effect); and the configs' `sha256sum`.
    `git add reports/m2/prereg.md configs/curve_*.yaml configs/eval_m2.yaml && git commit -m "phase5: pre-register M2 kill rule"`.
    This commit must exist before step 12 starts.
11. (0.5 h active, about 1.3 h unattended)
    `mkdir -p runs/curve_1k && caffeinate -i uv run python cli.py train --config configs/curve_1k.yaml 2>&1 | tee -a runs/curve_1k/train.log`.
    After a crash or sleep, rerun the same command; it resumes. Gate:
    `NANOHUNCH_ADAPTER=runs/curve_1k/adapter uv run pytest gates/test_numerics_gate.py -q`, expect `4 passed`.
12. (0.5 h active, about 4 h unattended, overnight) The same two commands with `curve_3k`.
13. (2.0 h) Calibrate and evaluate.
    `uv run python cli.py fit-cal --adapter runs/curve_1k/adapter --split cal --data data/built/v1/cal.jsonl --n-perms 1 --out runs/curve_1k/calibration.json`,
    the same for `curve_3k`, then `caffeinate -i uv run python cli.py eval --config configs/eval_m2.yaml`. Expect
    `reports/m2/eval.json` with, per predictor and slice: `acc`, `ece15`, `nll`, `flip`, and for non-baseline predictors
    `delta, lo, hi`.
14. (2.5 h) Write `reports/m2/README.md`: the curve plot (`reports/m2/curve.png`: x = 0, 1k, 3k decisions with B0 at 0; y =
    accuracy per slice with CI bars), the delta table with 95% CIs, flip rate vs B0, the training-loss plot from both
    `log.jsonl` files, wall time and peak memory, the skeleton reproduction line, and the decision taken under the
    pre-registered rule with a link to the prereg commit. `uv run ruff format . && uv run ruff check --fix .`,
    `uv run pytest -q` (expect `49 passed`: 45 before plus 4 in `tests/test_train.py`), commit
    `phase5: M2 learning curve and kill decision`, log hours in `ledger/hours.csv`.
15. (1.5 h) Buffer for interruptions and one extra resume. Steps sum to 24.0 h; the buffer covers one lost evening.

**Verification gate**

- `uv run python tools/loc.py` exits 0 with `train.py` included (core total at most 1,000 lines).
- `uv run pytest -q` prints `49 passed`; `uv run pytest gates/test_overfit32.py -q` prints `1 passed`.
- `NANOHUNCH_ADAPTER=<dir> uv run pytest gates/test_numerics_gate.py -q` prints `4 passed` for `runs/overfit32/adapter`,
  `runs/skeleton_repro/adapter`, `runs/curve_1k/adapter` and `runs/curve_3k/adapter`. No overnight run starts before the
  overfit32 gate passes.
- `reports/skeleton.json`: `abs(repro.all.acc - lora.all.acc) <= 0.02`.
- Each curve `log.jsonl`: every `loss` finite, mean loss of the last 10% of steps below the first 10%, `peak_mem_gb` under 24.
- `git log --format=%cI -1 -- reports/m2/prereg.md` is earlier than the `ts` of the first line of `runs/curve_3k/log.jsonl`.
- `reports/m2/README.md` committed by end of week 7 (R17 calendar rule).

**Rollback**

- Undo: `git revert` the phase commits; `rm -rf runs/curve_* runs/skeleton_repro runs/overfit32`. Phases 1 to 4 artefacts are
  untouched (this phase only appends a `repro` block to `reports/skeleton.json`; delete that key to undo).
- Point of no return: the start of step 11. Before it, a change to `fmt.py` costs one overfit rerun (20 min) and one
  skeleton repro (45 min). After it, every adapter encodes nanohunch-fmt-v1 and a format change means retraining all of them plus
  Phase 6.
- Data written during a failed window: checkpoints are written as `*.tmp` and renamed, so a crash leaves either a complete
  `ckpt-<step>` or an ignored `.tmp`; resume truncates `log.jsonl` to the resumed step, so no duplicate lines. `adapter/` exists
  only after a finished run. `reports/m2/` is committed only in step 14; a half-written eval is rerun from scratch (about 30
  min).

**Kill criterion**

- **Data scaling (tied to the ledger assumption that soft teacher labels generalise beyond the templates trained on).** If the
  held-out-template delta over calibrated B0 at 3k decisions (gold accuracy on `test_ood` spec-fact decisions, CI
  resampled by `state_id`) is below +1 pt or its 95% CI crosses 0, then stop scaling data, skip Phase 6, and go to Phase 7
  publishing the B0 + engine + calibration + order-robustness story.
- **Numerics.** If `gates/test_numerics_gate.py` still fails after 8 h of debugging (3 h inside this phase's budget, the rest
  from week-7 slack), then train the curve with stock `mlx_lm.lora` exactly as in Phase 1 step 7 (hard one-hot labels,
  full-vocab CE on the completion token, no augmentation), mark `train.py` as incomplete in `reports/m2/README.md`, and
  apply the data-scaling rule to those adapters.
