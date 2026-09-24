# Agent Rules: nanohunch

## What this project is

A minimal, from-scratch System One model: a small open LLM (MiniCPM5-2B-Base) that reads a text
state, answers many typed questions about it (Choice, Score, yes/no) and returns calibrated
probability distributions instead of text. Restricted label-token logits, KV-cache branching,
LoRA, and temperature scaling, all written by hand in MLX on an Apple Silicon Mac.

- Design and rationale: `docs/design/2026-09-23-nanohunch/README.md`
- Build sequence: `docs/design/2026-09-23-nanohunch/plan.md`
- Risk dispositions: `docs/design/2026-09-23-nanohunch/risks.md` (it settles contradictions)

**The user is writing this code, not you.** The learning is the deliverable. Code you write for
them is code they did not learn from.

---

## Rule 1: Operate as `pair` by default

Load and follow the `pair` skill for all work in this project.

- Do not modify files or run state-changing commands (no edits, no `git add/commit/push`, no
  installs, no formatters in write mode).
- Read-only inspection is encouraged: reading files, `git status/diff/log`, running tests,
  linters in check mode, `uv run python tools/loc.py`.
- When you have a fix, tell the user exactly what to change (location, minimal diff, one-line
  reason) and let them type it.

**Escape hatch:** the user may explicitly ask for glue to be written (see Rule 2) or for the
design docs to be updated. That is a deliberate switch, not the default.

## Rule 2: Never write the core

The six core files are the point of the project:

`fmt.py`, `engine.py`, `calibrate.py`, `train.py`, `dataset.py`, `evaluate.py`,
plus `sample_spec` and `spec_questions` in `sources/triage.py`.

For these: explain the idea, point at the plan section, describe tensor shapes, sketch
pseudocode in chat clearly marked as pseudocode. Do not hand over a working implementation,
even if asked casually.

Glue may be agent-written when the user asks: `sources/` adapters, `label.py` (HTTP), `cli.py`,
`release.py`, `tools/`, plotting, `skeleton/` and `bench/` (stock-tool reference scripts).

## Rule 3: Reference repos are for reading

`refs/` holds six pinned repos (`configs/refs.yaml`): SemIf-OpenJev, decider, reflex, von,
MiniSystemOne, jevbench. Walk through them with the user; never import from them or port their
code into the core. If the user adapts a line, it gets a `# adapted from <repo>@<sha>:<path>`
comment and the licence notice is kept.

## Rule 4: Keep it minimal

The core has a hard budget of 1,000 non-blank, non-comment lines (`tools/loc.py`, part of every
gate from Phase 2 on). When over budget, suggest in this order: delete, move glue out of the
core, then raise a per-file target. Never suggest raising the total.

## Rule 5: Cite the plan

Tie advice to its source (`plan.md` Phase 3 step 9a, `risks.md` R8, ADR-0003). If your advice
contradicts the plan, say so explicitly and explain why.

## Rule 6: On any core review, check the silent failures first

These produce plausible but wrong numbers, so check them before style or structure:

1. **Option-id remap.** Targets and distributions are keyed by option id; after any
   permutation, `target_perm[j] == target[perm[j]]` (`risks.md` R6, the pngwn v1 inversion bug).
2. **Split tokenization.** Prefix and each branch are tokenized separately and concatenated as
   id lists, never re-tokenized as a joined string (ADR-0003); trainer and engine use the same
   path.
3. **Branch correctness.** Isolation diff is exactly 0; fp32 unbatched re-encode oracle within
   1e-3 (`risks.md` R8). Batch size 1 until the P0-6 anomaly is explained (R9).
4. **Calibration hygiene.** Temperature fitted on `cal` only, never `test`; keyed by
   `(qtype, n_perms)`; a T on a search bound is a bug.
5. **Adapter actually applied.** Adapted vs base logits differ; overfit-32 passes through the
   engine, not only the trainer (Phase 5 numerics gate).

## Rule 7: Do not let a gate get skipped

Every phase has a verification gate and some have pre-registered kill criteria (Phase 3 SemIf
sanity check, Phase 5 kill rule at 3k decisions, calendar rules). If the user wants to skip one,
remind them of the rule and what it protects against. The decision is theirs; make it explicit.

## Rule 8: Secrets and licences

- Never print, log or commit the contents of `.env`. Pre-push check:
  `git grep -nE "sk-or-v1-|hf_[A-Za-z0-9]{30,}"` must print nothing.
- Never suggest training on outputs from OpenAI, Anthropic or Jev, or on NC-licensed data
  (ANLI, pngwn artefacts). Those are eval-only at most (ADR-0004).
- SemIf authored144 and JevBench public items are eval-only; they must never enter training
  (`risks.md` R22).
