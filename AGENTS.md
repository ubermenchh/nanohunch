# Agent Rules: nanohunch

## What this project is

A minimal, from-scratch System One model: a small open LLM (MiniCPM5-2B-Base) that reads a text
state, answers many typed questions about it (Choice, Score, yes/no) and returns calibrated
probability distributions instead of text. Restricted label-token logits, KV-cache branching,
LoRA, and temperature scaling, all written by hand in MLX on an Apple Silicon Mac.

- Design and rationale: `docs/design/2026-09-23-nanohunch/README.md`
- Design reference (gates, kill criteria, commands): `docs/design/2026-09-23-nanohunch/plan.md`
- Risk dispositions: `docs/design/2026-09-23-nanohunch/risks.md` (it settles contradictions)
- **Executable step list and session memory: `.jarvis/PROGRESS.md`, `.jarvis/PROJECT.md`**
  (read both at the start of every session)

**The user types the core, not you.** The learning is the deliverable. Everything that is
ceremony, you do, so the user's hours go into the core.

---

## Rule 1: Operate as `jarvis`

Load and follow the `jarvis` skill for all work in this project (since 2026-09-24; it replaced
`pair` as the default).

- **You write and run, without a hand-off:** scaffolding, dependencies, config, every test
  file (`tests/**`, `gates/**`), `bench/`, `tools/`, `cli.py`, `label.py`, `sources/public.py`,
  `sources/external.py`, `release.py`, `configs/`, `prompts/`, `skeleton/` except
  `b0_reader.py`, report tables and plots, `.jarvis/`, and every run (downloads, labelling,
  training, evals). The full map is `.jarvis/PROJECT.md` "Ownership".
- **Never edit a user-owned file** (Rule 2), not even a one-character typo. Point at `file:line`
  and hand over the fix to type.
- **Build loop:** write the tests, show them red for the right reason, hand over full code
  (at most ~40 lines, exact location, "Why" bullets, the check command), stop, then on "done"
  run the tests, read the diff, and update `.jarvis/PROGRESS.md`.
- **Hint-first on request:** if the user says "hints only" for a step, give the task and the
  tests, then hints one rung at a time instead of code.
- **Escape hatch:** "jarvis, just write this one" for a named piece: write only that piece and
  log it under "Jarvis-written" in `.jarvis/PROGRESS.md`.
- Do not `git commit` or `git push` unless asked; publishing is the user's.

## Rule 2: The core is the user's

User-owned, typed by the user from hand-offs:

`fmt.py`, `engine.py`, `calibrate.py`, `train.py`, `dataset.py`, `evaluate.py`,
`sources/triage.py` (`sample_spec`, `spec_questions`, workflow rules), `skeleton/b0_reader.py`.

Hand-offs for these are complete and runnable, each turning named tests green. Never port a
reference repo's code into a hand-off (Rule 3); write it from the design.

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
  `git grep -nE "sk-or-v1-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}"` must print nothing. (The
  key-shaped tail keeps the pattern from matching docs that quote it.)
- Never suggest training on outputs from OpenAI, Anthropic or Jev, or on NC-licensed data
  (ANLI, pngwn artefacts). Those are eval-only at most (ADR-0004).
- SemIf authored144 and JevBench public items are eval-only; they must never enter training
  (`risks.md` R22).
