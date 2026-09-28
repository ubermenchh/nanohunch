# nanohunch

**What:** the minimal open System One model: MiniCPM5-2B-Base reads a text state, answers many
typed questions (Choice, Score, yes/no) through restricted label-token logits with KV-cache
branching, LoRA-tuned in MLX, temperature-calibrated. Core at most 1,000 lines.
**Why (user's goal):** learning + portfolio. Understand every line of the core because you typed
it; publish a model and an honest calibrated eval against SemIf, decider-2b and pngwn.
**Constraints:** solo, about 20 h/week; under 200 USD total (expected 5 to 15); Apple M5 24 GB
(19.07 GB Metal working set), rented GPU only if a plan trigger fires; Apache/MIT data and
releasable teachers only (ADR-0004).
**Stack:** Python 3.12.13 (uv, `.python-version`), mlx 0.32.2, mlx-lm 0.31.3, numpy 2.5.3;
pytest + ruff (added at scaffold). Base model `openbmb/MiniCPM5-2B-Base` (downloaded, 4.7 GB).
**Commands:** test `uv run pytest -q` · gates `uv run pytest gates/<file> -q` · lint
`uv run ruff check .` · budget `uv run python tools/loc.py`
**Design reference:** `docs/design/2026-09-23-nanohunch/plan.md` (gates, kill criteria, commands
per phase), `risks.md` (settles contradictions), `adr/`.
**.jarvis/ committed:** undecided (ask once; record here)

## Ownership
- **Jarvis** (writes and runs, no hand-off): `pyproject.toml`, `uv.lock`, `.python-version`,
  `.gitignore`, `.env.example`, ruff/pytest config, `tests/**`, `gates/**` (all test files,
  fixtures, mocks), `bench/**`, `tools/**`, `cli.py`, `label.py` (OpenRouter HTTP, cache,
  ledger), `sources/public.py`, `sources/external.py`, `release.py`, `configs/**`, `prompts/**`,
  `skeleton/**` except `skeleton/b0_reader.py`, report tables and plots, `.jarvis/**`,
  design-doc upkeep, all runs (downloads, labelling, training, evals).
- **User** (types from hand-offs; jarvis never edits): `fmt.py`, `engine.py`, `calibrate.py`,
  `train.py`, `dataset.py`, `evaluate.py`, `sources/triage.py` (`sample_spec`,
  `spec_questions`, and the workflow 2 and 3 rules), `skeleton/b0_reader.py`.
- **User, non-code:** design decisions flagged in PROGRESS "Pending decisions", the blind audits
  (100 + 200 items), reading decoded rows, pre-registration text, report and model-card prose,
  publishing (push, HF visibility flips, tags), `.env` keys and OpenRouter prepay.
- Decided gray areas:
  - `skeleton/b0_reader.py`: user (2026-09-24). First contact with the label readout; the
    `engine.py` generalization is compared against it.
  - Teacher-label math (`renormalize`, `pool_orders`, `consensus`) lives in `dataset.py`: user;
    the HTTP around it in `label.py`: jarvis (2026-09-24).
  - Reference metric scripts in `skeleton/`/`bench/` (ECE, candidate mass, JSD): jarvis; the
    user's core versions are tested against them (2026-09-24).

## User preferences
- 2026-09-24: learn by building; wants to know what and why for every piece.
- 2026-09-24: hand-offs are full code by default (jarvis contract). Say "hints only" on any
  step to switch that step to hint-first.
- 2026-09-24: English-only, no em or en dashes in generated docs.
