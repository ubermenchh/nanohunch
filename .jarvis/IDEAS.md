# Ideas

Scores 1-5 (higher is better; Effort and Risk inverted). Source: design plan "Not doing" list
and the open-replication survey (2026-09-24).

| # | Idea | Impact | Effort⁻¹ | Learning | Novelty | Risk⁻¹ | Status | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Minimal core (<= 1,000 lines), Mac-trained, calibrated eval vs open replications | 4 | 3 | 5 | 4 | 4 | chosen | prior art: SemIf (untrained), decider (11.9k lines) |
| 2 | P = 2 reversed permutation pooling at inference | 3 | 5 | 3 | 1 | 5 | parked | trigger: flip > 30% (B0) or agreement < 0.90 (trained) |
| 3 | Order-invariant option masks (von `option_marker.py`) | 4 | 2 | 5 | 3 | 2 | parked | changes the format: a v2 format and ADR |
| 4 | All questions in one sequence with a block mask (decider, reflex) | 3 | 2 | 4 | 2 | 3 | parked | latency for a demo; trim is simpler |
| 5 | Qwen3-4B-Base training on one Modal function | 4 | 3 | 2 | 1 | 4 | parked | trigger: Phase 3 Q4 (> 3 pts), likely per SemIf |
| 6 | Official JevBench submission (sealed items) | 3 | 4 | 2 | 2 | 4 | parked | after release |
| 7 | HTTP `/v1/decide` + Gradio demo on ZeroGPU | 3 | 4 | 2 | 1 | 5 | parked | after release |
| 8 | Qwen3.5 hybrid branching (recurrent state snapshots) | 2 | 2 | 5 | 2 | 2 | parked | learning stretch |
| 9 | Distill from Jev outputs | 3 | 4 | 1 | 1 | 1 | rejected | terms unverified, release not clean (ADR-0004) |
