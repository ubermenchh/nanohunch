"""Predictor contracts: exercise wiring with fake render/scorer boundaries, no MLX weights."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fake_engine(monkeypatch):
    """Replace only model-facing seams so predictor behavior stays deterministic."""
    import fmt

    calls = {}

    class Scorer:
        def __init__(self, model, *, adapter_path=None):
            calls["init"] = (model, adapter_path)
            self.tok = object()

        def score(self, rendered):
            calls["score"] = rendered
            return ["branch"]

        def score_reencode(self, rendered):
            calls["reencode"] = rendered
            return ["branch"]

    engine = ModuleType("engine")
    engine.MLXBranchScorer = Scorer
    engine.pool = lambda branches: {"q1": np.log([0.75, 0.25])}
    engine.to_answer = lambda question, probs: SimpleNamespace(
        question_id=question.id, probs=probs, confidence=float(probs.max())
    )
    monkeypatch.setitem(sys.modules, "engine", engine)
    monkeypatch.setattr(
        fmt,
        "render",
        lambda tok, state, questions, *, n_perms, max_context: (
            calls.update(render=(tok, state, questions, n_perms, max_context)) or "rendered"
        ),
    )
    return calls


def test_b0_predictor_returns_canonical_answer_and_supports_reencode(fake_engine):
    """B0 preserves pooled option order and chooses the configured scoring oracle."""
    import evaluate
    from fmt import Question

    question = Question("q1", "choice", "Pick", ("first", "second"))
    predictor = evaluate.B0Predictor("toy-model", reencode=True, max_context=1234)

    answer = predictor.predict("evidence", [question], n_perms=2)[0]

    assert answer.probs == pytest.approx([0.75, 0.25])
    assert answer.question_id == "q1"
    assert fake_engine["render"][1:] == ("evidence", [question], 2, 1234)
    assert fake_engine["reencode"] == "rendered"
    assert "score" not in fake_engine


def test_b0_predictor_exposes_unscaled_pooled_logprobs(fake_engine):
    """Calibration fitting can consume T=1 pooled log-probabilities directly."""
    import evaluate
    from fmt import Question

    question = Question("q1", "choice", "Pick", ("first", "second"))
    predictor = evaluate.B0Predictor("toy-model")

    pooled = predictor.predict_logprobs("evidence", [question], n_perms=2)

    assert pooled["q1"] == pytest.approx(np.log([0.75, 0.25]))
    assert fake_engine["score"] == "rendered"


def test_adapter_predictor_loads_adapter_and_reuses_prediction_path(fake_engine):
    """Adapter inference uses the same canonical scoring path with adapter weights applied."""
    import evaluate
    from fmt import Question

    question = Question("q1", "choice", "Pick", ("first", "second"))
    predictor = evaluate.AdapterPredictor("toy-model", "adapter-dir")

    answer = predictor.predict("evidence", [question])[0]

    assert predictor.name == "adapter"
    assert fake_engine["init"] == ("toy-model", "adapter-dir")
    assert answer.probs == pytest.approx([0.75, 0.25])
    assert fake_engine["score"] == "rendered"
