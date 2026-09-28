"""Phase 4 teacher-label math and response-cache contracts."""

import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest


def test_renormalize_candidates_discards_non_label_mass():
    from dataset import renormalize

    top = [
        {"token": " A", "logprob": -0.22},
        {"token": " B", "logprob": -1.9},
        {"token": " C", "logprob": -3.5},
        {"token": "The", "logprob": -2.0},
    ]

    mass, probs = renormalize(top, [" A", " B", " C"], "choice")

    assert mass == pytest.approx(0.982, abs=0.005)
    np.testing.assert_allclose(probs, [0.817, 0.152, 0.031], atol=0.002)


def test_pool_two_orders_cancels_position_bias():
    from dataset import pool_orders

    def distribution(logits, order):
        displayed = np.asarray(logits)[list(order)] + np.asarray([0.5] + [0.0] * (len(order) - 1))
        probs = np.exp(displayed - displayed.max())
        probs /= probs.sum()
        return {f"option-{i}": float(p) for i, p in zip(order, probs, strict=True)}

    two_logits = [0.2, 0.0]
    pooled = pool_orders(distribution(two_logits, (0, 1)), distribution(two_logits, (1, 0)))
    expected = np.exp(two_logits)
    expected /= expected.sum()
    np.testing.assert_allclose([pooled["option-0"], pooled["option-1"]], expected, atol=1e-9)

    four_logits = [0.0, 0.3, 0.0, 0.0]
    forward = distribution(four_logits, (0, 1, 2, 3))
    backward = distribution(four_logits, (3, 2, 1, 0))
    pooled_four = pool_orders(forward, backward)
    assert max(forward, key=forward.get) == "option-0"
    assert max(backward, key=backward.get) == "option-3"
    assert max(pooled_four, key=pooled_four.get) == "option-1"


def test_consensus_averages_teacher_probabilities_by_option_id():
    from dataset import consensus

    result = consensus(
        [
            {"option-a": 0.8, "option-b": 0.2},
            {"option-b": 0.6, "option-a": 0.4},
        ]
    )

    assert result == pytest.approx({"option-a": 0.6, "option-b": 0.4})


def test_label_decision_cache_is_idempotent_and_excludes_headers(tmp_path):
    from fmt import Question
    from label import label_decision

    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "provider": "mock-host",
                "model": "mock-model-revision",
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.0},
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": " A", "logprob": -0.2},
                                        {"token": " B", "logprob": -2.5},
                                    ]
                                }
                            ]
                        }
                    }
                ],
            },
        )

    spec = SimpleNamespace(
        teacher_id="qwen3.6-35b-a3b",
        model="mock-model",
        provider_order=("mock-host",),
        top_logprobs=20,
    )
    state = {"state_id": "state-1", "state": "The evidence."}
    question = Question("q1", "choice", "Which?", ("yes", "no"))
    cache = tmp_path / "cache.jsonl"

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        first = label_decision(
            spec,
            state,
            question,
            (0, 1),
            cache=cache,
            option_ids=("q1:0", "q1:1"),
            client=client,
        )
        second = label_decision(
            spec,
            state,
            question,
            (0, 1),
            cache=cache,
            option_ids=("q1:0", "q1:1"),
            client=client,
        )

    assert first == second
    assert calls == 1
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert len(rows) == 1
    assert not {"headers", "authorization"} & rows[0].keys()
