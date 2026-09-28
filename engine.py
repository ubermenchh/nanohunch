"""Inference engine: label readout, state-once KV-Cache branching, pooling"""

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

from fmt import Perm, QType, Question, Rendered


def label_logits(hidden_last: mx.array, head_weight: mx.array, label_ids: mx.array) -> mx.array:
    if head_weight.dtype == mx.uint32:
        raise TypeError("quantized LM head: load the model in bf16 so label rows can be gathered")
    w = head_weight[label_ids]
    return hidden_last.astype(mx.float32) @ w.astype(mx.float32).T


@dataclass(frozen=True, slots=True)
class BranchLogits:
    question_id: str
    perm: Perm
    logits: np.ndarray


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def pool(branches: Sequence[BranchLogits]) -> dict[str, np.ndarray]:
    by_q: dict[str, list[np.ndarray]] = {}
    for b in branches:
        canon = np.empty(len(b.logits))
        canon[list(b.perm)] = _log_softmax(b.logits.astype(np.float64))
        by_q.setdefault(b.question_id, []).append(canon)
    return {q: _log_softmax(np.mean(lps, axis=0)) for q, lps in by_q.items()}


@dataclass(frozen=True, slots=True)
class Answer:
    question_id: str
    qtype: QType
    probs: np.ndarray
    confidence: float
    entropy_norm: float
    expected: float | None


def to_answer(q: Question, probs: np.ndarray) -> Answer:
    p = np.asarray(probs, dtype=np.float64)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    if q.type == "score":
        expected = float(np.dot(q.values, p))
    elif q.type == "noul":
        expected = float(p[0])
    else:
        expected = None
    return Answer(q.id, q.type, p, float(p.max()), float(entropy / np.log(len(p))), expected)


class MLXBranchScorer:
    def __init__(self, model_path, *, adapter_path=None, prefill_chunk=1024, dtype="bfloat16"):
        self.model, self.tok = load(model_path, adapter_path=adapter_path)
        if dtype == "float32":
            self.model.update(tree_map(lambda p: p.astype(mx.float32), self.model.parameters()))
        self.dtype, self.prefill_chunk = dtype, prefill_chunk

        text_model = getattr(self.model, "language_model", self.model)
        self._backbone = text_model.model

        tied = text_model.args.tie_word_embeddings
        self._head_weight = self._backbone.embed_tokens.weight if tied else text_model.lm_head.weight

    def score(self, r: Rendered) -> list[BranchLogits]:
        cache = make_prompt_cache(self.model)
        if not can_trim_prompt_cache(cache):
            raise RuntimeError("cache not trimmable")
        for i in range(0, len(r.prefix_ids), self.prefill_chunk):
            self._backbone(mx.array(r.prefix_ids[i : i + self.prefill_chunk])[None], cache=cache)
            mx.eval([c.state for c in cache])
        p = cache[0].offset
        assert p == len(r.prefix_ids), (p, len(r.prefix_ids))
        out = []
        for b in r.branches:  # batch 1 only
            h = self._backbone(mx.array(b.token_ids)[None], cache=cache)[:, -1, :]
            z = label_logits(h, self._head_weight, mx.array(b.label_ids))
            mx.eval(z)
            trim_prompt_cache(cache, len(b.token_ids))
            assert cache[0].offset == p, (cache[0].offset, p)
            out.append(BranchLogits(b.question_id, b.perm, np.array(z[0], dtype=np.float32)))
        return out

    def score_reencode(self, r: Rendered) -> list[BranchLogits]:
        out = []
        for b in r.branches:  # the oracle: no cache, one full forward of prefix + branch each
            h = self._backbone(mx.array(r.prefix_ids + b.token_ids)[None])[:, -1, :]
            z = label_logits(h, self._head_weight, mx.array(b.label_ids))
            out.append(BranchLogits(b.question_id, b.perm, np.array(z[0], dtype=np.float32)))
        return out
