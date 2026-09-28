"""Teacher requests, label extraction, and an auditable JSONL response cache."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
from dotenv import load_dotenv

from dataset import renormalize
from fmt import FORMAT_VERSION
from skeleton.fmt_ref import branch_text, label_strings, prefix_text

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MIN_CANDIDATE_MASS = 0.9


@dataclass(frozen=True, slots=True)
class TeacherSpec:
    teacher_id: str
    model: str
    provider_order: tuple[str, ...]
    top_logprobs: int = 20


class CreditExhausted(RuntimeError):
    """The provider rejected a request because the account has no credit."""


class TeacherUnavailable(RuntimeError):
    """No provider endpoint currently supports the requested model parameters."""


def request_body(spec: TeacherSpec, prompt: str, system: str) -> dict:
    return {
        "model": spec.model,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": spec.top_logprobs,
        "reasoning": {"enabled": False},
        "usage": {"include": True},
        "provider": {
            "order": list(spec.provider_order),
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }


def cache_key(
    teacher_id: str,
    model_version: str,
    format_version: str,
    state_id: str,
    question_id: str,
    perm: tuple[int, ...],
    prompt: str,
) -> str:
    parts = (
        teacher_id,
        model_version,
        format_version,
        state_id,
        question_id,
        ",".join(map(str, perm)),
        hashlib.sha256(prompt.encode()).hexdigest(),
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _read_cache(path: Path) -> dict[str, dict]:
    found = {}
    if not path.exists():
        return found
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("key"), str):
            found[row["key"]] = row
    return found


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _request(client: httpx.Client, body: dict) -> dict:
    for attempt in range(3):
        try:
            response = client.post(ENDPOINT, json=body)
        except httpx.TimeoutException:
            if attempt == 2:
                raise
            time.sleep(2 * 2**attempt * random.uniform(0.5, 1.5))
            continue
        if response.status_code == 402:
            raise CreditExhausted("teacher account has insufficient credit")
        if response.status_code == 404:
            message = response.json().get("error", {}).get("message", "")
            if message.startswith("No endpoints found"):
                raise TeacherUnavailable(message)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 2:
                time.sleep(2 * 2**attempt * random.uniform(0.5, 1.5))
                continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("teacher request retries exhausted")


def _value(obj, key: str):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _prompt_parts(state, question, perm: tuple[int, ...]) -> tuple[str, list[str], list[int]]:
    state_text = _value(state, "state") if isinstance(state, dict) else str(state)
    question_type = _value(question, "qtype") if isinstance(question, dict) else question.type
    question_text = _value(question, "text") if isinstance(question, dict) else question.text
    options = _value(question, "options") if isinstance(question, dict) else question.options
    values = _value(question, "values") if isinstance(question, dict) else question.values
    option_texts = [option["text"] if isinstance(option, dict) else option for option in options]
    displayed_options = [option_texts[i] for i in perm]
    displayed_values = [values[i] for i in perm] if question_type == "score" else []
    prompt = prefix_text(state_text) + branch_text(question_type, question_text, displayed_options, displayed_values)
    return prompt, displayed_options, displayed_values


def _system_line(qtype: str) -> str:
    return {
        "choice": "Answer with a capital letter only.",
        "noul": "Answer yes or no only.",
        "score": "Answer with the number of one level only.",
    }[qtype]


def label_decision(
    spec: TeacherSpec,
    state,
    q,
    perm: tuple[int, ...],
    *,
    cache: Path,
    option_ids: tuple[str, ...],
    client: httpx.Client | None = None,
) -> dict | None:
    qid = _value(q, "id") if isinstance(q, dict) else q.id
    qtype = _value(q, "qtype") if isinstance(q, dict) else q.type
    values = _value(q, "values") if isinstance(q, dict) else q.values
    prompt, _displayed, _displayed_values = _prompt_parts(state, q, perm)
    state_id = state.get("state_id") if isinstance(state, dict) else None
    if state_id is None:
        state_text = _value(state, "state") if isinstance(state, dict) else str(state)
        state_id = hashlib.sha256(state_text.encode()).hexdigest()
    key = cache_key(spec.teacher_id, spec.model, FORMAT_VERSION, state_id, qid, perm, prompt)
    cached = _read_cache(cache).get(key)
    if cached is not None:
        return cached

    labels = label_strings(qtype, len(option_ids), values)
    body = request_body(spec, prompt, _system_line(qtype))
    owned_client = client is None
    if owned_client:
        load_dotenv()
        token = os.getenv("OPENROUTER_API_KEY")
        if not token:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {token}"})
    try:
        response = _request(client, body)
        choice = response["choices"][0]
        logprobs = choice.get("logprobs")
        top = (logprobs or {}).get("content", [{}])[0].get("top_logprobs", [])
        mass, probs = renormalize(top or [], labels, qtype)
        if mass < MIN_CANDIDATE_MASS:
            response = _request(client, body)
            choice = response["choices"][0]
            logprobs = choice.get("logprobs")
            top = (logprobs or {}).get("content", [{}])[0].get("top_logprobs", [])
            mass, probs = renormalize(top or [], labels, qtype)
        if mass < MIN_CANDIDATE_MASS:
            _append_jsonl(
                cache.parent / "dropped.jsonl",
                {
                    "key": key,
                    "teacher_id": spec.teacher_id,
                    "question_id": qid,
                    "mass": mass,
                    "reason": "low_candidate_mass",
                },
            )
            return None
    finally:
        if owned_client:
            client.close()

    probs_by_option_id = {option_ids[canonical_i]: probs[display_i] for display_i, canonical_i in enumerate(perm)}
    usage = response.get("usage", {})
    row = {
        "key": key,
        "teacher_id": spec.teacher_id,
        "model_version": spec.model,
        "host": response.get("provider", spec.provider_order[0]),
        "date": date.today().isoformat(),
        "format_version": FORMAT_VERSION,
        "perm": list(perm),
        "labels": labels,
        "top_logprobs": top,
        "candidate_mass": mass,
        "probs_by_option_id": probs_by_option_id,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "usd": usage.get("cost", 0.0),
    }
    _append_jsonl(cache, row)
    return row
