"""Phase 3 source adapters: normalize external rows without training on eval-only data."""

import dataclasses
import json
import sys
import types

import pytest


@pytest.fixture
def adapters(monkeypatch):
    """Provide only the EvalItem shape so glue tests do not invent the user's evaluator core."""

    @dataclasses.dataclass(frozen=True)
    class EvalItem:
        state_id: str
        group_key: str
        state: str
        question: object
        gold: int | None
        consensus: object | None
        meta: dict

    fake = types.ModuleType("evaluate")
    fake.EvalItem = EvalItem
    monkeypatch.setitem(sys.modules, "evaluate", fake)
    from sources import external, public

    return external, public


def test_norm_hash_collapses_case_and_whitespace(adapters):
    external, _ = adapters

    assert external.norm_hash(" A\n  B ") == external.norm_hash("a b")
    assert len(external.norm_hash("a b")) == 64


def test_eval_only_hash_export_contains_normalized_state_and_question(adapters, tmp_path):
    external, _ = adapters
    source = tmp_path / "items.jsonl"
    source.write_text('{"state":" A  B ","question":"Will it ship?"}\n')
    output = tmp_path / "hashes.txt"

    count = external.write_eval_only_hashes([source], output)

    assert count == 2
    assert output.read_text().splitlines() == sorted([external.norm_hash("a b"), external.norm_hash("will it ship?")])


def test_semif_conversion_preserves_label_option_alignment(adapters):
    external, _ = adapters
    row = {
        "id": "semif-1",
        "group_id": "case-1",
        "family": "evidence",
        "state": "The report says shipment arrived.",
        "question": "Did it arrive?",
        "options": [{"id": "yes", "description": "Arrived"}, {"id": "no", "description": "Not arrived"}],
        "label": 1,
        "target_distribution": [0.2, 0.8],
    }

    item = external.convert_semif(row)

    assert item.question.options == ("yes: Arrived", "no: Not arrived")
    assert item.gold == 1
    assert item.consensus.tolist() == pytest.approx([0.2, 0.8])
    assert item.meta["family"] == "evidence"
    assert item.meta["source"] == "semif"


def test_jev_noul_is_canonical_yes_no_even_if_source_is_reversed(adapters):
    external, _ = adapters
    row = {
        "id": "jev-1",
        "group": "case-1",
        "family": "policy",
        "state": "Receipt absent.",
        "question": {"type": "noul", "instructions": "May we refund?"},
        "labels": ["no", "yes"],
        "expected": "no",
    }

    item = external.convert_jevbench(row)

    assert item.question.options == ("yes", "no")
    assert item.gold == 1
    assert item.meta["source"] == "jevbench"


def test_jev_structured_state_is_serialized_without_losing_fields(adapters):
    external, _ = adapters
    state = {"policy": ["Require approval."], "request": {"text": "Proceed?"}}
    row = {
        "id": "jev-structured",
        "group": "case-structured",
        "family": "policy",
        "state": state,
        "question": {"type": "choice", "instructions": "What next?"},
        "labels": ["wait", "proceed"],
        "expected": "wait",
    }

    item = external.convert_jevbench(row)

    assert item.state == json.dumps(state, ensure_ascii=False, sort_keys=True)


def test_pngwn_uses_configured_fields_and_skips_unknown_type(adapters):
    external, _ = adapters
    fields = {
        "state": "context",
        "question": "prompt",
        "options": "choices",
        "type": "kind",
        "gold": "answer",
        "distribution": "probs",
    }
    row = {
        "context": "Order is delayed.",
        "prompt": "Was it delivered?",
        "choices": ["yes", "no"],
        "kind": "noul",
        "answer": 0,
        "probs": [0.7, 0.3],
    }

    item = external.convert_pngwn(row, fields)

    assert item.question.options == ("yes", "no")
    assert item.gold == 0
    assert item.consensus.tolist() == pytest.approx([0.7, 0.3])
    assert external.convert_pngwn(row | {"kind": "unsupported"}, fields) is None


def test_public_row_converters_preserve_gold_and_group(adapters):
    _, public = adapters

    boolq = public.convert_boolq({"passage": "A package arrived.", "question": "Did it arrive?", "answer": True})
    arc = public.convert_arc(
        {
            "id": "arc-7",
            "question": "Pick one",
            "choices": {"label": ["A", "B", "C"], "text": ["x", "y", "z"]},
            "answerKey": "B",
        }
    )
    csqa = public.convert_csqa(
        {
            "id": "csqa-2",
            "question": {"stem": "Pick one", "choices": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}]},
            "answerKey": "A",
        }
    )
    csqa_hub = public.convert_csqa(
        {"id": "csqa-3", "question": "Pick one", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "B"}
    )
    hotpot = public.convert_hotpot(
        {
            "_id": "hp-3",
            "context": {"title": ["T"], "sentences": [["One.", "Two."]]},
            "question": "True?",
            "answer": "yes",
        }
    )
    hotpot_hub = public.convert_hotpot(
        {"id": "hp-4", "context": {"title": ["T"], "sentences": [["One."]]}, "question": "True?", "answer": "no"}
    )

    assert (boolq["qtype"], boolq["gold"], boolq["group_key"]) == ("noul", 0, boolq["group_key"])
    assert (arc["qtype"], arc["gold"], arc["group_key"]) == ("choice", 1, "arc-7")
    assert (csqa["qtype"], csqa["gold"], csqa["group_key"]) == ("choice", 0, "csqa-2")
    assert (csqa_hub["state"], csqa_hub["options"], csqa_hub["gold"]) == (
        "Pick one",
        ["x", "y"],
        1,
    )
    assert hotpot["state"] == "T\nOne. Two.\n\n"
    assert (hotpot["qtype"], hotpot["gold"], hotpot["group_key"]) == ("noul", 0, "hp-3")
    assert (hotpot_hub["id"], hotpot_hub["gold"], hotpot_hub["group_key"]) == (
        "hotpot:hp-4",
        1,
        "hp-4",
    )


def test_hotpot_padding_stays_in_the_base_groups_split(adapters, monkeypatch):
    _, public = adapters
    from dataset import assign_split

    rows = [
        {"id": f"hp:{i}", "group_key": f"hp-{i}", "source": "hotpot", "state": "x" * 520, "meta": {}}
        for i in range(150)
    ]
    rows.extend(
        {"id": f"pad-source:{i}", "group_key": f"pad-source-{i}", "source": "hotpot", "state": "y" * 30, "meta": {}}
        for i in range(150, 300)
    )
    fractions = {"train": 0.8, "cal": 0.08, "test": 0.12}
    monkeypatch.setattr(public, "_n_tokens", lambda _tok, row: len(row["state"]))

    pads = public._pad_hotpot(rows[:150], object(), (1900, 3900, 7600))

    assert len(pads) == 450
    for base in rows[:150]:
        variants = [row for row in pads if row["group_key"] == base["group_key"]]
        assert len(variants) == 3
        assert {assign_split(row["group_key"], row["source"], fractions, "test-salt") for row in variants} == {
            assign_split(base["group_key"], base["source"], fractions, "test-salt")
        }


def test_hotpot_padding_skips_bases_too_long_for_short_bucket(adapters, monkeypatch):
    """Each reused base must fit the smallest target bucket before padding starts."""
    _, public = adapters
    too_long = {"id": "too-long", "group_key": "too-long", "source": "hotpot", "state": "x" * 2000, "meta": {}}
    bases = [
        {"id": f"base:{i}", "group_key": f"base:{i}", "source": "hotpot", "state": "b" * 520, "meta": {}}
        for i in range(150)
    ]
    candidates = [
        {"id": f"candidate:{i}", "group_key": f"candidate:{i}", "source": "hotpot", "state": "c" * 30, "meta": {}}
        for i in range(150)
    ]
    monkeypatch.setattr(public, "_n_tokens", lambda _tok, row: len(row["state"]))

    pads = public._pad_hotpot([too_long, *bases, *candidates], object(), (1900, 3900, 7600))

    assert len(pads) == 450
    assert all(row["group_key"] != "too-long" for row in pads)
