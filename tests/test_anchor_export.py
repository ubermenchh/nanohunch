"""Export only public SemIf/JeVBench anchors into the common eval-row format."""

import json


def test_prepare_anchors_writes_converted_public_only_rows(tmp_path):
    """Anchor export uses project converters and canonicalizes the SemIf yes/no order."""
    from tools.prepare_anchors import prepare_anchors

    semif = tmp_path / "authored144.jsonl"
    semif.write_text(
        json.dumps(
            {
                "id": "semi-1",
                "group_id": "case-1",
                "family": "policy",
                "state": "Receipt missing.",
                "question": "Refund?",
                "options": [{"id": "no", "description": "No"}, {"id": "yes", "description": "Yes"}],
                "label": 0,
                "split": "test",
                "target_distribution": [0.8, 0.2],
            }
        )
        + "\n"
    )
    jev = tmp_path / "jev.jsonl"
    jev.write_text(
        json.dumps(
            {
                "id": "jev-1",
                "group": "request-1",
                "family": "policy",
                "state": "Rule says no.",
                "question": {"type": "choice", "instructions": "Choose."},
                "labels": ["A", "B"],
                "expected": "B",
                "split": "public",
            }
        )
        + "\n"
    )

    counts = prepare_anchors(semif, [jev], tmp_path / "out")

    semi_row = json.loads((tmp_path / "out/semif_authored144.jsonl").read_text())
    jev_row = json.loads((tmp_path / "out/jevbench_public.jsonl").read_text())
    assert counts == {"semif_authored144": 1, "jevbench_public": 1}
    assert (semi_row["options"], semi_row["gold"]) == (["yes: Yes", "no: No"], 1)
    assert semi_row["source"] == "semif" and semi_row["consensus"] == [0.2, 0.8]
    assert (jev_row["source"], jev_row["gold"]) == ("jevbench", 1)
