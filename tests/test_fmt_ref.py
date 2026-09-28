"""Reference nanohunch-fmt-v1 text (ADR-0003 + Amendment 1). fmt.py is later checked against this."""


def test_prefix_matches_adr():
    from skeleton.fmt_ref import PREAMBLE, prefix_text

    assert PREAMBLE == "You will answer questions about the STATE. Answer with the label of exactly one option.\n\n"
    assert prefix_text("S") == PREAMBLE + "### STATE\nS\n### END STATE\n\n"


def test_choice_branch():
    from skeleton.fmt_ref import branch_text

    assert branch_text("choice", "Q?", ("x", "y")) == "### QUESTION\nQ?\nA. x\nB. y\nAnswer:"


def test_noul_score_branch_and_labels():
    from skeleton.fmt_ref import branch_text, label_strings

    assert branch_text("noul", "Q?", ("yes", "no")) == "### QUESTION\nQ?\n(yes/no)\nAnswer:"
    assert branch_text("score", "Q?", ("low", "high"), (0, 1)) == "### QUESTION\nQ?\n0: low\n1: high\nAnswer:"
    assert label_strings("noul", 2) == [" yes", " no"]
    assert label_strings("choice", 3) == [" A", " B", " C"]
    # Amendment 1 option (b): bare digits; the space id is appended at the id level, not in the text
    assert label_strings("score", 3, (0, 1, 2)) == ["0", "1", "2"]
