"""Reference nanohunch-fmt-v1 text (ADR-0003 with Amendment 1), shared by bench/ and skeleton/.

Text level only. Two id-level rules live with whoever encodes: the prefix and the branch are
encoded separately and concatenated (rule 2), and a Score branch gets `space_id(tok)` appended
after "Answer:" so the bare-digit labels are read at that position (Amendment 1, option (b)).
Your `fmt.py` owns its own copy of this text; the tests compare the two.
"""

from collections.abc import Sequence

PREAMBLE = "You will answer questions about the STATE. Answer with the label of exactly one option.\n\n"


def prefix_text(state: str) -> str:
    return f"{PREAMBLE}### STATE\n{state}\n### END STATE\n\n"


def option_lines(qtype: str, options: Sequence[str], values: Sequence[int] = ()) -> str:
    if qtype == "choice":
        return "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    if qtype == "score":
        return "\n".join(f"{v}: {o}" for v, o in zip(values, options, strict=True))
    return "(yes/no)"


def branch_text(qtype: str, question: str, options: Sequence[str], values: Sequence[int] = ()) -> str:
    return f"### QUESTION\n{question}\n{option_lines(qtype, options, values)}\nAnswer:"


def label_strings(qtype: str, n: int, values: Sequence[int] = ()) -> list[str]:
    if qtype == "choice":
        return [f" {chr(65 + i)}" for i in range(n)]
    if qtype == "score":
        return [str(v) for v in values]
    return [" yes", " no"]


def space_id(tok) -> int:
    """The single id that "Answer: " adds after "Answer:" (242 MiniCPM5, 220 Qwen3; P0-1)."""
    ctx = tok.encode("Answer:", add_special_tokens=False)
    with_space = tok.encode("Answer: ", add_special_tokens=False)
    if len(with_space) != len(ctx) + 1 or with_space[: len(ctx)] != ctx:
        raise ValueError(f"'Answer: ' -> {with_space} is not 'Answer:' {ctx} plus one space id")
    return with_space[-1]
