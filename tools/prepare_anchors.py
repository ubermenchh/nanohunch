"""Convert pinned public anchors to the common JSONL evaluation input schema."""

import argparse
import json
from pathlib import Path

from sources.external import convert_jevbench, convert_semif

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _row(item) -> dict:
    """Serialize the adapter's canonical EvalItem without changing option order."""
    return {
        "id": item.question.id,
        "state_id": item.state_id,
        "group_key": item.group_key,
        "state": item.state,
        "qtype": item.question.type,
        "question": item.question.text,
        "options": list(item.question.options),
        "values": list(item.question.values),
        "gold": item.gold,
        "consensus": None if item.consensus is None else item.consensus.tolist(),
        "source": item.meta["source"],
        "meta": item.meta,
    }


def _write(path: Path, items: list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(_row(item), ensure_ascii=False, sort_keys=True) + "\n" for item in items))
    return len(items)


def prepare_anchors(semif_path: Path, jev_paths: list[Path], out_dir: Path) -> dict[str, int]:
    """Write project-authored SemIf and public JevBench anchors, excluding private tiers."""
    semif = [convert_semif(row) for row in _read_jsonl(semif_path)]
    jevbench = []
    for path in jev_paths:
        for row in _read_jsonl(path):
            item = convert_jevbench(row)
            if item is None:
                continue
            # The public easy/original/hard files carry no "split" field of their own;
            # tag tier from the source filename so per-tier ECE can be reported later.
            item.meta["tier"] = path.stem
            jevbench.append(item)
    return {
        "semif_authored144": _write(out_dir / "semif_authored144.jsonl", semif),
        "jevbench_public": _write(out_dir / "jevbench_public.jsonl", jevbench),
    }


def main() -> None:
    """Export only the pinned public SemIf and JevBench input files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--semif", type=Path, default=ROOT / "refs/semif/benchmarks/data/authored144.jsonl")
    ap.add_argument(
        "--jevbench",
        type=Path,
        nargs="+",
        default=[
            ROOT / "refs/jevbench/datasets/public/original.jsonl",
            ROOT / "refs/jevbench/datasets/public/easy.jsonl",
            ROOT / "refs/jevbench/datasets/public/hard.jsonl",
        ],
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/built/external")
    args = ap.parse_args()
    print(json.dumps(prepare_anchors(args.semif, args.jevbench, args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
