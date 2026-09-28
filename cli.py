"""Command-line glue for the shared evaluation and data-building modules."""

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

MODEL = "openbmb/MiniCPM5-2B-Base"
CONFIG = "configs/eval_m1.yaml"


def _parser() -> argparse.ArgumentParser:
    """Define Phase 3 commands while keeping domain work in its owning modules."""
    ap = argparse.ArgumentParser(prog="cli.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--config", default=CONFIG)
    ev.add_argument("--split", default="test")
    ev.add_argument("--predictor", choices=("b0", "adapter", "trained"), default="b0")
    ev.add_argument("--model")
    ev.add_argument("--adapter")
    ev.add_argument("--calibration", default=None)
    ev.add_argument("--data")
    ev.add_argument("--out-dir", default=None)
    ev.add_argument("--baseline")
    ev.add_argument("--reencode", action="store_true")
    build = sub.add_parser("build")
    build.add_argument("--config", default="configs/eval_data_v1.yaml")
    build.add_argument("--decode", type=int, default=0)
    fit = sub.add_parser("fit-cal")
    fit.add_argument("--adapter", default="none")
    fit.add_argument("--split", default="cal")
    fit.add_argument("--n-perms", type=int, default=1)
    fit.add_argument("--out", required=True)
    fit.add_argument("--model", default=MODEL)
    fit.add_argument("--config", default=CONFIG)
    fit.add_argument("--data")
    fit.add_argument("--reencode", action="store_true")
    return ap


def cmd_eval(args: argparse.Namespace) -> dict:
    """Load configuration and delegate predictions and metrics to evaluate.run_eval."""
    import evaluate

    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    cfg = raw.get("eval", raw)
    data_path = args.data or cfg["data"][args.split]
    out_dir = args.out_dir or f"reports/{args.split}_{args.predictor}"
    calibration_path = args.calibration
    if calibration_path is None:
        calibration_path = raw.get("predictors", {}).get(args.predictor, {}).get("calibration")
    calibration = None
    if calibration_path and calibration_path != "none":
        from calibrate import Calibration

        calibration = Calibration(**json.loads(Path(calibration_path).read_text()))
    model = args.model or MODEL
    predictor_name = "adapter" if args.predictor == "trained" else args.predictor
    if predictor_name == "adapter":
        adapter = args.adapter or raw.get("predictors", {}).get(args.predictor, {}).get("adapter")
        if not adapter:
            raise ValueError("--adapter is required for the adapter predictor")
        predictor = evaluate.AdapterPredictor(model, adapter, calibration=calibration, reencode=args.reencode)
    else:
        predictor = evaluate.B0Predictor(model, calibration=calibration, reencode=args.reencode)
    items = evaluate.load_items(data_path)
    return evaluate.run_eval(
        predictor,
        items,
        cfg=cfg,
        baseline=args.baseline,
        out_dir=Path(out_dir),
    )


def cmd_build(args: argparse.Namespace) -> None:
    """Delegate public eval-set construction to its data adapter."""
    from sources.public import build_eval_v1

    if args.decode:
        raise NotImplementedError("decoded-row inspection is not available in the Phase 3 builder")
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    counts = build_eval_v1(cfg)
    print(json.dumps(counts, sort_keys=True))


def _model_revision(model: str) -> str:
    """Resolve a cached model to its immutable Hub snapshot hash without network access."""
    path = Path(model)
    if path.is_dir():
        revision = path.name
    else:
        from huggingface_hub import snapshot_download

        revision = Path(snapshot_download(repo_id=model, local_files_only=True)).name
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError(f"model {model!r} does not resolve to a 40-character snapshot hash")
    return revision


def cmd_fit_cal(args: argparse.Namespace) -> None:
    """Fit per-qtype temperatures on cal only and save T=1 audit records."""
    import evaluate
    from calibrate import T_MAX, T_MIN, Calibration, fit_temperature
    from fmt import FORMAT_VERSION

    if args.split != "cal":
        raise ValueError("fit-cal may only use --split cal; calibration on test is forbidden")
    if args.n_perms < 1:
        raise ValueError("--n-perms must be positive")
    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    cfg = raw.get("eval", raw)
    data_path = Path(args.data or cfg["data"]["cal"])
    model = args.model or MODEL
    max_context = cfg.get("max_context", 8192)
    predictor = (
        evaluate.B0Predictor(model, reencode=args.reencode, max_context=max_context)
        if args.adapter == "none"
        else evaluate.AdapterPredictor(model, args.adapter, reencode=args.reencode, max_context=max_context)
    )
    grouped: dict[str, tuple[list[np.ndarray], list[int]]] = {}
    audit_rows = []
    for item in evaluate.load_items(data_path):
        if item.gold is None:
            continue
        key = f"{item.question.type}:{args.n_perms}"
        pooled = np.asarray(
            predictor.predict_logprobs(item.state, [item.question], n_perms=args.n_perms)[item.question.id],
            dtype=np.float64,
        )
        xs, ys = grouped.setdefault(key, ([], []))
        xs.append(pooled)
        ys.append(int(item.gold))
        probs = np.exp(pooled - pooled.max())
        probs /= probs.sum()
        audit_rows.append(
            {
                "question_id": item.question.id,
                "qtype": item.question.type,
                "n_perms": args.n_perms,
                "gold": int(item.gold),
                "probs": probs.tolist(),
                "correct_gold": int(np.argmax(probs)) == int(item.gold),
            }
        )
    if not grouped:
        raise ValueError(f"no gold-judged items found in calibration split: {data_path}")
    temperatures = {}
    for key, (pooled, targets) in grouped.items():
        temperature = fit_temperature(pooled, targets)
        if temperature <= T_MIN * 1.01 or temperature >= T_MAX * 0.99:
            print(f"fit-cal: temperature for {key} is at search bound: {temperature}", file=sys.stderr)
            raise SystemExit(1)
        temperatures[key] = temperature
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()[:12]
    cal = Calibration(
        model_revision=_model_revision(model),
        format_version=FORMAT_VERSION,
        temperature=temperatures,
        default_perms={key.split(":", 1)[0]: args.n_perms for key in temperatures},
        fitted_on=f"eval_v1/cal@{digest}",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataclasses.asdict(cal), indent=2) + "\n")
    (out.parent / "cal_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows)
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    {"eval": cmd_eval, "build": cmd_build, "fit-cal": cmd_fit_cal}[args.cmd](args)


if __name__ == "__main__":
    main()
