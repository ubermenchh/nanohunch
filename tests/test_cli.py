"""Phase 3 CLI glue: config selection and flags must reach the shared evaluator unchanged."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


def test_eval_command_uses_selected_split_and_cli_overrides(tmp_path, monkeypatch):
    """CLI overrides select the requested data and predictor without loading an MLX model."""
    import cli
    import evaluate

    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {"eval": {"data": {"cal": "cal.jsonl", "test": "test.jsonl"}, "bins": 15, "flip_suite": ["reverse"]}}
        )
    )
    seen = {}

    class Predictor:
        def __init__(self, model, *, calibration=None, **kwargs):
            seen["predictor"] = (model, calibration, kwargs)

    monkeypatch.setattr(evaluate, "load_items", lambda path: seen.setdefault("data", path), raising=False)
    monkeypatch.setattr(evaluate, "B0Predictor", Predictor, raising=False)
    monkeypatch.setattr(evaluate, "AdapterPredictor", Predictor, raising=False)
    monkeypatch.setattr(
        evaluate,
        "run_eval",
        lambda predictor, items, *, cfg, baseline, out_dir: seen.update(run=(predictor, items, cfg, baseline, out_dir)),
        raising=False,
    )

    cli.main(
        [
            "eval",
            "--config",
            str(config),
            "--split",
            "test",
            "--predictor",
            "b0",
            "--model",
            "toy-model",
            "--calibration",
            "none",
            "--baseline",
            "previous",
            "--out-dir",
            str(tmp_path / "reports" / "current"),
        ]
    )

    assert str(seen["data"]) == "test.jsonl"
    assert seen["predictor"] == ("toy-model", None, {"reencode": False})
    _, _, cfg, baseline, out_dir = seen["run"]
    assert cfg["flip_suite"] == ["reverse"]
    assert baseline == "previous"
    assert out_dir == tmp_path / "reports" / "current"


def test_fit_cal_writes_calibration_and_t1_audit_rows(tmp_path, monkeypatch):
    """fit-cal groups by qtype, records provenance and keeps an auditable T=1 sidecar."""
    import calibrate
    import cli
    import evaluate
    from fmt import FORMAT_VERSION, Question

    data = tmp_path / "cal.jsonl"
    data.write_text("fixture data\n")
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump({"eval": {"data": {"cal": str(data)}, "max_context": 1234}}))
    items = [
        SimpleNamespace(state="s1", gold=0, question=Question("q1", "choice", "Pick", ("a", "b"))),
        SimpleNamespace(state="s2", gold=1, question=Question("q2", "choice", "Pick", ("a", "b"))),
        SimpleNamespace(state="s3", gold=0, question=Question("q3", "noul", "True?", ("yes", "no"))),
    ]
    predictor_kwargs = []

    class Predictor:
        def __init__(self, model, **kwargs):
            self.model = model
            predictor_kwargs.append(kwargs)

        def predict_logprobs(self, state, questions, *, n_perms):
            return {questions[0].id: np.log([0.8, 0.2])}

    monkeypatch.setattr(evaluate, "load_items", lambda path: items, raising=False)
    monkeypatch.setattr(evaluate, "B0Predictor", Predictor, raising=False)
    monkeypatch.setattr(evaluate, "AdapterPredictor", Predictor, raising=False)
    seen = []
    monkeypatch.setattr(calibrate, "fit_temperature", lambda xs, ys: seen.append((xs, ys)) or 1.25)
    monkeypatch.setattr(cli, "_model_revision", lambda model: "a" * 40, raising=False)
    out = tmp_path / "calibration.json"

    cli.main(
        [
            "fit-cal",
            "--config",
            str(config),
            "--data",
            str(data),
            "--n-perms",
            "2",
            "--model",
            "toy-model",
            "--reencode",
            "--out",
            str(out),
        ]
    )

    saved = json.loads(out.read_text())
    assert saved["format_version"] == FORMAT_VERSION
    assert saved["model_revision"] == "a" * 40
    assert saved["temperature"] == {"choice:2": 1.25, "noul:2": 1.25}
    assert saved["fitted_on"] == f"eval_v1/cal@{hashlib.sha256(data.read_bytes()).hexdigest()[:12]}"
    audit = [json.loads(line) for line in (tmp_path / "cal_items.jsonl").read_text().splitlines()]
    assert len(audit) == 3 and audit[0]["probs"] == pytest.approx([0.8, 0.2])
    assert len(seen) == 2
    assert predictor_kwargs == [{"reencode": True, "max_context": 1234}]


def test_fit_cal_rejects_temperature_on_search_bound(tmp_path, monkeypatch, capsys):
    """A bound-valued T indicates the optimizer did not find an interior solution."""
    import calibrate
    import cli
    import evaluate
    from fmt import Question

    data = tmp_path / "cal.jsonl"
    data.write_text("cal\n")
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump({"eval": {"data": {"cal": str(data)}}}))
    item = SimpleNamespace(state="s", gold=0, question=Question("q", "noul", "True?", ("yes", "no")))

    class Predictor:
        def __init__(self, model, **kwargs):
            pass

        def predict_logprobs(self, state, questions, *, n_perms):
            return {"q": np.log([0.8, 0.2])}

    monkeypatch.setattr(evaluate, "load_items", lambda path: [item], raising=False)
    monkeypatch.setattr(evaluate, "B0Predictor", Predictor, raising=False)
    monkeypatch.setattr(calibrate, "fit_temperature", lambda xs, ys: 0.05)
    monkeypatch.setattr(cli, "_model_revision", lambda model: "b" * 40, raising=False)

    with pytest.raises(SystemExit, match="1"):
        cli.main(["fit-cal", "--config", str(config), "--out", str(tmp_path / "cal.json")])
    assert "noul:1" in capsys.readouterr().err


def test_fit_cal_refuses_test_split(tmp_path):
    """Temperature fitting on held-out test data would contaminate the final measurement."""
    import cli

    with pytest.raises(ValueError, match="only use --split cal"):
        cli.main(
            [
                "fit-cal",
                "--split",
                "test",
                "--config",
                str(tmp_path / "unused.yaml"),
                "--out",
                str(tmp_path / "cal.json"),
            ]
        )


def test_model_revision_requires_immutable_snapshot_hash(tmp_path):
    """Calibration provenance must not record a moving model alias as a revision."""
    import cli

    valid = tmp_path / ("c" * 40)
    valid.mkdir()
    assert cli._model_revision(str(valid)) == "c" * 40
    mutable = tmp_path / "latest"
    mutable.mkdir()
    with pytest.raises(ValueError, match="40-character snapshot hash"):
        cli._model_revision(str(mutable))
