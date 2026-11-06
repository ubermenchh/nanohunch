# Confidence intervals from the logits that run.py saved. No model needed.
# One file:  uv run python analyze.py runs/cases_ce.pkl           trained minus B0
# Two files: uv run python analyze.py runs/cases_ce.pkl runs/cases_brier0.5.pkl   second trained minus first trained
import pickle
import sys

from eval import bootstrap_diff, item_stats


def load(path):
    # Pickle can run code when loaded. These files come only from our own run.py, so load nothing else here.
    with open(path, "rb") as f:
        return pickle.load(f)


def with_all(by_set):
    return dict(by_set) | {"all": [c for cases in by_set.values() for c in cases]}


paths = sys.argv[1:] or ["runs/cases_ce.pkl"]
d = load(paths[0])
if len(paths) == 2:
    e = load(paths[1])
    title, a_cases, b_cases, a_T, b_T = (
        "second minus first, both trained",
        d["trained"],
        e["trained"],
        d["T_trained"],
        e["T_trained"],
    )
else:
    title, a_cases, b_cases, a_T, b_T = "trained minus B0", d["b0"], d["trained"], d["T_b0"], d["T_trained"]
A, B = with_all(a_cases), with_all(b_cases)

for setting, Ta, Tb in (("raw (T=1)", 1.0, 1.0), ("with each model's T", a_T, b_T)):
    print(f"\n{title}, {setting}, 95% interval from 2000 resamples of the questions")
    print(f"{'set':6} {'n':>4}  " + "  ".join(f"{k:>24}" for k in ("hit", "nll", "ece", "flip")))
    for name in A:
        a, b = item_stats(A[name], Ta), item_stats(B[name], Tb)
        cells = []
        for key in ("hit", "nll", "ece", "flip"):
            diff, lo, hi = bootstrap_diff(a, b, key)
            sig = "*" if lo > 0 or hi < 0 else " "  # * = the interval excludes 0
            cells.append(f"{diff:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        print(f"{name:6} {len(A[name]):>4}  " + "  ".join(f"{c:>24}" for c in cells))
print(
    "\nhit = accuracy, nll = loss (lower is better), ece = calibration error (lower is better), flip = order sensitivity"
)
