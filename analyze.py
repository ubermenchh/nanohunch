# Confidence intervals for trained minus B0, from the logits that run.py saved. No model needed.
# Usage: uv run python analyze.py [runs/cases.pkl]
import pickle
import sys

from eval import bootstrap_diff, item_stats

d = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/cases.pkl", "rb"))
sets = dict(d["b0"]) | {"all": [c for cases in d["b0"].values() for c in cases]}
sets_tr = dict(d["trained"]) | {"all": [c for cases in d["trained"].values() for c in cases]}

for setting, Ta, Tb in (("raw (T=1)", 1.0, 1.0), ("with each model's T", d["T_b0"], d["T_trained"])):
    print(f"\ntrained minus B0, {setting}, 95% interval from 2000 resamples of the questions")
    print(f"{'set':6} {'n':>4}  " + "  ".join(f"{k:>24}" for k in ("hit", "nll", "ece", "flip")))
    for name in sets:
        a, b = item_stats(sets[name], Ta), item_stats(sets_tr[name], Tb)
        cells = []
        for key in ("hit", "nll", "ece", "flip"):
            diff, lo, hi = bootstrap_diff(a, b, key)
            sig = "*" if lo > 0 or hi < 0 else " "  # * = the interval excludes 0
            cells.append(f"{diff:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        print(f"{name:6} {len(sets[name]):>4}  " + "  ".join(f"{c:>24}" for c in cells))
print(
    "\nhit = accuracy, nll = loss (lower is better), ece = calibration error (lower is better), flip = order sensitivity"
)
