# Generate rule-conditioned decision questions with decider's generator (Apache-2.0, pinned commit, stdlib only; read before use)
# and write them as data_cache/rules.jsonl in the same shape as decider's teacher data, so data.load_decider reads them.
#   uv run python make_rules.py [--n 20000] [--seed 0]
import importlib.util
import json
import sys
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path

URL = "https://raw.githubusercontent.com/Mapika/decider/91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc/decider/data/rules.py"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20000
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
src = Path("data_cache/decider_rules.py")
src.parent.mkdir(exist_ok=True)
if not src.exists():
    urllib.request.urlretrieve(URL, src)


@dataclass
class Q:
    text: str
    options: list
    gold: int


@dataclass
class Example:
    context: str
    qs: list
    task: str


# the two names rules.py imports from the rest of decider; render_state only writes a dict as JSON for states under 8 list items
for name in ("decider", "decider.data", "decider.data.core", "decider.systemone"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["decider.data.core"].Example, sys.modules["decider.data.core"].Q = Example, Q
sys.modules["decider.systemone"].render_state = lambda s: s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
spec = importlib.util.spec_from_file_location("decider_rules", src)
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)

# data.LETTERS has 8 labels, so questions with more options (some forms and pointers) are dropped
kept = [ex for ex in rules.build(N * 2, SEED) if len(ex.qs[0].options) <= 8][:N]
with open("data_cache/rules.jsonl", "w") as f:
    for ex in kept:
        q = ex.qs[0]
        if q.options == ["no", "yes"]:  # a yes/no question
            kind, crit, answer = "noul", None, "yes" if q.gold == 1 else "no"
        else:  # a choice: ids a, b, c ...
            ids = [chr(97 + i) for i in range(len(q.options))]
            kind, crit, answer = "choice", dict(zip(ids, q.options, strict=True)), ids[q.gold]
        question = dict(type=kind, instructions=q.text, criteria=crit, answer=answer)
        f.write(json.dumps({"state": ex.context, "questions": [question]}, ensure_ascii=False) + "\n")
print(f"wrote {len(kept)} questions to data_cache/rules.jsonl")
