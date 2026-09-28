"""Model dir with a passthrough chat template (plan Phase 0 step 4).

Stock mlx_lm.lora wraps every prompt/completion row in the tokenizer's chat template
(mlx_lm/tuner/datasets.py:107-127). This dir symlinks the snapshot's files and replaces the
template with plain concatenation, so rows stay raw nanohunch-fmt-v1 text.

usage: uv run python -m bench.make_raw_model REPO OUT_DIR bos|nobos
(P0-1: MiniCPM5 adds BOS -> bos; Qwen3-4B-Base does not -> nobos)
"""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

TEMPLATE_FILES = ("tokenizer_config.json", "chat_template.jinja", "chat_template.json")


def main() -> None:
    repo, out, bos = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "bos"
    src = Path(snapshot_download(repo, local_files_only=True))
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in TEMPLATE_FILES:
            (out / f.name).unlink(missing_ok=True)
            (out / f.name).symlink_to(f.resolve())
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    cfg["chat_template"] = ("{{ bos_token }}" if bos else "") + "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{out}: passthrough template, bos={bos}")


if __name__ == "__main__":
    main()
