"""Fetch the pinned reference repos in configs/refs.yaml into refs/<name>/ as source archives.

Reading only (AGENTS.md Rule 3). Archives instead of clones: no nested git metadata, and the pin is
exact. The full commit sha GitHub resolved is written to refs/<name>/.ref_sha.
"""

import io
import pathlib
import shutil
import sys
import tarfile

import httpx
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fetch(name: str, repo: str, sha: str, client: httpx.Client) -> str:
    url = f"https://github.com/{repo}/archive/{sha}.tar.gz"
    resp = client.get(url)
    resp.raise_for_status()
    dest = ROOT / "refs" / name
    shutil.rmtree(dest, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        top = tar.getmembers()[0].name.split("/")[0]  # "<repo>-<full sha>"
        tar.extractall(ROOT / "refs", filter="data")
    (ROOT / "refs" / top).rename(dest)
    full_sha = top.rsplit("-", 1)[1]
    (dest / ".ref_sha").write_text(full_sha + "\n")
    return full_sha


def main() -> int:
    refs = yaml.safe_load((ROOT / "configs" / "refs.yaml").read_text())["refs"]
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for r in refs:
            full = fetch(r["name"], r["repo"], r["sha"], client)
            ok = full.startswith(r["sha"])
            print(f"{r['name']:14s} {r['repo']:30s} {full[:12]} {'ok' if ok else 'SHA MISMATCH'}")
            if not ok:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
