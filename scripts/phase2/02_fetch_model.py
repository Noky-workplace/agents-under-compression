#!/usr/bin/env python3
"""Download GGUF files from Hugging Face by repo + filename pattern.

Asks the Hub which files exist first, so a renamed file gives a clear error
instead of a broken download. Resumes partial downloads.

    python3 scripts/phase2/02_fetch_model.py                       # default Q4_K_M
    python3 scripts/phase2/02_fetch_model.py --pattern 'Q8_0|Q6_K|Q5_K_M|Q3_K_M'
    python3 scripts/phase2/02_fetch_model.py --pattern 'bf16|f16' --dest ~/archive/models
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_REPO = "bartowski/Qwen_Qwen3-8B-GGUF"
DEFAULT_PATTERN = r"Q4_K_M"


def list_files(repo: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        info = json.load(resp)
    return [s["rfilename"] for s in info.get("siblings", [])]


def pick(files: list[str], pattern: str) -> list[str]:
    rx = re.compile(rf"[-._]({pattern})\.gguf$", re.IGNORECASE)
    return sorted(f for f in files if rx.search(f))


def download(repo: str, filename: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(filename).name
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "--retry", "5", "-o", str(target), url], check=True)
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--pattern", default=DEFAULT_PATTERN, help="regex for the quant tag")
    p.add_argument("--dest", type=Path, default=Path("~/models"))
    p.add_argument("--list", action="store_true", help="only list matching files")
    args = p.parse_args(argv)

    files = list_files(args.repo)
    wanted = pick(files, args.pattern)
    if not wanted:
        ggufs = [f for f in files if f.endswith(".gguf")]
        print(f"No file in {args.repo} matches {args.pattern!r}. Available:", file=sys.stderr)
        print("\n".join(f"  {f}" for f in ggufs), file=sys.stderr)
        return 1
    for f in wanted:
        if args.list:
            print(f)
            continue
        path = download(args.repo, f, args.dest.expanduser())
        print(f"saved {path} ({path.stat().st_size / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
