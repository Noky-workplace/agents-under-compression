"""Benchmark GGUF models with llama-bench while recording VRAM, heat and power.

One JSON line per (model, ngl) run is appended to the results file, so a
crashed or interrupted sweep keeps everything measured so far.

    python -m inference.bench.run_bench ~/models/*.gguf --ngl auto
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from . import vram
from .gguf import read_header
from .gpu import GpuSampler, read_gpu, wait_for_cooldown

DEFAULT_BIN_DIR = Path(os.environ.get("LLAMA_BIN_DIR", "~/src/llama.cpp/build/bin")).expanduser()
DEFAULT_OUT = Path("research/results/phase2/bench.jsonl")
OOM_MARKERS = ("out of memory", "failed to allocate", "CUDA error", "unable to allocate")


def parse_bench_json(stdout: str) -> dict[str, float]:
    """Extract prompt-processing (pp) and text-generation (tg) speeds.

    llama-bench prints a JSON array, sometimes surrounded by log lines.
    """
    start, end = stdout.find("["), stdout.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in llama-bench output")
    out: dict[str, float] = {}
    for row in json.loads(stdout[start : end + 1]):
        n_prompt, n_gen = int(row.get("n_prompt", 0)), int(row.get("n_gen", 0))
        if n_prompt > 0 and n_gen == 0:
            kind = "pp"
        elif n_gen > 0 and n_prompt == 0:
            kind = "tg"
        else:
            continue
        out[f"{kind}_ts"] = float(row["avg_ts"])
        out[f"{kind}_ts_std"] = float(row.get("stddev_ts", 0.0))
        out[f"{kind}_n"] = float(n_prompt or n_gen)
    if not out:
        raise ValueError("llama-bench output had no pp/tg rows")
    return out


def auto_ngl(n_layer: int) -> list[int]:
    """CPU only, quarter steps, all blocks, and all blocks + output head."""
    steps = {0, n_layer // 4, n_layer // 2, (3 * n_layer) // 4, n_layer, n_layer + 1}
    return sorted(steps)


def bench_cmd(bin_dir: Path, model: Path, ngl: int, n_prompt: int, n_gen: int, reps: int) -> list[str]:
    return [
        str(bin_dir / "llama-bench"),
        "-m", str(model),
        "-ngl", str(ngl),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(reps),
        "-o", "json",
    ]  # fmt: skip


def run_one(
    model: Path,
    ngl: int,
    *,
    bin_dir: Path = DEFAULT_BIN_DIR,
    n_prompt: int = 512,
    n_gen: int = 128,
    reps: int = 3,
    timeout_s: float = 1800.0,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    gpu_reader=read_gpu,
    cooldown=wait_for_cooldown,
) -> dict:
    header = read_header(model)
    predicted = vram.estimate(header, ngl, max(n_prompt, n_gen))
    record: dict = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model": model.name,
        "file_mib": round(header.file_size / vram.MIB, 1),
        "n_layer": header.n_layer,
        "ngl": ngl,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "reps": reps,
        "predicted_mib": round(predicted.total_mib, 1),
        "predicted_weights_mib": round(predicted.weights_mib, 1),
        "predicted_kv_mib": round(predicted.kv_mib, 1),
    }

    record["start_temp_c"] = cooldown(reader=gpu_reader)
    record["baseline_mib"] = gpu_reader().mem_used_mib

    proc = popen(
        bench_cmd(bin_dir, model, ngl, n_prompt, n_gen, reps),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with GpuSampler(reader=gpu_reader, on_over_temp=proc.terminate) as sampler:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            record["error"] = "timeout"

    stats = sampler.stats
    record |= {
        "exit_code": proc.returncode,
        "peak_mib": stats.peak_mem_mib,
        "used_mib": round(stats.peak_mem_mib - record["baseline_mib"], 1),
        "peak_temp_c": stats.peak_temp_c,
        "mean_power_w": round(stats.mean_power_w, 1),
        "over_temp": stats.over_temp,
    }
    if stats.over_temp:
        record["error"] = "aborted: GPU over temperature limit"
    elif proc.returncode != 0 and "error" not in record:
        low = (stderr or "").lower()
        oom = any(m.lower() in low for m in OOM_MARKERS)
        record["error"] = "oom" if oom else (stderr or "").strip()[-400:]
    if "error" not in record:
        try:
            record |= parse_bench_json(stdout)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            record["error"] = f"parse: {e}"
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument("--ngl", default="auto", help="'auto' or comma list, e.g. 0,18,37")
    p.add_argument("--n-prompt", type=int, default=512)
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    for model in args.models:
        model = model.expanduser()
        n_layer = read_header(model).n_layer
        ngls = auto_ngl(n_layer) if args.ngl == "auto" else [int(x) for x in args.ngl.split(",")]
        for ngl in ngls:
            rec = run_one(
                model,
                ngl,
                bin_dir=args.bin_dir,
                n_prompt=args.n_prompt,
                n_gen=args.n_gen,
                reps=args.reps,
            )
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            status = rec.get("error") or f"pp {rec.get('pp_ts', 0):.0f} t/s, tg {rec.get('tg_ts', 0):.1f} t/s"
            print(f"{model.name} ngl={ngl}: {status} | {rec['used_mib']} MiB", file=sys.stderr)
            failures += "error" in rec
            if rec.get("over_temp"):
                print("GPU too hot - stopping the sweep.", file=sys.stderr)
                return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
