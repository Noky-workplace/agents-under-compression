"""Turn benchmark JSONL into a markdown report (and charts if matplotlib exists).

python -m inference.bench.report research/results/phase2/bench.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def latest(rows: list[dict]) -> list[dict]:
    """Keep the most recent run of each (model, ngl)."""
    seen: dict[tuple[str, int], dict] = {}
    for r in rows:
        seen[(r["model"], r["ngl"])] = r
    return sorted(seen.values(), key=lambda r: (r["model"], r["ngl"]))


def overhead_fit(rows: list[dict]) -> dict | None:
    """Fit the fixed VRAM overhead and report how well the model then predicts."""
    ok = [r for r in rows if "error" not in r and r["ngl"] > 0]
    if not ok:
        return None
    derived = [r["predicted_weights_mib"] + r["predicted_kv_mib"] for r in ok]
    overhead = sum(r["used_mib"] - d for r, d in zip(ok, derived, strict=True)) / len(ok)
    errors = [abs(r["used_mib"] - (d + overhead)) for r, d in zip(ok, derived, strict=True)]
    return {
        "overhead_mib": round(overhead, 1),
        "mae_mib": round(sum(errors) / len(errors), 1),
        "max_err_mib": round(max(errors), 1),
        "n": len(ok),
    }


def weight_fit(rows: list[dict]) -> dict[str, float]:
    """OLS used_mib - KV = fixed + scale * weights on successful records.

    Use training records only. CPU runs are included because used_mib records
    GPU allocations even at ngl=0. A constant-weight dataset cannot identify
    the slope and is rejected rather than silently substituting a default.
    """
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise ValueError("need successful records to fit")
    xs = [r["predicted_weights_mib"] for r in ok]
    ys = [r["used_mib"] - r["predicted_kv_mib"] for r in ok]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    variance = sum((x - mx) ** 2 for x in xs)
    if not variance:
        raise ValueError("need distinct weight sizes to fit")
    scale = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / variance
    return {"overhead_mib": my - scale * mx, "weight_scale": scale}


def model_errors(rows: list[dict], overhead_mib: float, weight_scale: float) -> dict:
    """Before/after absolute errors by offload regime; no fitting on these rows."""
    groups: dict[str, list[tuple[float, float]]] = {k: [] for k in ("zero", "partial", "full")}
    for r in rows:
        if "error" in r:
            continue
        group = "zero" if r["ngl"] == 0 else "full" if r["ngl"] > r["n_layer"] else "partial"
        predicted = overhead_mib + weight_scale * r["predicted_weights_mib"] + r["predicted_kv_mib"]
        groups[group].append((abs(r["used_mib"] - r["predicted_mib"]), abs(r["used_mib"] - predicted)))
    return {
        group: {
            "n": len(errors),
            "before_mae": sum(e[0] for e in errors) / len(errors),
            "before_max": max(e[0] for e in errors),
            "after_mae": sum(e[1] for e in errors) / len(errors),
            "after_max": max(e[1] for e in errors),
        }
        for group, errors in groups.items()
        if errors
    }


def to_markdown(rows: list[dict]) -> str:
    head = "| model | file MiB | ngl | pp t/s | tg t/s | VRAM MiB | predicted | peak °C | W | note |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            "| {model} | {file_mib:.0f} | {ngl} | {pp} | {tg} | {used} | {pred:.0f} | {temp:.0f} | {pw:.0f} | {note} |".format(
                model=r["model"],
                file_mib=r["file_mib"],
                ngl=r["ngl"],
                pp=f"{r['pp_ts']:.0f}" if "pp_ts" in r else "-",
                tg=f"{r['tg_ts']:.1f}" if "tg_ts" in r else "-",
                used=f"{r.get('used_mib', 0):.0f}",
                pred=r.get("predicted_mib", 0),
                temp=r.get("peak_temp_c", 0),
                pw=r.get("mean_power_w", 0),
                note=r.get("error", ""),
            )
        )
    fit = overhead_fit(rows)
    if fit:
        lines += [
            "",
            (
                f"Fixed-only diagnostic (raw weight scale 1): fitted overhead **{fit['overhead_mib']} MiB**, "
                f"mean abs error {fit['mae_mib']} MiB, worst {fit['max_err_mib']} MiB "
                f"over {fit['n']} GPU runs."
            ),
        ]
    try:
        scaled = weight_fit(rows)
    except ValueError:
        scaled = None  # A single weight size cannot identify both parameters.
    if scaled:
        lines += [
            "",
            (
                f"Two-parameter in-sample fit: fixed {scaled['overhead_mib']:.1f} MiB, "
                f"weight scale {scaled['weight_scale']:.6f}; KV coefficient held at 1. "
                "The table preserves predictions recorded at benchmark time."
            ),
        ]
    return "\n".join(lines) + "\n"


def charts(rows: list[dict], out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    ok = [r for r in rows if "tg_ts" in r]
    if not ok:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, ylabel, name in (
        ("tg_ts", "generation speed (tokens/s)", "speed_vs_ngl.png"),
        ("used_mib", "VRAM used (MiB)", "vram_vs_ngl.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in sorted({r["model"] for r in ok}):
            pts = [r for r in ok if r["model"] == model]
            ax.plot([r["ngl"] for r in pts], [r[key] for r in pts], marker="o", label=model)
        ax.set_xlabel("layers on GPU (ngl)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, help="markdown file (default: next to results)")
    args = p.parse_args(argv)
    rows = latest(load(args.results))
    md = to_markdown(rows)
    out = args.out or args.results.with_suffix(".md")
    out.write_text(md)
    made = charts(rows, out.parent / "charts")
    print(md)
    for path in made:
        print(f"chart: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
