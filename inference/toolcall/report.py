"""Task-bootstrap metrics and paired label comparisons, including/excluding timeouts."""

import argparse
import json
from pathlib import Path

from .score import METRICS, aggregate, compare, is_timeout


def report(path, comparison=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    metrics_names = (*METRICS, "timeout_rate")
    for exclude in (False, True):
        selected = [r for r in rows if not (exclude and is_timeout(r))]
        print("\n" + ("Without timed-out requests" if exclude else "All requests"))
        labels = sorted({r["label"] for r in rows})
        print("| Label | Category | " + " | ".join(metrics_names) + " |")
        print("| --- | --- | " + " | ".join("---" for _ in metrics_names) + " |")
        for label in labels:
            summaries = aggregate([r for r in selected if r["label"] == label])
            categories = ["overall", *sorted({r["category"] for r in rows if r["label"] == label})]
            for category in categories:
                metrics = summaries.get(category, aggregate([])["overall"])
                cells = [
                    "NA" if v["mean"] is None else
                    f"{v['mean']:.3f} [{v['low']:.3f}, {v['high']:.3f}] (tasks={v['n']})"
                    for v in metrics.values()
                ]
                print(f"| {label} | {category} | " + " | ".join(cells) + " |")
        if comparison:
            # Conditional comparison uses the same non-timeout task/seed support on both sides.
            paired = selected
            if exclude:
                bad = {(r["task"], r.get("seed", 0)) for r in rows
                       if r["label"] in comparison and is_timeout(r)}
                paired = [r for r in selected if (r["task"], r.get("seed", 0)) not in bad]
            print("Paired args_correct sign test " + str(tuple(comparison)) + ": "
                  + json.dumps(compare(paired, *comparison), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    report(args.file, args.compare)
