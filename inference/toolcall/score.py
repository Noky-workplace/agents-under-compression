"""Trajectory scoring and deterministic percentile bootstrap intervals."""

import json
import random
from collections import Counter

from .tasks import validate

METRICS = ("schema_valid", "tool_selection", "args_correct", "abstained_correctly", "silent_failure")


def matches(value, matcher):
    if "eq" in matcher:
        return type(value) is type(matcher["eq"]) and value == matcher["eq"]
    if "approx" in matcher:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (abs(value - matcher["approx"]) <= matcher["tol"])
        )
    if "in" in matcher:
        return any(type(value) is type(v) and value == v for v in matcher["in"])
    return matcher.get("any") is True


def score(task, calls, content=""):
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in task["tools"]}
    malformed = not isinstance(calls, list)
    calls = calls if not malformed else [None]
    content = content if isinstance(content, str) else ""
    parsed, valid = [], not malformed
    for call in calls:
        try:
            fn = call["function"]
            if not isinstance(fn["name"], str) or not isinstance(fn["arguments"], str):
                raise TypeError("invalid function fields")
            args = json.loads(fn["arguments"])
            valid = validate(args, schemas[fn["name"]]) and valid
            parsed.append((fn["name"], args))
        except (ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError):
            valid = False
            fn = call.get("function") if isinstance(call, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            parsed.append((name if isinstance(name, str) else None, None))
    expected = task["expected"]
    selection = Counter(n for n, _ in parsed) == Counter(c["name"] for c in expected)

    # Bipartite matching handles repeated tool names without depending on response order.
    def assign(index, remaining):
        if index == len(expected):
            return not remaining
        exp = expected[index]
        for i, (name, args) in enumerate(remaining):
            if (
                name == exp["name"]
                and isinstance(args, dict)
                and set(args) == set(exp["args"])
                and all(matches(args[k], m) for k, m in exp["args"].items())
                and assign(index + 1, remaining[:i] + remaining[i + 1 :])
            ):
                return True
        return False

    if task["category"] == "multi_turn":
        # Dependent calls must follow the scripted trajectory, unlike parallel calls.
        correct = valid and selection and all(
            name == exp["name"]
            and isinstance(args, dict)
            and set(args) == set(exp["args"])
            and all(matches(args[k], m) for k, m in exp["args"].items())
            for (name, args), exp in zip(parsed, expected, strict=False)
        )
    else:
        correct = valid and selection and assign(0, parsed)
    abstained = not calls and not expected and (task["category"] != "clarify" or bool(content.strip()))
    return {
        "failure_reason": (
            "invalid_tool_call" if not valid else
            "wrong_tool_selection" if not selection else
            "incorrect_arguments" if not correct else
            "empty_content" if not calls and not content.strip() else None
        ),
        "schema_valid": valid,
        "tool_selection": selection,
        "args_correct": correct,
        "abstained_correctly": abstained,
        "silent_failure": bool(expected) and valid and selection and not correct,
    }


def bootstrap(values, seed=42, samples=2000):
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return {
        "mean": sum(values) / len(values),
        "low": means[int(samples * 0.025)],
        "high": means[min(samples - 1, int(samples * 0.975))],
        "n": len(values),
    }


def is_timeout(row):
    """Recognize new flags and historical transport timeout errors."""
    if "timed_out" in row:
        return bool(row["timed_out"])
    reason = row.get("failure_reason") or row.get("scores", {}).get("failure_reason") or ""
    return reason.startswith("response_error:") and (
        "TimeoutError" in reason or "timed out" in reason.lower()
    )


def task_means(rows, metric):
    grouped = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.get("task", f"row-{index}"), []).append(
            int(is_timeout(row)) if metric == "timeout_rate" else int(row["scores"][metric])
        )
    return {task: sum(values) / len(values) for task, values in grouped.items()}


def aggregate(rows):
    # Average repetitions within a task, then resample independent tasks.
    groups = {"overall": rows}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    return {
        category: {
            metric: bootstrap(list(task_means(group, metric).values()))
            for metric in (*METRICS, "timeout_rate")
        }
        for category, group in groups.items()
    }


def compare(rows, a, b):
    """Exact two-sided paired sign test on per-task argument accuracy.

    Ties are excluded from the binomial test; each shared task gets one vote.
    """
    import math

    left = task_means([r for r in rows if r["label"] == a], "args_correct")
    right = task_means([r for r in rows if r["label"] == b], "args_correct")
    common = sorted(left.keys() & right.keys())
    wins = sum(left[t] > right[t] for t in common)
    losses = sum(left[t] < right[t] for t in common)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
    return {"tasks": len(common), "wins": wins, "losses": losses,
            "ties": len(common) - n, "p_value": p if common else None}
