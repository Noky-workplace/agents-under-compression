"""Append-only, resumable evaluation runner."""

import argparse
import http.client
import json
import math
import time
import urllib.error
from pathlib import Path

from .client import complete
from .score import METRICS, score
from .tasks import DEFAULT_TASKS, load_tasks


def evaluate(task, base_url, seed, timeout=300, temperature=0, echo_reasoning=True,
             one_call=False, reject_placeholders=False, provenance=False):
    if one_call or reject_placeholders or provenance:
        from .enforcement import evaluate_enforced

        return evaluate_enforced(task, base_url, seed, timeout, temperature, echo_reasoning,
                                 one_call, reject_placeholders, provenance, complete)
    messages = list(task["messages"])
    calls, responses, latency, texts = [], [], 0.0, []
    dependent_batch = False
    failure_reason = None
    timed_out = False
    limit = len(task["expected"]) + 2 if task["category"] == "multi_turn" else 1
    for _ in range(limit):
        started = time.perf_counter()
        try:
            response, elapsed = complete(base_url, messages, task["tools"], seed, timeout, temperature)
        except (OSError, ValueError, http.client.HTTPException, RecursionError) as exc:
            timed_out = isinstance(exc, TimeoutError) or (
                isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)
            )
            latency += time.perf_counter() - started
            failure_reason = f"response_error: {type(exc).__name__}: {exc}"
            break
        responses.append(response)
        latency += elapsed
        try:
            choices = response["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError("choices must be a nonempty list")
            message = choices[0]["message"]
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("content must be text or null")
            turn_calls = message.get("tool_calls", [])
            if turn_calls is None:
                turn_calls = []
            if not isinstance(turn_calls, list):
                raise TypeError("tool_calls must be a list")
            if not turn_calls and not (content or "").strip():
                raise ValueError("empty content")
            texts.append(content or "")
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            failure_reason = f"invalid_response: {exc}"
            break
        offset = len(calls)
        calls.extend(turn_calls)
        if not score(task, turn_calls)["schema_valid"]:
            failure_reason = "invalid_tool_call"
            break
        if any(
            not isinstance(call.get("id"), str) or not call["id"]
            or call.get("type") != "function" for call in turn_calls
        ):
            failure_reason = "invalid_tool_call_envelope"
            break
        if task["category"] != "multi_turn" or not turn_calls:
            break
        dependent_batch |= len(turn_calls) > 1
        echoed = dict(message)
        if not echo_reasoning:
            echoed.pop("reasoning_content", None)
        messages.append(echoed)
        results = task["tool_results"]
        if len(calls) > len(results):
            break
        for i, call in enumerate(turn_calls):
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(results[offset + i])}
            )
    scores = score(task, calls, "\n".join(texts))
    if dependent_batch:
        scores["args_correct"] = False
        scores["silent_failure"] = scores["schema_valid"] and scores["tool_selection"]
    failure_reason = failure_reason or scores["failure_reason"]
    if dependent_batch:
        failure_reason = failure_reason or "dependent_calls_batched"
    if failure_reason and failure_reason not in {
        "wrong_tool_selection", "incorrect_arguments", "dependent_calls_batched",
        "invalid_tool_call", "invalid_tool_call_envelope"
    }:
        scores.update({metric: False for metric in METRICS})
    if failure_reason == "invalid_tool_call_envelope":
        scores["args_correct"] = False
        scores["silent_failure"] = False
    scores["failure_reason"] = failure_reason
    return {
        "timed_out": timed_out,
        "failure_reason": failure_reason,
        "responses": responses,
        "latency_seconds": latency,
        "scores": scores,
    }


def run(base_url, label, seeds, out, tasks=DEFAULT_TASKS, timeout=300, temperature=0,
        echo_reasoning=True, one_call=False, reject_placeholders=False, provenance=False):
    tasks = load_tasks(tasks)
    out = Path(out)
    done = set()
    if out.exists():
        # A process may die during its final write. Only an incomplete final
        # record can be discarded; corruption in completed records is an error.
        data = out.read_bytes()
        offset = 0
        for line in data.splitlines(keepends=True):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if offset + len(line) != len(data) or line.endswith(b"\n"):
                    raise
                with out.open("r+b") as stream:
                    stream.truncate(offset)
                break
            done.add((row["label"], row["task"], row["seed"]))
            offset += len(line)
        else:
            if data and not data.endswith(b"\n"):
                with out.open("ab") as stream:
                    stream.write(b"\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as stream:
        for task in tasks:
            for seed in range(seeds):
                if (label, task["id"], seed) in done:
                    continue
                result = evaluate(task, base_url, seed, timeout, temperature, echo_reasoning,
                                  one_call, reject_placeholders, provenance)
                stream.write(
                    json.dumps(
                        dict(label=label, task=task["id"], seed=seed, category=task["category"],
                             temperature=temperature, echo_reasoning=echo_reasoning, **result)
                    )
                    + "\n"
                )
                stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seeds", type=int, default=1, help="Seeds only affect sampling above temperature 0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--no-echo-reasoning", dest="echo_reasoning", action="store_false",
                        help="Omit reasoning_content from assistant history for comparison")
    parser.add_argument("--one-call", action="store_true")
    parser.add_argument("--reject-placeholders", action="store_true")
    parser.add_argument("--provenance", action="store_true")
    args = parser.parse_args()
    if (args.seeds < 1 or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.temperature) or args.temperature < 0):
        parser.error("seeds/timeout must be positive and temperature finite and nonnegative")
    run(**vars(args))


if __name__ == "__main__":
    main()
