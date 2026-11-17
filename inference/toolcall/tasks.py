"""Task loading and the deliberately small JSON schema subset used by the suite."""

import json
import math
from pathlib import Path

DEFAULT_TASKS = Path(__file__).parent / "tasks" / "finance_v1.jsonl"


def validate(value, schema):
    kind = schema.get("type")
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        ),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }
    if kind not in checks or not checks[kind]():
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if kind == "object":
        props = schema.get("properties", {})
        if not set(schema.get("required", [])).issubset(value):
            return False
        if schema.get("additionalProperties") is False and set(value) - props.keys():
            return False
        return all(validate(v, props[k]) for k, v in value.items() if k in props)
    if kind == "array":
        return all(validate(v, schema["items"]) for v in value)
    return True


def check_schema(schema):
    assert isinstance(schema, dict)
    assert schema["type"] in {"object", "array", "string", "integer", "number", "boolean", "null"}
    assert not set(schema) - {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "description",
        "title",
        "default",
        "example",
        "examples",
    }, "unsupported schema keyword"
    if schema["type"] == "object":
        assert isinstance(schema.get("properties", {}), dict)
        assert isinstance(schema.get("required", []), list)
        assert set(schema.get("required", [])) <= schema.get("properties", {}).keys()
        assert isinstance(schema.get("additionalProperties", True), bool)
        for child in schema.get("properties", {}).values():
            check_schema(child)
    if schema["type"] == "array":
        check_schema(schema["items"])
    if "enum" in schema:
        assert isinstance(schema["enum"], list) and schema["enum"]
        assert all(validate(value, schema) for value in schema["enum"])


def load_tasks(path=DEFAULT_TASKS):
    tasks, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        task_id = f"line {line_no}"
        try:
            task = json.loads(line)
            assert isinstance(task, dict)
            task_id = task.get("id", task_id)
            assert "id" in task
            assert isinstance(task_id, str) and task_id and task_id not in seen, "invalid/duplicate id"
            assert task["category"] in {"single", "parallel", "multi_turn", "abstain", "clarify"}
            assert isinstance(task["messages"], list) and task["messages"]
            assert all(
                isinstance(m, dict)
                and m.get("role") in {"system", "user", "assistant", "tool"}
                and isinstance(m.get("content"), str)
                for m in task["messages"]
            )
            assert isinstance(task["tools"], list) and task["tools"]
            schemas = {}
            for tool in task["tools"]:
                assert tool["type"] == "function"
                fn = tool["function"]
                assert isinstance(fn["name"], str) and fn["name"]
                assert fn["name"] not in schemas
                check_schema(fn["parameters"])
                schemas[fn["name"]] = fn["parameters"]
                assert fn["parameters"]["type"] == "object"
            assert isinstance(task["expected"], list)
            for call in task["expected"]:
                schema = schemas[call["name"]]
                assert isinstance(call["args"], dict)
                assert set(call["args"]) <= schema.get("properties", {}).keys()
                samples = {}
                for key, matcher in call["args"].items():
                    assert isinstance(matcher, dict)
                    if set(matcher) == {"eq"}:
                        samples[key] = matcher["eq"]
                    elif set(matcher) == {"approx", "tol"}:
                        assert validate(matcher["tol"], {"type": "number"}) and matcher["tol"] >= 0
                        assert validate(matcher["approx"], {"type": "number"})
                        assert schema["properties"][key]["type"] in {"number", "integer"}
                        samples[key] = matcher["approx"]
                    elif set(matcher) == {"in"}:
                        assert isinstance(matcher["in"], list) and matcher["in"]
                        assert all(validate(v, schema["properties"][key]) for v in matcher["in"])
                        samples[key] = matcher["in"][0]
                    else:
                        assert set(matcher) == {"any"} and matcher["any"] is True
                        assert key in schema["properties"]
                assert set(schema.get("required", [])) <= call["args"].keys()
                assert all(validate(v, schema["properties"][k]) for k, v in samples.items())
            if task["category"] in {"abstain", "clarify"}:
                assert not task["expected"]
            if task["category"] not in {"abstain", "clarify"}:
                assert task["expected"], "data task needs expected calls"
            if task["category"] != "multi_turn":
                assert "tool_results" not in task, "tool_results is multi-turn only"
            if task["category"] == "multi_turn":
                assert isinstance(task["tool_results"], list)
                assert len(task["tool_results"]) == len(task["expected"])
            seen.add(task_id)
            tasks.append(task)
        except (ValueError, KeyError, TypeError, AttributeError, AssertionError) as exc:
            raise ValueError(f"Task {task_id} (line {line_no}): malformed task: {exc}") from exc
    return tasks
