import copy
import io
import json
import queue
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from inference.toolcall.client import complete
from inference.toolcall.run_eval import evaluate, run
from inference.toolcall.score import aggregate, bootstrap, score
from inference.toolcall.tasks import load_tasks, validate


def calls(task):
    return [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps({k: v["eq"] for k, v in c["args"].items()}),
            },
        }
        for i, c in enumerate(task["expected"])
    ]


def response(tool_calls=None, content=None):
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls or []}}]
    }


@pytest.fixture
def endpoint():
    state = {"requests": [], "responses": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            state["requests"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(state["responses"].pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        # Sandboxed hosts may forbid loopback binding. Still execute the real
        # http.server handler in a thread, using HTTP bytes over memory streams.
        work = queue.Queue()

        class MemorySocket:
            def __init__(self, payload):
                self.input = io.BytesIO(payload)
                self.output = io.BytesIO()

            def makefile(self, *args):
                return self.input

            def sendall(self, data):
                self.output.write(data)

        def serve():
            while (item := work.get()) is not None:
                sock, ready = item
                Handler(sock, ("127.0.0.1", 0), None)
                ready.set()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        def transport(request, timeout):
            sock = MemorySocket(
                b"POST /v1/chat/completions HTTP/1.0\r\nContent-Length: "
                + str(len(request.data)).encode()
                + b"\r\n\r\n"
                + request.data
            )
            ready = threading.Event()
            work.put((sock, ready))
            assert ready.wait(timeout)
            return io.BytesIO(sock.output.getvalue().split(b"\r\n\r\n", 1)[1])

        try:
            with patch("urllib.request.urlopen", transport):
                yield "http://127.0.0.1:8080", state
        finally:
            work.put(None)
            thread.join()
        return
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join()


def test_seed_suite():
    tasks = load_tasks()
    assert len(tasks) >= 80
    assert {t["category"] for t in tasks} == {"single", "parallel", "multi_turn", "abstain", "clarify"}
    for task in tasks:
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in task["tools"]}
        for call in task["expected"]:
            assert validate({k: v["eq"] for k, v in call["args"].items()}, schemas[call["name"]])
        assert score(task, calls(task), "Please clarify")["args_correct"]


def test_scoring_errors():
    task = load_tasks()[0]
    good = calls(task)
    assert score(task, good)["args_correct"]
    bad = copy.deepcopy(good)
    bad[0]["function"]["name"] = "detect_regime"
    assert not score(task, bad)["tool_selection"]
    bad = copy.deepcopy(good)
    bad[0]["function"]["arguments"] = "{"
    assert not score(task, bad)["schema_valid"]
    bad[0]["function"]["arguments"] = json.dumps({"symbol": "BTCUSDT"})
    assert not score(task, bad)["schema_valid"]
    bad[0]["function"]["arguments"] = json.dumps(
        {"symbol": "ETHUSDT", "window_minutes": 60, "as_of": "2026-11-01T12:00:00Z"}
    )
    assert score(task, bad)["silent_failure"]
    abstain = next(t for t in load_tasks() if t["category"] == "abstain")
    assert not score(abstain, good)["abstained_correctly"]
    clarify = next(t for t in load_tasks() if t["category"] == "clarify")
    assert not score(clarify, [], "")["abstained_correctly"]
    assert score(clarify, [], "Which symbol?")["abstained_correctly"]


def test_perfect_endpoint(endpoint):
    url, state = endpoint
    task = load_tasks()[0]
    state["responses"] = [response(calls(task))]
    result = evaluate(task, url, 7, 2)
    assert result["scores"]["args_correct"]
    assert result["latency_seconds"] >= 0
    assert state["requests"][0]["seed"] == 7
    assert state["requests"][0]["temperature"] == 0
    assert state["requests"][0]["tools"] == task["tools"]


def test_multi_turn(endpoint):
    url, state = endpoint
    task = next(t for t in load_tasks() if t["category"] == "multi_turn")
    first, second = calls(task)
    state["responses"] = [response([first]), response([second]), response(content="Done")]
    assert evaluate(task, url, 0, 2)["scores"]["args_correct"]
    feedback = state["requests"][1]["messages"][-1]
    assert feedback["tool_call_id"] == first["id"]
    assert json.loads(feedback["content"]) == task["tool_results"][0]
    assert json.loads(state["requests"][2]["messages"][-1]["content"]) == task["tool_results"][1]


def test_resume_after_interruption(endpoint, tmp_path):
    url, state = endpoint
    task = load_tasks()[0]
    suite = tmp_path / "tasks.jsonl"
    suite.write_text(json.dumps(task) + "\n")
    output = tmp_path / "results.jsonl"
    state["responses"] = [response(calls(task)), response(calls(task))]
    original = evaluate

    def interrupted(task, base_url, seed, timeout, *options):
        if seed == 1:
            raise RuntimeError("interrupted")
        return original(task, base_url, seed, timeout, *options)

    with patch("inference.toolcall.run_eval.evaluate", interrupted), pytest.raises(RuntimeError):
        run(url, "Q4", 2, output, suite)
    assert len(output.read_text().splitlines()) == 1
    run(url, "Q4", 2, output, suite)
    run(url, "Q4", 2, output, suite)
    assert len(state["requests"]) == 2
    assert [json.loads(line)["seed"] for line in output.read_text().splitlines()] == [0, 1]


def test_retry_connection(endpoint):
    url, state = endpoint
    state["responses"] = [response(content="ok")]
    from urllib.request import urlopen

    attempts = []

    def flaky(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError("connection reset")
        return urlopen(*args, **kwargs)

    with patch("urllib.request.urlopen", flaky):
        raw, _ = complete(url, [], [], 1, 2)
    assert raw["choices"][0]["message"]["content"] == "ok"
    assert len(attempts) == 2


def test_bootstrap():
    assert bootstrap([1, 1, 1]) == {"mean": 1, "low": 1, "high": 1, "n": 3}
    assert bootstrap([0, 1]) == {"mean": 0.5, "low": 0, "high": 1, "n": 2}
    assert bootstrap([0, 1, 1]) == bootstrap([0, 1, 1])
    assert bootstrap([])["n"] == 0
    metrics = aggregate([{"category": "single", "scores": score(load_tasks()[0], [])}])
    assert set(metrics) == {"overall", "single"}


def test_loader_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    task = load_tasks()[0]
    del task["expected"][0]["args"]["as_of"]
    path.write_text(json.dumps(task))
    with pytest.raises(ValueError, match=task["id"]):
        load_tasks(path)
    path.write_text("{")
    with pytest.raises(ValueError, match="line 1"):
        load_tasks(path)


def test_matchers_and_types():
    from inference.toolcall.score import matches

    assert matches(1.01, {"approx": 1, "tol": 0.02})
    assert matches("BTCUSDT", {"in": ["BTCUSDT", "ETHUSDT"]})
    assert matches(None, {"any": True})
    assert not matches(True, {"eq": 1})
    assert not validate(True, {"type": "integer"})
    assert not validate("x", {"type": "string", "enum": ["y"]})


@pytest.mark.parametrize("error", ["wrong_tool", "invalid_json", "missing_arg", "unexpected_call"])
def test_endpoint_failures(endpoint, error):
    url, state = endpoint
    task = load_tasks()[0]
    returned = calls(task)
    if error == "wrong_tool":
        returned[0]["function"]["name"] = "order_book_imbalance"
    elif error == "invalid_json":
        returned[0]["function"]["arguments"] = "{"
    elif error == "missing_arg":
        args = json.loads(returned[0]["function"]["arguments"])
        del args["as_of"]
        returned[0]["function"]["arguments"] = json.dumps(args)
    else:
        task = next(t for t in load_tasks() if t["category"] == "abstain")
    state["responses"] = [response(returned)]
    result = evaluate(task, url, 0, 2)["scores"]
    assert not result["args_correct"]
    if error in {"invalid_json", "missing_arg"}:
        assert not result["schema_valid"]
    if error == "unexpected_call":
        assert not result["abstained_correctly"]


def test_parallel_order_and_repeated_names():
    task = next(
        t
        for t in load_tasks()
        if t["category"] == "parallel" and len({c["name"] for c in t["expected"]}) == 1
    )
    assert score(task, list(reversed(calls(task))))["args_correct"]
    duplicate = calls(task)
    duplicate[1] = duplicate[0]
    assert score(task, duplicate)["silent_failure"]


def test_report(tmp_path, capsys):
    from inference.toolcall.report import report

    task = load_tasks()[0]
    row = {"label": "Q4", "category": "single", "scores": score(task, calls(task))}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n")
    report(path)
    text = capsys.readouterr().out
    assert "| Q4 | overall |" in text
    assert "| Q4 | single |" in text
    assert "1.000 [1.000, 1.000]" in text


def test_resume_torn_write(endpoint, tmp_path):
    url, state = endpoint
    task = load_tasks()[0]
    suite = tmp_path / "tasks.jsonl"
    suite.write_text(json.dumps(task) + "\n")
    out = tmp_path / "results.jsonl"
    state["responses"] = [response(calls(task)), response(calls(task))]
    run(url, "Q4", 1, out, suite)
    with out.open("a") as stream:
        stream.write('{"label":')
    run(url, "Q4", 2, out, suite)
    assert len([json.loads(line) for line in out.read_text().splitlines()]) == 2
    assert len(state["requests"]) == 2


def test_seed_diversity():
    import re
    from collections import Counter

    tasks = load_tasks()
    counts = Counter(t['category'] for t in tasks)
    assert all(counts[c] >= 12 for c in ('single', 'parallel', 'multi_turn', 'abstain', 'clarify'))
    seen = {}
    for task in tasks:
        normalized = json.dumps(task['messages'], sort_keys=True)
        normalized = re.sub(r'\b(?:BTC|ETH|SOL|XRP)(?:USDT)?\b', '<symbol>', normalized)
        normalized = re.sub(r'\d+', '<number>', normalized)
        assert normalized not in seen, (task['id'], seen.get(normalized))
        seen[normalized] = task['id']


@pytest.mark.parametrize('malformed', [None, [], {'function': None}, {'function': []}])
def test_malformed_call_is_scored(malformed):
    assert not score(load_tasks()[0], [malformed])['schema_valid']


def test_multi_turn_order():
    task = next(t for t in load_tasks() if t['category'] == 'multi_turn')
    assert not score(task, list(reversed(calls(task))))['args_correct']


def test_dependent_calls_cannot_be_batched(endpoint):
    url, state = endpoint
    task = next(t for t in load_tasks() if t['category'] == 'multi_turn')
    state['responses'] = [response(calls(task)), response(content='Done')]
    assert not evaluate(task, url, 0, 2)['scores']['args_correct']


@pytest.mark.parametrize('change', ['unknown_argument', 'false_any', 'extra_results', 'empty_expected'])
def test_loader_rejects_invalid_expectations(tmp_path, change):
    task = load_tasks()[0]
    if change == 'unknown_argument':
        task['expected'][0]['args']['extra'] = {'any': True}
    elif change == 'false_any':
        task['expected'][0]['args']['symbol'] = {'any': 1}
    elif change == 'extra_results':
        task['tool_results'] = []
    else:
        task['expected'] = []
    path = tmp_path / 'bad.jsonl'
    path.write_text(json.dumps(task))
    with pytest.raises(ValueError, match=task['id']):
        load_tasks(path)


@pytest.mark.parametrize('name', [[], {}])
def test_review_unhashable_name(name):
    result = score(load_tasks()[0], [{'function': {'name': name, 'arguments': '{}'}}])
    assert not result['schema_valid']
    assert result['failure_reason']


@pytest.mark.parametrize('call', [{'function': None}, None])
def test_review_multiturn_envelope(call):
    task = next(t for t in load_tasks() if t['category'] == 'multi_turn')
    with patch('inference.toolcall.run_eval.complete', return_value=(response([call]), 0)):
        result = evaluate(task, 'unused', 0, 1)
    assert not result['scores']['schema_valid']
    assert result['failure_reason']


@pytest.mark.parametrize('raw', [
    {}, None, [], {'choices': []}, {'choices': [None]},
    response(content=''), response(tool_calls=[{'function': {'name': 'unknown', 'arguments': '{}'}}]),
    {'choices': [{'message': {'tool_calls': {}, 'content': 'text'}}]},
    response(tool_calls=[{'function': {'name': 'get_price_window', 'arguments': {}}}]),
])
def test_invalid_response_persisted_and_continues(tmp_path, raw):
    task = next(t for t in load_tasks() if t['category'] == 'multi_turn')
    suite = tmp_path / 'tasks.jsonl'
    suite.write_text(json.dumps(task) + '\n')
    out = tmp_path / 'out.jsonl'
    good = [response([call]) for call in calls(task)] + [response(content='Done')]
    with patch('inference.toolcall.run_eval.complete', side_effect=[(raw, 0)] + [(r, 0) for r in good]):
        run('unused', 'test', 2, out, suite)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]['failure_reason']
    assert not rows[0]['scores']['args_correct']
    assert rows[1]['scores']['args_correct']


@pytest.mark.parametrize('error', [
    urllib.error.HTTPError('unused', 500, 'server error', {}, None),
    json.JSONDecodeError('truncated', '{', 1),
    __import__('http.client', fromlist=['IncompleteRead']).IncompleteRead(b'{', 20),
])
def test_transport_failure_becomes_row(error):
    with patch('urllib.request.urlopen', side_effect=error):
        result = evaluate(load_tasks()[0], 'http://unused', 0, 1)
    assert result['failure_reason'].startswith('response_error:')
    assert not result['scores']['args_correct']


def test_300_mutated_responses_never_crash():
    import random

    rng = random.Random(73491)
    task = load_tasks()[0]
    values = [None, [], {}, '', 0, True, ['bad'], {'bad': 1}, '{', '[]', 'null']
    for _ in range(300):
        raw = response(calls(task))
        message = raw['choices'][0]['message']
        mutation = rng.randrange(6)
        value = copy.deepcopy(rng.choice(values))
        if mutation == 0:
            raw = value
        elif mutation == 1:
            raw['choices'] = value
        elif mutation == 2:
            message['tool_calls'] = value
        elif mutation == 3:
            message['tool_calls'][0]['function']['name'] = value
        elif mutation == 4:
            message['tool_calls'][0]['function']['arguments'] = value
        else:
            message['content'] = value
        score(task, message.get('tool_calls'), message.get('content'))
        with patch('inference.toolcall.run_eval.complete', return_value=(raw, 0)):
            result = evaluate(task, 'unused', 0, 1)
        assert isinstance(result['scores']['args_correct'], bool)


def test_duration_prompt_requests_measurement():
    task = next(t for t in load_tasks() if t['id'] == 'finance-004')
    assert 'Retrieve BTC realized volatility' in task['messages'][-1]['content']
    assert task['expected'][0]['args']['window_minutes'] == {'eq': 1440}


def test_independent_metrics_survive_persistence(tmp_path):
    task = load_tasks()[0]
    call = calls(task)[0]
    args = json.loads(call['function']['arguments'])
    del args['as_of']
    call['function']['arguments'] = json.dumps(args)
    suite, out = tmp_path / 'tasks.jsonl', tmp_path / 'rows.jsonl'
    suite.write_text(json.dumps(task) + '\n')
    with patch('inference.toolcall.run_eval.complete', return_value=(response([call]), 0)):
        run('unused', 'test', 1, out, suite)
    rows = [json.loads(out.read_text())]
    assert rows[0]['scores']['tool_selection']
    assert not rows[0]['scores']['schema_valid']
    assert aggregate(rows)['overall']['tool_selection']['mean'] == 1


@pytest.mark.parametrize('category', ['single', 'parallel', 'multi_turn'])
@pytest.mark.parametrize('field,value', [('id', None), ('id', []), ('id', ''),
                                         ('type', None), ('type', 'not_function')])
def test_envelopes_checked_for_every_category(category, field, value):
    task = next(t for t in load_tasks() if t['category'] == category)
    batch = calls(task)
    if value is None:
        del batch[0][field]
    else:
        batch[0][field] = value
    with patch('inference.toolcall.run_eval.complete', return_value=(response(batch), 0)):
        result = evaluate(task, 'unused', 0, 1)
    assert result['failure_reason'] == 'invalid_tool_call_envelope'
    assert result['scores']['schema_valid']  # Argument schema is independent.
    assert result['scores']['tool_selection']
    assert not result['scores']['args_correct']


@pytest.mark.parametrize('echo', [True, False])
def test_reasoning_echo_is_exact_and_switchable(echo):
    task = next(t for t in load_tasks() if t['category'] == 'multi_turn')
    first = response([calls(task)[0]])
    first['choices'][0]['message']['reasoning_content'] = 'preserve these tokens exactly'
    original = copy.deepcopy(first)
    seen = []
    replies = iter([first, response([calls(task)[1]]), response(content='Done')])

    def fake(base_url, messages, tools, seed, timeout, temperature):
        seen.append(copy.deepcopy(messages))
        assert temperature == 0.7
        return next(replies), 0

    with patch('inference.toolcall.run_eval.complete', fake):
        result = evaluate(task, 'unused', 0, temperature=0.7, echo_reasoning=echo)
    assert result['scores']['args_correct']
    expected = copy.deepcopy(original['choices'][0]['message'])
    if not echo:
        expected.pop('reasoning_content')
    assert seen[1][-2] == expected
    assert first == original


@pytest.mark.parametrize('error', [TimeoutError('slow'), urllib.error.URLError(TimeoutError('slow'))])
def test_timeouts_are_failures_reported_separately(error):
    from inference.toolcall.score import is_timeout

    with patch('inference.toolcall.run_eval.complete', side_effect=error):
        result = evaluate(load_tasks()[0], 'unused', 0)
    assert is_timeout(result)
    assert not result['scores']['args_correct']
    assert not is_timeout({'failure_reason': 'response_error: ConnectionError: reset'})
    assert is_timeout({'failure_reason': 'response_error: URLError: <urlopen error timed out>'})


def test_bootstrap_clusters_tasks_instead_of_seeds():
    good = score(load_tasks()[0], calls(load_tasks()[0]))
    bad = dict(good, args_correct=False)
    rows = [{"task": 'a', "category": 'single', "scores": good}] * 3
    rows += [{"task": 'b', "category": 'single', "scores": bad}]
    metric = aggregate(rows)['overall']['args_correct']
    assert metric['mean'] == 0.5
    assert metric['n'] == 2
    repeated = rows * 3
    assert aggregate(repeated) == aggregate(rows)


def test_paired_sign_test_common_tasks_and_ties():
    from inference.toolcall.score import compare

    rows = []
    for i in range(7):
        for label in ('Q4', 'Q6'):
            rows.append({"label": label, "task": str(i), "category": 'multi_turn',
                             "scores": {'args_correct': label == 'Q6' or i == 6}})
    rows.append({"label": 'Q4', "task": 'unpaired', "scores": {'args_correct': True}})
    result = compare(rows, 'Q4', 'Q6')
    assert result == {"tasks": 7, "wins": 0, "losses": 6, "ties": 1, "p_value": 0.03125}
    assert compare(rows, 'absent', 'Q6')['p_value'] is None


def test_v1_chains_and_stop_behavior():
    from inference.toolcall.tasks import DEFAULT_TASKS

    tasks = load_tasks(DEFAULT_TASKS.with_name('finance_v1.jsonl'))
    chains = [t for t in tasks if t['category'] == 'multi_turn']
    assert len(chains) >= 40
    assert any(len(t['expected']) == 3 for t in chains)
    stopping = [t for t in chains if len(t['expected']) == 1]
    assert stopping
    for task in chains:
        replies = [(response([call]), 0) for call in calls(task)]
        replies.append((response(content='Done'), 0))
        with patch('inference.toolcall.run_eval.complete', side_effect=replies):
            assert evaluate(task, 'unused', 0)['scores']['args_correct'], task['id']
    task = stopping[-1]
    extra = calls(task)[0]
    with patch('inference.toolcall.run_eval.complete', side_effect=[
        (response([extra]), 0), (response([extra]), 0), (response(content='Done'), 0)
    ]):
        assert not evaluate(task, 'unused', 0)['scores']['args_correct']


def test_report_timeout_exclusion_and_empty_group(tmp_path, capsys):
    from inference.toolcall.report import report

    good = score(load_tasks()[0], calls(load_tasks()[0]))
    rows = [{"label": 'A', "task": 'one', "seed": 0, "category": 'single', "scores": good, "timed_out": True},
            {"label": 'B', "task": 'one', "seed": 0, "category": 'single', "scores": good, "timed_out": False}]
    path = tmp_path / 'results.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report(path, ['A', 'B'])
    text = capsys.readouterr().out
    assert 'Without timed-out requests' in text
    assert 'timeout_rate' in text
    assert '| A | overall | NA' in text
    assert '"tasks": 0' in text


def test_client_sends_temperature_and_default_timeout():
    sent = []

    def transport(request, timeout):
        sent.append((json.loads(request.data), timeout))
        return io.BytesIO(json.dumps(response(content='Done')).encode())

    with patch('urllib.request.urlopen', transport):
        complete('http://unused', [], [], 3, temperature=0.6)
    assert sent[0][0]['temperature'] == 0.6
    assert sent[0][1] == 300
