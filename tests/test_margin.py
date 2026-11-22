"""Pre-registered margin statistics and fail-closed measurement checks."""

import json

import pytest

from inference.toolcall import margin


def test_auc_ties_and_direction():
    assert margin.auc([3, 2, 1, 0], [True, True, False, False]) == 1
    assert margin.auc([1, 1, 1, 1], [True, True, False, False]) == .5
    assert margin.auc([1, 0, 1, 0], [True, True, False, False]) == .5


def test_permutation_fixed_seed():
    result = margin.permutation([4, 3, 2, 1], [True, True, False, False])
    assert result == margin.permutation([4, 3, 2, 1], [True, True, False, False])
    assert result['auc'] == 1
    assert .15 < result['p_value'] < .18
    assert result['decision'] == 'not informative'


def test_both_probability_formats():
    chat = {'choices': [{'logprobs': {'content': [
        {'token': 'a', 'bytes': [97], 'logprob': -.2, 'top_logprobs': [
            {'token': 'a', 'logprob': -.2}, {'token': 'b', 'logprob': -1.2}]}]}}]}
    native = {'completion_probabilities': [
        {'content': 'a', 'probs': [{'tok_str': 'a', 'prob': .8}, {'tok_str': 'b', 'prob': .2}]}]}
    assert margin.positions(chat)[0]['bytes'] == [97]
    assert margin.token_margins(margin.positions(chat)) == pytest.approx([1])
    assert margin.token_margins(margin.positions(native)) == pytest.approx([1.38629436112])


def test_reject_missing_candidates_and_nonfinite():
    for pos in ([{'token': 'a', 'logprob': -.2, 'top_logprobs': []}],
                [{'token': 'a', 'logprob': float('nan'), 'top_logprobs': [
                    {'token': 'b', 'logprob': -1}]}]):
        with pytest.raises(ValueError):
            margin.token_margins(pos)


def test_chat_tool_text_and_special_tokens_must_be_covered():
    response = {'choices': [{'message': {'content': '', 'tool_calls': [
        {'function': {'name': 'lookup', 'arguments': '{}'}}]},
        'logprobs': {'content': [{'token': 'hello', 'logprob': -1, 'top_logprobs': [
            {'token': 'hello', 'logprob': -1}, {'token': 'x', 'logprob': -2}]}]}}],
        'usage': {'completion_tokens': 2}}
    with pytest.raises(ValueError):
        margin.validate_coverage(response, margin.positions(response), raw=False)


def test_mismatch_ignores_ids_but_preserves_turns():
    def call(value, ident):
        return {'id': ident, 'function': {'name': 'x', 'arguments': json.dumps({'v': value})}}
    baseline = {'t': [[call(1, 'old')]], 'u': [[]]}
    assert margin.mismatch_share({'t': [[call(1, 'new')]], 'u': [[]]}, baseline) == 0
    with pytest.raises(ValueError, match='5%'):
        margin.mismatch_share({'t': [[call(2, 'new')]], 'u': [[]]}, baseline)
    with pytest.raises(ValueError, match='support'):
        margin.mismatch_share({'t': []}, baseline)


def test_raw_parsing_uses_project_call_envelope():
    msg = margin.raw_message('<tool_call>\n{"name":"x","arguments":{"v":1}}\n</tool_call>')
    assert json.loads(msg['tool_calls'][0]['function']['arguments']) == {'v': 1}
    assert margin.raw_message('Please clarify')['content'] == 'Please clarify'
    with pytest.raises(ValueError):
        margin.raw_message('<tool_call>broken')


def test_analysis_primary_population_and_secondary_tables():
    rows, baseline = [], []
    for i in range(20):
        task = f't{i}'
        # Q3 failures have the lower margins, so negative margin has AUC 1.
        gap = .1 if i < 8 else 2
        rows.append({'task': task, 'turn': 0, 'calls': [], 'tokens': [
            {'token': 'a', 'logprob': -.1, 'top_logprobs': [
                {'token': 'a', 'logprob': -.1}, {'token': 'b', 'logprob': -.1 - gap}]}]})
        for quant in ('Q8_0', 'Q3_K_M', 'Q4_K_M'):
            baseline.append({'task': task, 'label': quant + '-nothink-wait', 'category': 'single',
                             'responses': [{'choices': [{'message': {'tool_calls': []}}]}],
                             'scores': {'args_correct': quant != 'Q3_K_M' or i >= 8}})
    result = margin.analyze(rows, baseline)
    assert result['primary']['auc'] == 1
    assert result['primary']['decision'] == 'supported'
    assert result['primary']['positives'] == 8
    assert result['secondary']['q4']['decision'] == 'not informative'
    assert result['secondary']['lowest_quarter']['share'] == pytest.approx(5 / 8)
    assert result['secondary']['by_category']['single'] == result['primary']
    with pytest.raises(ValueError, match='Duplicate or missing'):
        margin.analyze(rows + rows[:1], baseline)


def test_complete_native_coverage_and_top_ten_cap():
    response = {'content': 'x', 'tokens_predicted': 2, 'completion_probabilities': [
        {'content': 'x', 'probs': [{'tok_str': 'x', 'prob': .5}, {'tok_str': 'y', 'prob': .4}]},
        {'content': '<|im_end|>', 'probs': [
            {'tok_str': '<|im_end|>', 'prob': .5}, {'tok_str': 'z', 'prob': .4}]}]}
    tokens = margin.positions(response)
    margin.validate_coverage(response, tokens, raw=True)
    response['tokens_predicted'] = 3
    with pytest.raises(ValueError, match='positions'):
        margin.validate_coverage(response, tokens, raw=True)
    item = {'token': 'a', 'logprob': -.1, 'top_logprobs': [
        {'token': str(i), 'logprob': -i - 1} for i in range(15)]}
    assert len(margin.positions({'choices': [{'logprobs': {'content': [item]}}]})[0]['top_logprobs']) == 10


def test_measure_fallback_and_partial_only(tmp_path, monkeypatch):
    task = {'id': 't', 'category': 'single', 'messages': [], 'tools': [], 'expected': []}
    baseline = [{'task': 't', 'label': 'Q8_0-nothink-wait',
                 'responses': [{'choices': [{'message': {'tool_calls': []}}]}]}]
    monkeypatch.setattr(margin, 'load_tasks', lambda _: [task])
    monkeypatch.setattr(margin, 'load', lambda _: baseline)
    seen = []

    def generate(base, task, messages, mode):
        seen.append(mode)
        if mode != 'raw':
            raise ValueError('No tool-token probabilities')
        # Preflight tool call is not part of the measured trajectory.
        calls = [{'function': {'name': 'x', 'arguments': '{}'}}] if len(seen) == 3 else []
        return {'content': 'done', 'tool_calls': calls}, [
            {'token': 'a', 'logprob': -.1, 'top_logprobs': [{'token': 'b', 'logprob': -1}]}]

    monkeypatch.setattr(margin, 'generate', generate)
    output = tmp_path / 'out.jsonl'
    partial = margin.measure('unused', output)
    assert seen == ['chat', 'native-chat', 'raw', 'raw']
    assert partial.exists()
    assert not output.exists()
    assert json.loads(output.with_suffix('.validation.json').read_text())['mismatch_share'] == 0
    with pytest.raises(ValueError, match='One-run'):
        margin.measure('unused', output)


def test_measure_abort_never_promotes(tmp_path, monkeypatch):
    task = {'id': 't', 'category': 'single', 'messages': [], 'tools': [], 'expected': []}
    baseline = [{'task': 't', 'label': 'Q8_0-nothink-wait',
                 'responses': [{'choices': [{'message': {'tool_calls': []}}]}]}]
    monkeypatch.setattr(margin, 'load_tasks', lambda _: [task])
    monkeypatch.setattr(margin, 'load', lambda _: baseline)
    calls = 0

    def generate(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise TimeoutError('telemetry supervisor killed server')
        return {'tool_calls': [{'function': {'name': 'x', 'arguments': '{}'}}]}, []

    monkeypatch.setattr(margin, 'generate', generate)
    output = tmp_path / 'out.jsonl'
    with pytest.raises(TimeoutError):
        margin.measure('unused', output)
    assert not output.exists()
    assert output.with_suffix('.partial.jsonl').exists()


@pytest.mark.parametrize('reading', ['85', 'unavailable'])
def test_gpu_supervisor_aborts_partial_run(tmp_path, reading):
    """Real shell supervisor with fake hardware: never promote interrupted output."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / 'scripts/phase2/gpu_margin.sh'
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    source = tmp_path / 'source'
    (source / 'tools/server').mkdir(parents=True)
    (source / 'tools/server/README.md').write_text(
        '/v1/chat/completions /apply-template /completion n_probs\n')
    temp = tmp_path / 'temperature'
    temp.write_text('50')

    def executable(name, body):
        path = bindir / name
        path.write_text(body)
        path.chmod(0o755)

    executable('pgrep', '#!/bin/sh\nexit 1\n')
    executable('curl', '#!/bin/sh\nexit 0\n')
    executable('llama-server', '#!/bin/sh\nif [ "$1" = --help ]; then\n'
               'echo "--reasoning-budget --jinja --host --port"; exit 0; fi\nexec /bin/sleep 30\n')
    executable('nvidia-smi', '#!/bin/sh\ncat "$FAKE_TEMP"\n')
    executable('sleep', '#!/bin/sh\nexec /bin/sleep 0.02\n')
    executable('uv', f'#!{sys.executable}\nimport os, time\nfrom pathlib import Path\n'
               'Path("research/results/phase2/margin-q8.partial.jsonl").write_text("partial")\n'
               f'Path(os.environ["FAKE_TEMP"]).write_text({reading!r})\ntime.sleep(30)\n')
    result = subprocess.run(['bash', str(script)], check=False, cwd=tmp_path,
                            env={**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
                                 'REPO': str(tmp_path), 'LLAMA_SOURCE': str(source),
                                 'LLAMA_BIN_DIR': str(bindir), 'FAKE_TEMP': str(temp)},
                            capture_output=True, text=True, timeout=15)
    out = tmp_path / 'research/results/phase2'
    assert result.returncode != 0
    assert (out / 'margin-q8.partial.jsonl').exists()
    assert not (out / 'margin-q8.jsonl').exists()
    assert not (out / 'margin-analysis.json').exists()
    assert 'abort' in (out / 'margin-status.md').read_text()


def test_chat_coverage_rejects_missing_argument_tokens():
    text = '<tool_call>{"name":"x","arguments":{}}</tool_call><|im_end|>'
    response = {'usage': {'completion_tokens': 1}, 'choices': [{'message': {
        'tool_calls': [{'function': {'name': 'x', 'arguments': '{"missing":1}'}}]}}]}
    tokens = [{'token': text, 'logprob': -.1, 'top_logprobs': [{'token': 'z', 'logprob': -2}]}]
    with pytest.raises(ValueError, match='tool-call'):
        margin.validate_coverage(response, tokens)


def test_mismatch_exactly_five_percent_is_allowed():
    baseline = {str(i): [[]] for i in range(20)}
    current = dict(baseline)
    current['0'] = [[{'function': {'name': 'x', 'arguments': '{}'}}]]
    assert margin.mismatch_share(current, baseline) == .05


def test_mismatch_distinguishes_boolean_and_number_arguments():
    def turns(value):
        return [[{'function': {'name': 'x', 'arguments': json.dumps({'v': value})}}]]
    with pytest.raises(ValueError, match='5%'):
        margin.mismatch_share({'t': turns(True)}, {'t': turns(1)})


def test_report_keeps_primary_and_secondary_separate():
    stats = margin.permutation([1] * 16, [True] * 8 + [False] * 8)
    assert stats['decision'] == 'not supported'
    assert stats['p_value'] == 1
    report = {'primary': stats, 'mismatch_share': 0, 'secondary': {
        'q4': stats, 'first_position': stats, 'mean_margin': stats,
        'by_category': {'single': stats},
        'lowest_quarter': {'tasks': 4, 'cutoff': 1, 'captured_failures': 2, 'share': .25,
                           'ties': 'fractional inclusion'},
        'q8_confidence': {**stats, 'mean_min_margin_success': 1, 'mean_min_margin_failure': 1}}}
    text = margin.render_report(report)
    assert 'Decision: **not supported**' in text
    assert '| Analysis |' in text
    assert '| single |' in text
    assert 'fractional inclusion' in text


def test_fixed_decision_rule_with_no_negative_class():
    # Only the preregistered fewer-than-eight-positives rule says not informative.
    result = margin.permutation([1] * 8, [True] * 8)
    assert result['auc'] is None
    assert result['decision'] == 'not supported'


@pytest.mark.parametrize('probability_text', [
    'different reply<|im_end|>', 'Let me check.<|im_end|>', '',
])
def test_raw_generate_rejects_unrelated_or_missing_probabilities(monkeypatch, probability_text):
    content = 'Let me check.<tool_call>{"name":"actual","arguments":{}}</tool_call>'
    response = {'content': content, 'tokens_predicted': int(bool(probability_text)),
                'completion_probabilities': ([{'content': probability_text, 'probs': [
                    {'tok_str': probability_text, 'prob': .8}, {'tok_str': 'z', 'prob': .2}]}]
                    if probability_text else [])}
    monkeypatch.setattr(margin, 'request', lambda base, endpoint, payload:
                        {'prompt': 'rendered'} if endpoint == '/apply-template' else response)
    with pytest.raises(ValueError, match='probabilities|Probabilities'):
        margin.generate('unused', {'tools': []}, [], 'raw')


def test_chat_rejects_unrelated_free_text():
    response = {'usage': {'completion_tokens': 1}, 'choices': [{'message': {
        'content': 'actual reply', 'tool_calls': []}}]}
    with pytest.raises(ValueError, match='content'):
        margin.validate_coverage(response, [{'token': 'different reply<|im_end|>'}])


@pytest.mark.parametrize('raw', [True, False])
def test_mixed_text_and_call_coverage_with_split_utf8(raw):
    text = '查詢 <tool_call>{"name":"lookup","arguments":{"v":1}}</tool_call>'
    encoded = (text + '<|im_end|>').encode()
    tokens = [{'token': 'unusable byte display', 'bytes': list(encoded[:1])},
              {'token': 'unusable byte display', 'bytes': list(encoded[1:])}]
    response = ({'tokens_predicted': 2, 'content': text} if raw else {
        'usage': {'completion_tokens': 2}, 'choices': [{'message': margin.raw_message(text)}]})
    margin.validate_coverage(response, tokens, raw=raw)


def test_measure_does_not_write_mismatched_turn(tmp_path, monkeypatch):
    task = {'id': 't', 'category': 'single', 'messages': [], 'tools': [], 'expected': []}
    baseline = [{'task': 't', 'label': 'Q8_0-nothink-wait',
                 'responses': [{'choices': [{'message': {'tool_calls': []}}]}]}]
    monkeypatch.setattr(margin, 'load_tasks', lambda _: [task])
    monkeypatch.setattr(margin, 'load', lambda _: baseline)
    completions = 0
    content = '<tool_call>{"name":"actual","arguments":{}}</tool_call>'

    def request(base, endpoint, payload):
        nonlocal completions
        if endpoint == '/v1/chat/completions':
            raise ValueError('Chat probabilities unavailable')
        if endpoint == '/apply-template':
            return {'prompt': 'rendered'}
        completions += 1
        text = (content if completions == 1 else 'unrelated') + '<|im_end|>'
        return {'content': content, 'tokens_predicted': 1, 'completion_probabilities': [
            {'content': text, 'probs': [
                {'tok_str': text, 'prob': .8}, {'tok_str': 'z', 'prob': .2}]}]}

    monkeypatch.setattr(margin, 'request', request)
    output = tmp_path / 'out.jsonl'
    with pytest.raises(ValueError, match='Raw probabilities do not match'):
        margin.measure('unused', output)
    assert output.with_suffix('.partial.jsonl').read_text() == ''
    assert not output.exists()
    assert not output.with_suffix('.validation.json').exists()


def _pos(token, logprob, alt, alt_logprob, id_=None, raw=None):
    pos = {'token': token, 'logprob': logprob, 'bytes': list((token if raw is None else raw).encode()),
           'top_logprobs': [{'token': token, 'logprob': logprob}, {'token': alt, 'logprob': alt_logprob}]}
    if id_ is not None:
        pos['id'] = id_
    return pos


def test_installed_server_format_chat_forced_think_and_empty_end_of_turn():
    # Shape recorded by the 18 Nov ASUS probe: forced </think> and <|im_end|> with empty text.
    call = '<tool_call>\n{"name": "x", "arguments": {"a": 1}}\n</tool_call>'
    items = [_pos('<think>', 0.0, '<tool_call>', -21.0, 151667),
             _pos('</think>', -24.9, '\n', -1e-5, 151668),
             _pos('\n\n', -1e-3, '\n', -7.0, 271),
             _pos(call, -.01, 'y', -5.0),
             _pos('', -2e-7, '</think>', -15.0, 151645)]
    response = {'usage': {'completion_tokens': 5}, 'choices': [{
        'logprobs': {'content': items},
        'message': {'content': '', 'tool_calls': margin.raw_message(call)['tool_calls']}}]}
    tokens = margin.positions(response)
    assert margin.mark_forced_reasoning(tokens) == 2
    margin.validate_coverage(response, tokens)
    margins = margin.token_margins(tokens)
    assert len(margins) == 3 and min(margins) > 0


def test_empty_text_needs_end_of_turn_id():
    call = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    tokens = [_pos(call, -.01, 'y', -5.0), _pos('', -1e-7, 'z', -9.0)]
    response = {'tokens_predicted': 2, 'content': call}
    with pytest.raises(ValueError, match='end-of-turn'):
        margin.validate_coverage(response, tokens, raw=True)
    tokens[1]['id'] = 151645
    margin.validate_coverage(response, tokens, raw=True)


def test_unclosed_forced_reasoning_is_rejected():
    with pytest.raises(ValueError, match='not closed'):
        margin.mark_forced_reasoning([_pos('<think>', 0.0, 'a', -3.0)])


def test_measure_records_failing_turn_without_probabilities(tmp_path, monkeypatch):
    task = {'id': 't', 'category': 'single', 'messages': [], 'tools': [], 'expected': []}
    baseline = [{'task': 't', 'label': 'Q8_0-nothink-wait',
                 'responses': [{'choices': [{'message': {'tool_calls': []}}]}]}]
    monkeypatch.setattr(margin, 'load_tasks', lambda _: [task])
    monkeypatch.setattr(margin, 'load', lambda _: baseline)
    calls = []

    def generate(base, task, messages, mode):
        calls.append(mode)
        if len(calls) == 1:
            return {'content': '', 'tool_calls': [{'function': {'name': 'x', 'arguments': '{}'}}]}, []
        raise ValueError('Chat probabilities do not match response content')

    monkeypatch.setattr(margin, 'generate', generate)
    output = tmp_path / 'out.jsonl'
    with pytest.raises(ValueError):
        margin.measure('unused', output)
    error = json.loads(output.with_suffix('.error.json').read_text())
    assert error == {'task': 't', 'turn': 0, 'category': 'single',
                     'error': 'ValueError: Chat probabilities do not match response content'}


def test_chat_parallel_calls_with_newline_between_blocks():
    # finance-037 (first parallel task) stopped the 17 Nov run: '\n' between calls vs content ''.
    a = '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>'
    b = '<tool_call>\n{"name": "b", "arguments": {"y": 2}}\n</tool_call>'
    items = [_pos(a, -.01, 'z', -5.0), _pos('\n', -.01, 'z', -6.0), _pos(b, -.01, 'z', -5.0),
             _pos('', -1e-7, 'z', -9.0, 151645)]
    message = {'content': '', 'tool_calls': margin.raw_message(a + '\n' + b)['tool_calls']}
    response = {'usage': {'completion_tokens': 4},
                'choices': [{'logprobs': {'content': items}, 'message': message}]}
    margin.validate_coverage(response, margin.positions(response))
    message['content'] = 'unrelated'
    with pytest.raises(ValueError, match='content'):
        margin.validate_coverage(response, margin.positions(response))
