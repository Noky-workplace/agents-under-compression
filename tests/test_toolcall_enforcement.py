"""Scripted model trajectories exercise enforcement without an inference host."""
import copy
import json
from unittest.mock import patch

import pytest

from inference.toolcall.run_eval import evaluate


def call(x, ident='c'):
    return {'id': ident, 'type': 'function',
            'function': {'name': 'f', 'arguments': json.dumps({'x': x})}}


def reply(*calls):
    return {'choices': [{'message': {'role': 'assistant', 'content': 'Done',
                                     'tool_calls': list(calls)}}]}


def task(values=(1, 2), category='multi_turn', user='Use 1 then the result.'):
    return {'category': category, 'messages': [{'role': 'user', 'content': user}],
            'tools': [{'function': {'name': 'f', 'parameters': {'type': 'object',
                       'properties': {'x': {'type': 'string' if isinstance(values[0], str) else 'number'}}, 'required': ['x']}}}],
            'expected': [{'name': 'f', 'args': {'x': {'eq': x}}} for x in values],
            'tool_results': [{'x': x + 1} if isinstance(x, int) else {'x': x} for x in values]}


def scripted(t, replies, **options):
    seen = []
    iterator = iter(replies)
    def model(url, messages, *args):
        seen.append(copy.deepcopy(messages))
        return next(iterator), 0.25
    with patch('inference.toolcall.run_eval.complete', model):
        result = evaluate(t, 'unused', 0, **options)
    return result, seen


def test_one_call_retry_preserves_strict_failure():
    result, seen = scripted(task(), [reply(call(1), call(2, 'd')), reply(call(2)), reply()],
                            one_call=True)
    assert not result['scores']['args_correct']
    assert result['success_after_enforcement']
    assert result['enforcement'] == [{'turn': 1, 'rule': 'one_call', 'argument': None,
                                     'call_index': 1}]
    assert 'not executed: one tool call per reply' in seen[1][-1]['content']
    assert len(result['executed_calls']) == 2


@pytest.mark.parametrize('value', ['<result>', 'RESULT_OF_f', 'previous_x', 'placeholder',
                                   'to_be_set', 'unknown', 'TBD'])
def test_placeholder_recovery(value):
    result, seen = scripted(task(('2',)),
                            [reply(call(value)), reply(call('2')), reply()], reject_placeholders=True)
    assert result['success_after_enforcement']
    assert result['enforcement'][0]['argument'] == 'x'
    assert 'earlier result or the user' in seen[1][-1]['content']


@pytest.mark.parametrize('value', [9, 14])
def test_provenance_guessed_and_legitimate_free_choice_false_rejection(value):
    result, _ = scripted(task((14,), 'single', 'Choose a number.'),
                         [reply(call(value)), reply()], provenance=True)
    assert not result['success_after_enforcement']
    assert result['executed_calls'] == []
    assert result['enforcement'][0]['rule'] == 'provenance'


def test_provenance_prior_results_and_numeric_equality():
    result, _ = scripted(task(), [reply(call(1)), reply(call(2)), reply()], provenance=True)
    assert result['success_after_enforcement']
    assert not result['enforcement']
    result, _ = scripted(task((14.0,), 'single', 'Use 14.'),
                         [reply(call(14.0)), reply()], provenance=True)
    assert not result['enforcement']


def test_same_reply_result_is_not_provenance():
    result, _ = scripted(task(), [reply(call(1), call(2)), reply()], provenance=True)
    assert result['enforcement'][0]['call_index'] == 1


def test_turn_cap():
    result, _ = scripted(task(('2',)), [reply(call('<x>'))] * 8, reject_placeholders=True)
    assert result['turn_cap_hit']
    assert result['model_turns'] == 8
    assert not result['success_after_enforcement']


def test_options_off_exact_reproduction():
    replies = [reply(call(1)), reply(call(2)), reply()]
    old, _ = scripted(task(), replies)
    explicit, _ = scripted(task(), replies, one_call=False, reject_placeholders=False, provenance=False)
    assert explicit == old
    assert 'enforcement' not in old


def test_enforcement_analysis_fixture():
    from inference.toolcall.compare_runs import build_enforcement_report

    t = task((14,), 'single', 'Choose a number.')
    result, _ = scripted(t, [reply(call(14)), reply()], provenance=True)
    baseline = {'label': 'Q4_K_M-nothink-wait', 'task': 'a', 'category': 'single',
                    'scores': {'args_correct': True}, 'latency_seconds': 1, 'responses': [reply(call(14))]}
    row = dict(result, label='Q4_K_M-nothink-prov', task='a', category='single')
    report = build_enforcement_report([baseline], [row], {'a': t})
    assert '| provenance | single | 1 | 1 |' in report
    assert 'lowers single scores' in report
    assert '| single | executed | 1 | 0 | 1.0' in report
    assert 'Median seconds' in report


@pytest.mark.parametrize('annotation', ['enum', 'default', 'example'])
def test_schema_annotations_are_provenance(annotation, tmp_path):
    from inference.toolcall.tasks import load_tasks

    t = task((14,), 'single', 'Choose a number.')
    t['id'] = 'annotation'
    del t['tool_results']
    t['tools'][0]['type'] = 'function'
    schema = t['tools'][0]['function']['parameters']['properties']['x']
    schema[annotation] = [14] if annotation == 'enum' else 14
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps(t) + '\n')
    loaded = load_tasks(path)[0]
    result, _ = scripted(loaded, [reply(call(14)), reply()], provenance=True)
    assert not result['enforcement']
    assert result['success_after_enforcement']


def test_case_sensitive_strings_and_nested_arguments():
    t = task(('BTC',), 'single', 'Use btc.')
    result, _ = scripted(t, [reply(call('BTC')), reply()], provenance=True)
    assert result['enforcement'][0]['argument'] == 'x'
    t['tools'][0]['function']['parameters']['properties']['x'] = {
        'type': 'array', 'items': {'type': 'string'}}
    t['expected'][0]['args']['x'] = {'eq': ['BTC']}
    result, _ = scripted(t, [reply(call(['<value>'])), reply()], reject_placeholders=True)
    assert result['enforcement'][0]['argument'] == 'x[0]'


def test_all_real_tasks_ideal_model_under_enforcement():
    from inference.toolcall.tasks import DEFAULT_TASKS, load_tasks

    for t in load_tasks(DEFAULT_TASKS.with_name('finance_v1_wait.jsonl')):
        batch = [{'id': f'c{i}', 'type': 'function', 'function': {
            'name': c['name'], 'arguments': json.dumps({k: m['eq'] for k, m in c['args'].items()})}}
                 for i, c in enumerate(t['expected'])]
        replies = ([reply(c) for c in batch] + [reply()] if t['category'] == 'multi_turn'
                   else [reply(*batch)])
        result, _ = scripted(t, replies,
                             one_call=t['category'] != 'parallel', reject_placeholders=True)
        assert result['success_after_enforcement'], t['id']
        assert not result['enforcement']


def test_provenance_incremental_comparison():
    from inference.toolcall.compare_runs import build_enforcement_report

    t = task((14,), 'single', 'Choose a number.')
    enforced, _ = scripted(t, [reply(call(14)), reply()], one_call=True)
    prov, _ = scripted(t, [reply(call(14)), reply()], provenance=True)
    base = dict(enforced, label='Q4-nothink-wait', task='a', category='single')
    rows = [dict(enforced, label='Q4-nothink-enforce', task='a', category='single'),
            dict(prov, label='Q4-nothink-prov', task='a', category='single')]
    report = build_enforcement_report([base], rows, {'a': t})
    assert 'Adding provenance lowers single scores (executed)' in report


def test_one_call_does_not_validate_or_execute_extra_call():
    invalid = call('<placeholder>', 'extra')
    result, seen = scripted(task(), [reply(call(1), invalid), reply(call(2)), reply()], one_call=True)
    assert result['success_after_enforcement']
    assert len(result['executed_calls']) == 2
    assert 'one tool call per reply' in seen[1][-1]['content']


def test_placeholder_rule_runs_before_argument_schema_validation():
    t = task(('BTC',))
    t['tools'][0]['function']['parameters']['properties']['x']['enum'] = ['BTC']
    result, seen = scripted(t, [reply(call('<symbol>')), reply(call('BTC')), reply()],
                            reject_placeholders=True)
    assert result['enforcement'][0]['rule'] == 'reject_placeholders'
    assert result['success_after_enforcement']
    assert 'earlier result or the user' in seen[1][-1]['content']


def test_false_rejection_audit_handles_malformed_skipped_arguments():
    from inference.toolcall.compare_runs import build_enforcement_report

    t = task()
    malformed = call(2, 'extra')
    malformed['function']['arguments'] = '{'
    result, _ = scripted(t, [reply(call(1), malformed), reply(call(2)), reply()], one_call=True)
    assert result['success_after_enforcement']
    baseline = dict(result, label='Q4-nothink-wait', task='a', category='multi_turn')
    row = dict(result, label='Q4-nothink-enforce', task='a', category='multi_turn')
    report = build_enforcement_report([baseline], [row], {'a': t})
    assert '| one_call | multi_turn | 1 | 0 |' in report


@pytest.mark.parametrize('category,values', [('single', (1,)), ('parallel', (1, 2)),
                                             ('multi_turn', (1, 2))])
@pytest.mark.parametrize('option', ['one_call', 'reject_placeholders', 'provenance'])
def test_matched_no_rejection_control(category, values, option):
    t = task(values, category, 'Use 1 and 2.')
    replies = ([reply(call(x)) for x in values] + [reply()] if category == 'multi_turn'
               else [reply(*(call(x, str(x)) for x in values))])
    # one_call intentionally rejects a parallel batch; use a one-call parallel control.
    if category == 'parallel' and option == 'one_call':
        t = task((1,), category, 'Use 1.')
        replies = [reply(call(1))]
    baseline, old_history = scripted(t, replies)
    enforced, new_history = scripted(t, replies, **{option: True})
    assert not enforced['enforcement']
    assert {k: enforced[k] for k in baseline} == baseline
    assert new_history == old_history


def test_wrong_call_keeps_historical_ordinal_result():
    t = task()
    _, old = scripted(t, [reply(call(9)), reply(call(2)), reply()])
    _, new = scripted(t, [reply(call(9)), reply(call(2)), reply()], one_call=True)
    assert new == old


def test_eight_correct_calls_score_even_at_cap():
    result, _ = scripted(task(tuple(range(8))), [reply(call(x)) for x in range(8)], one_call=True)
    assert result['turn_cap_hit']
    assert result['success_after_enforcement']
    assert not result['completed']


def test_correct_trajectory_before_timeout_still_scores():
    with patch('inference.toolcall.run_eval.complete', side_effect=[(reply(call(1)), .25), TimeoutError()]):
        result = evaluate(task((1,)), 'unused', 0, one_call=True)
    assert result['success_after_enforcement']
    assert result['transport_failure']
    assert not result['completed']


@pytest.mark.parametrize('value,rejected', [(9007199254740993, False), (9007199254740992, True)])
def test_large_integer_provenance(value, rejected):
    result, _ = scripted(task((value,), 'single', 'Use 9007199254740993.'),
                         [reply(call(value))], provenance=True)
    assert bool(result['enforcement']) == rejected


@pytest.mark.parametrize('name', ['default', 'enum', 'example', 'examples', 'const'])
def test_schema_property_names_do_not_grant_provenance(name):
    from inference.toolcall.enforcement import schema_values

    schema = {'type': 'object', 'properties': {name: {'type': 'string', 'description': 'forbidden'}}}
    assert list(schema_values(schema)) == []
    schema['properties'][name]['default'] = 'allowed'
    assert list(schema_values(schema)) == ['allowed']


@pytest.mark.parametrize('option', ['one_call', 'reject_placeholders', 'provenance'])
def test_invalid_argument_preserves_legacy_metrics(option):
    t = task((1,), 'single', 'Use invalid.')
    old, _ = scripted(t, [reply(call('invalid'))])
    new, _ = scripted(t, [reply(call('invalid'))], **{option: True})
    assert not new['enforcement']
    assert new['scores'] == old['scores']
    assert new['scores']['tool_selection']


@pytest.mark.parametrize('telemetry', ['85', 'unavailable'])
def test_gpu_abort_interrupts_evaluator_and_resumes(tmp_path, telemetry):
    """Run the actual shell driver with local substitutes, never a GPU or network."""
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / 'bin'
    fake.mkdir()
    cache = tmp_path / '.cache/hybrid-agents'
    cache.mkdir(parents=True)
    (tmp_path / 'research/results/phase2').mkdir(parents=True)
    (tmp_path / 'tasks.jsonl').write_text('{}\n')
    def executable(name, body):
        path = fake / name
        path.write_text('#!/bin/bash\n' + body)
        path.chmod(0o755)
    executable('pgrep', 'exit 1\n')
    executable('curl', 'exit 0\n')
    executable('sleep', '/bin/sleep 0.02\n')
    executable('git', 'echo forbidden-publish >> "$REPO/published"; exit 1\n')
    executable('llama-server', '''
if [[ "$*" == *--help* ]]; then echo '--reasoning-budget --jinja --host --port'; exit; fi
while :; do /bin/sleep 0.02; done
''')
    executable('nvidia-smi', '''
if [[ "$*" != *temperature.gpu* || "$*" == *memory.used* ]]; then echo '0, 40'; exit; fi
if [ ! -f "$REPO/evaluating" ]; then echo 40; exit; fi
if [ "$SIMULATED_TEMP" = unavailable ]; then exit 1; fi
echo "$SIMULATED_TEMP"
''')
    executable('uv', '''
if [ "$1" = sync ]; then exit; fi
if [[ "$*" != *inference.toolcall.run_eval* ]]; then exit 1; fi
echo "$*" >> "$REPO/invocations"
# A completed first task survives. An interrupted second task must not be written.
echo '{"label": "Q3_K_M-nothink-enforce", "task": "done"}' > "$RES"
touch "$REPO/evaluating"
while [ ! -f "$REPO/allow_finish" ]; do /bin/sleep 0.02; done
echo '{"label": "Q3_K_M-nothink-enforce", "task": "interrupted"}' >> "$RES"
''')
    # Two tasks ensure a completed first task cannot satisfy label resume checks.
    (tmp_path / 'tasks.jsonl').write_text('{}\n{}\n')
    env = dict(os.environ, HOME=str(tmp_path), REPO=str(tmp_path), LLAMA_BIN_DIR=str(fake),
               PATH=f'{fake}:/usr/bin:/bin', TASKS='tasks.jsonl', RES='research/results/phase2/run.jsonl',
               NOTHINK_QUANTS='Q3_K_M', THINK_QUANTS='', SIMULATED_TEMP=telemetry)
    driver = root / 'scripts/phase2/gpu_toolcall_v4.sh'
    proc = subprocess.run(['bash', str(driver)], env=env, capture_output=True, timeout=10, check=False)
    assert proc.returncode != 0
    status = (tmp_path / 'research/results/phase2/run-status.md').read_text()
    assert 'ok (ctx' not in status
    assert 'thermal' in status
    assert not (tmp_path / 'published').exists()
    rows = (tmp_path / env['RES']).read_text().splitlines()
    assert len(rows) == 1 and 'interrupted' not in rows[0]
    # A rerun must start evaluation again rather than skipping the incomplete label.
    (tmp_path / 'evaluating').unlink()
    # Simulate termination halfway through writing the active row.
    with (tmp_path / env['RES']).open('a') as stream:
        stream.write('{"label": "Q3_K_M-nothink-enforce", "task":')
    proc = subprocess.run(['bash', str(driver)], env=env, capture_output=True, timeout=10, check=False)
    assert proc.returncode != 0
    assert (tmp_path / 'evaluating').exists()
    invocations = (tmp_path / 'invocations').read_text().splitlines()
    assert len(invocations) == 2
    assert all('--label Q3_K_M-nothink-enforce' in cmd for cmd in invocations)


@pytest.mark.parametrize('bad_call', [None, {}, {'function': {'name': 'f', 'arguments': '{}'}}])
def test_malformed_call_preserves_legacy_scores(bad_call):
    t = task((1,), 'single')
    old, _ = scripted(t, [reply(bad_call)])
    new, _ = scripted(t, [reply(bad_call)], one_call=True)
    assert new['scores'] == old['scores']


def test_single_stops_before_later_timeout():
    with patch('inference.toolcall.run_eval.complete', side_effect=[(reply(call(1)), .25), TimeoutError()]):
        result = evaluate(task((1,), 'single'), 'unused', 0, one_call=True)
    assert result['success_after_enforcement']
    assert result['completed']
    assert result['transport_failure'] is None
    assert len(result['responses']) == 1
