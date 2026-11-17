"""Regression evidence for the historical table and comparison definitions."""

from itertools import product

import pytest

from inference.toolcall.compare_runs import (
    RESULTS,
    ROOT,
    build_report,
    index,
    interaction,
    load,
    paired,
    recovered,
    score_cells,
    sign_test,
)


def test_reproduce_v2_table():
    groups = index(load(RESULTS / 'toolcall-v2.jsonl'))
    markdown = (RESULTS / 'toolcall-v2-analysis.md').read_text()
    expected = [line.split('|')[2:-1] for line in markdown.splitlines()
                if line.startswith('| Q') and 'thinking o' in line and ' vs ' not in line]
    assert len(expected) == len(groups) == 7
    for rows, cells in zip(groups.values(), expected, strict=True):
        assert score_cells(list(rows.values())) == [c.strip() for c in cells]
    comparisons = [
        ('Q3_K_M-nothink', 'Q4_K_M-nothink', None, 3, 13),
        ('Q3_K_M-nothink', 'Q8_0-nothink', None, 2, 15),
        ('Q4_K_M-nothink', 'Q5_K_M-nothink', None, 2, 5),
        ('Q4_K_M-nothink', 'Q6_K-nothink', None, 4, 5),
        ('Q4_K_M-nothink', 'Q8_0-nothink', None, 4, 7),
        ('Q4_K_M-think', 'Q4_K_M-nothink', None, 11, 11),
        ('Q4_K_M-think', 'Q4_K_M-nothink', 'multi_turn', 4, 10),
        ('Q6_K-think', 'Q6_K-nothink', None, 13, 8),
    ]
    for a, b, category, wins, losses in comparisons:
        assert paired(groups[a], groups[b], category) == (wins, losses, sign_test(wins, losses))


def test_sign_test_and_interaction():
    assert sign_test(0, 0) == 1
    assert sign_test(0, 6) == 0.03125
    assert sign_test(3, 13) == sign_test(13, 3)
    values = [1, -2, 0, 2]
    sums = [sum(x * d for x, d in zip(signs, values, strict=True))
            for signs in product((-1, 1), repeat=len(values))]
    assert interaction(values) == sum(abs(s) >= abs(sum(values)) for s in sums) / len(sums)


def test_recovery_order_arguments_and_extras():
    def call(value):
        return {'function': {'name': 'f', 'arguments': '{"x": ' + str(value) + '}'}}

    task = {'expected': [{'name': 'f', 'args': {'x': {'eq': 1}}},
                         {'name': 'f', 'args': {'x': {'eq': 2}}}]}

    def row(values):
        return {'responses': [{'choices': [{'message': {'tool_calls': [call(v)]}}]}
                              for v in values]}

    assert recovered(row([0, 1, 0, 2]), task)
    assert not recovered(row([2, 1]), task)
    assert not recovered(row([1, 3]), task)
    assert not recovered(row([1]), task)


def test_support_validation():
    with pytest.raises(ValueError, match='Unmatched'):
        paired({}, {'a': {}})
    with pytest.raises(ValueError, match='Repeated'):
        index([{'label': 'x', 'task': 'a'}] * 2)


def test_report_reproducible():
    tasks = {t['id']: t for t in load(ROOT / 'inference/toolcall/tasks/finance_v1_wait.jsonl')}
    generated = build_report(load(RESULTS / 'toolcall-v2.jsonl'),
                             load(RESULTS / 'toolcall-v3-wait.jsonl'), tasks)
    assert generated == (RESULTS / 'toolcall-v3-analysis.md').read_text()
    assert generated.count('\n## ') == 7


@pytest.mark.parametrize('success', [False, True])
def test_findings_follow_changed_results(success):
    old = load(RESULTS / 'toolcall-v2.jsonl')
    new = load(RESULTS / 'toolcall-v3-wait.jsonl')
    tasks = {t['id']: t for t in load(ROOT / 'inference/toolcall/tasks/finance_v1_wait.jsonl')}
    for row in new:
        row['scores']['args_correct'] = success
        row['scores']['abstained_correctly'] = success
        row['responses'] = []
    findings = build_report(old, new, tasks).split('## 7. Findings\n\n')[1]
    if success:
        assert 'eliminating strict multi-step failures' in findings
    else:
        assert 'improves strict' not in findings
        assert 'Q3 loses 19/40' in findings
        assert '24 category comparisons are significant' in findings
    assert 'Q4 thinking retains 7/31' not in findings
    assert 'Recovered failures remain present in every run' not in findings
    if not success:
        assert 'abstention scores are unchanged' not in findings
    assert 'Q3 gains 12/40' not in findings
    assert 'Overall gains for Q4 thinking-off, Q8 thinking-off and Q4 thinking-on' not in findings


def test_findings_do_not_claim_elimination_when_baseline_has_no_failures():
    old = load(RESULTS / 'toolcall-v2.jsonl')
    new = load(RESULTS / 'toolcall-v3-wait.jsonl')
    tasks = {t['id']: t for t in load(ROOT / 'inference/toolcall/tasks/finance_v1_wait.jsonl')}
    for row in old + new:
        if row['category'] == 'multi_turn':
            row['scores']['args_correct'] = True
    findings = build_report(old, new, tasks).split('## 7. Findings\n\n')[1]
    assert 'eliminating strict multi-step failures' not in findings
    assert 'No strict multi-step failures occur in v3.' in findings


def test_findings_do_not_claim_recoveries_remain_without_baseline_recoveries():
    old = load(RESULTS / 'toolcall-v2.jsonl')
    new = load(RESULTS / 'toolcall-v3-wait.jsonl')
    tasks = {t['id']: t for t in load(ROOT / 'inference/toolcall/tasks/finance_v1_wait.jsonl')}
    for row in old:
        row['responses'] = []
    findings = build_report(old, new, tasks).split('## 7. Findings\n\n')[1]
    assert 'Recovered failures remain' not in findings
    assert 'Recovered failures occur in 6/6 v3 runs' in findings
