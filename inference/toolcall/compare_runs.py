"""Reproduce paired tool-call analysis using only the standard library.

Run: python -m inference.toolcall.compare_runs [--output PATH]
"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from .score import matches

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'research/results/phase2'
CATEGORIES = ('single', 'parallel', 'abstain', 'clarify', 'multi_turn')


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def passed(row):
    return bool(row['scores']['abstained_correctly'] if row['category'] in
                ('abstain', 'clarify') else row['scores']['args_correct'])


def index(rows):
    result = {}
    for row in rows:
        group = result.setdefault(row['label'], {})
        if row['task'] in group:
            raise ValueError('Repeated task: analysis requires one observation per task')
        group[row['task']] = row
    return result


def sign_test(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n)


def paired(left, right, category=None):
    if left.keys() != right.keys():
        raise ValueError('Unmatched task support')
    pairs = [(passed(left[t]), passed(right[t])) for t in left
             if category is None or left[t]['category'] == category]
    a = sum(x and not y for x, y in pairs)
    b = sum(y and not x for x, y in pairs)
    return a, b, sign_test(a, b)


def calls(row):
    return [response['choices'][0]['message'].get('tool_calls') or []
            for response in row['responses']]


def recovered(row, task):
    """Expected calls form an ordered subsequence; extra/wrong calls are allowed."""
    expected = iter(task['expected'])
    target = next(expected, None)
    for turn in calls(row):
        for call in turn:
            try:
                fn = call['function']
                args = json.loads(fn['arguments'])
                if (target is not None and fn['name'] == target['name']
                        and isinstance(args, dict) and args.keys() == target['args'].keys()
                        and all(matches(args[k], m) for k, m in target['args'].items())):
                    target = next(expected, None)
            except (KeyError, TypeError, ValueError):
                continue
    return target is None


def score_cells(rows):
    groups = [rows] + [[r for r in rows if r['category'] == c] for c in CATEGORIES]
    return [f'{sum(map(passed, g)) / len(g):.2f}' for g in groups] + [
        f'{statistics.median(r["latency_seconds"] for r in rows):.1f}']


def interaction(differences):
    """Exact paired sign-flip test of differences in task-level improvements."""
    counts = Counter({0: 1})
    for d in differences:
        if d:
            updated = Counter()
            for value, count in counts.items():
                updated[value + abs(d)] += count
                updated[value - abs(d)] += count
            counts = updated
    observed = abs(sum(differences))
    return sum(n for x, n in counts.items() if abs(x) >= observed) / sum(counts.values())


def pcell(p):
    return str(p) + (' (not significant)' if p >= .05 else '')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])



def joined(items):
    if len(items) < 2:
        return ''.join(items)
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def run_name(label):
    return label.split('_')[0] + (' thinking-off' if label.endswith('-nothink') else ' thinking-on')


def findings(old, new, tasks, pairs):
    """Describe measured changes, retaining the original wording when supported."""
    multi = [(b, *paired(old[a], new[b], 'multi_turn')) for a, b in pairs]
    sentences = []
    if all(y > x and p < .05 for _, x, y, p in multi):
        change = ('The wait instruction improves strict multi-step success in every matched run '
                  '(all multi-step sign-test p < 0.05)')
    else:
        change = 'Multi-step changes: ' + '; '.join(
            f'{run_name(a)}: {y - x:+d} tasks ({pcell(p)})'
            for (a, _), (_, x, y, p) in zip(pairs, multi, strict=True))
    failures = sum(not passed(r) for group in new.values() for r in group.values()
                   if r['category'] == 'multi_turn')
    baseline_failures = sum(not passed(r) for a, _ in pairs for r in old[a].values()
                            if r['category'] == 'multi_turn')
    if failures:
        sentences.append(change + ', but does not eliminate failures.')
    elif baseline_failures:
        sentences.append(change + ', eliminating strict multi-step failures.')
    else:
        sentences.append(change + '. No strict multi-step failures occur in v3.')

    batches, recoveries, baseline_recoveries = {}, [], []
    for a, b in pairs:
        eligible = [r for t, r in new[b].items() if r['category'] == 'multi_turn'
                    and len(tasks[t]['expected']) >= 2]
        batches[a] = (sum(bool(calls(r)) and len(calls(r)[0]) >= 2 for r in eligible),
                      len(eligible))
        failed = [r for r in new[b].values() if r['category'] == 'multi_turn' and not passed(r)]
        recoveries.append(sum(recovered(r, tasks[r['task']]) for r in failed))
        baseline_failed = [r for r in old[a].values()
                           if r['category'] == 'multi_turn' and not passed(r)]
        baseline_recoveries.append(sum(recovered(r, tasks[r['task']]) for r in baseline_failed))
    off = [a for a, _ in pairs if a.endswith('-nothink')]
    on = [a for a, _ in pairs if not a.endswith('-nothink')]
    if (off and len({batches[a] for a in off}) == 1
            and all(batches[a][0] < sum(bool(calls(r)) and len(calls(r)[0]) >= 2
                                      for t, r in old[a].items()
                                      if r['category'] == 'multi_turn'
                                      and len(tasks[t]['expected']) >= 2) for a in off)):
        n, total = batches[off[0]]
        batch_text = f'First-response batching falls to {n}/{total} at every thinking-off precision'
    else:
        batch_text = 'First-response batching: ' + '; '.join(
            f'{run_name(a)} {batches[a][0]}/{batches[a][1]}' for a in off)
    if on:
        batch_text += ', while ' + joined([
            f'{a.split("_")[0]} thinking retains {batches[a][0]}/{batches[a][1]} batched responses'
            for a in on])
    sentences.append(batch_text + '.')
    if recoveries and all(recoveries) and all(baseline_recoveries):
        sentences.append('Recovered failures remain present in every run and are excluded from headline scores.')
    else:
        sentences.append(f'Recovered failures occur in {sum(n > 0 for n in recoveries)}/{len(pairs)} '
                         'v3 runs and are excluded from headline scores.')

    drops = [c for c in CATEGORIES[:-1] if any(
        paired(old[a], new[b], c)[0] > paired(old[a], new[b], c)[1] for a, b in pairs)]
    category_names = {'single': 'single', 'parallel': 'parallel', 'clarify': 'clarification',
                      'abstain': 'abstention'}
    side_tests = [paired(old[a], new[b], c) for a, b in pairs for c in CATEGORIES[:-1]]
    if drops:
        side = joined([category_names[c] for c in drops]).capitalize() + ' scores sometimes drop'
    else:
        side = 'No single, parallel, abstention or clarification score drops'
    if all(p >= .05 for _, _, p in side_tests):
        side += ', but every category comparison is not significant'
    else:
        side += f'; {sum(p < .05 for _, _, p in side_tests)} category comparisons are significant'
    if all(paired(old[a], new[b], 'abstain')[:2] == (0, 0) for a, b in pairs):
        side += '; abstention scores are unchanged'
    sentences.append(side + '.')

    base = 'Q3_K_M-nothink'
    ids = [t for t, r in old[base].items() if r['category'] == 'multi_turn']

    def delta(a, t):
        return int(passed(new[a + '-wait'][t])) - int(passed(old[a][t]))

    higher = [a for a in off if a != base]
    gains = [sum(delta(a, t) for t in ids) for a in higher]
    tests = [interaction([delta(base, t) - delta(a, t) for t in ids]) for a in higher]
    base_gain = sum(delta(base, t) for t in ids)
    if gains:
        direction = 'gains' if base_gain >= 0 else 'loses'
        precision = (f'Q3 {direction} {abs(base_gain)}/{len(ids)} multi-step tasks versus '
                     f'{min(gains)}/{len(ids)}–{max(gains)}/{len(ids)} at higher precisions')
        precision += (', and every paired precision interaction is not significant.'
                      if all(p >= .05 for p in tests) else
                      f'; {sum(p < .05 for p in tests)} paired precision interactions are significant.')
        sentences.append(precision)
    overall = [(a, *paired(old[a], new[b])) for a, b in pairs]
    nonsig_gains = [run_name(a) for a, x, y, p in overall if y > x and p >= .05]
    if nonsig_gains:
        sentences.append('Overall gains for ' + joined(nonsig_gains) + ' are not significant.')
    else:
        sentences.append('Overall changes: ' + '; '.join(
            f'{run_name(a)}: {y - x:+d} tasks ({pcell(p)})' for a, x, y, p in overall) + '.')
    return ' '.join(sentences)


def build_report(v2, v3, tasks):
    old, new = index(v2), index(v3)
    labels = list(new)
    pairs = [(label.removesuffix('-wait'), label) for label in labels]
    lines = ['# Tool-calling v3: wait instruction', '',
             ('Sources: `toolcall-v2.jsonl`, `toolcall-v3-wait.jsonl`, and '
             '`inference/toolcall/tasks/finance_v1_wait.jsonl`. Regenerate with '
             '`python -m inference.toolcall.compare_runs`. Strict success uses argument accuracy '
             'for call tasks and correct abstention/nonempty clarification for no-call tasks. '
             'One observation per task; paired tests exclude ties. Tests are exploratory, '
             'two-sided and unadjusted for multiple comparisons. Q8 uses 4k context; others 8k. '
             'Q6 thinking has no v3 counterpart. Seconds per task is the median, matching v2.'), '', '## 1. Scores', '',
             table(['Run', 'All 114', 'Single (36)', 'Parallel (12)', 'Abstain (12)',
                    'Clarify (14)', 'Multi-step (40)', 'Seconds per task'],
                   [[label, *score_cells(list(new[label].values()))] for label in labels]), '',
             '## 2. Paired changes', '',
             'Exact binomial sign-test p values; matching uses task IDs.', '',
             table(['v2 → v3 label', 'Subset', 'Only v2 right', 'Only v3 right', 'p'],
                   [[b, category or 'all', aonly, bonly, pcell(p)]
                    for a, b in pairs for category in (None, 'multi_turn')
                    for aonly, bonly, p in [paired(old[a], new[b], category)]]), '',
             '## 3. Batching rate', '',
             ('Denominator: multi-step tasks with at least two expected calls. '
             'Numerator: first response has at least two calls (including independent calls).'), '']
    batching, recovery = [], []
    for a, b in pairs:
        batch, rec = [b], [b]
        for group in (old[a], new[b]):
            eligible = [r for t, r in group.items() if r['category'] == 'multi_turn'
                        and len(tasks[t]['expected']) >= 2]
            n = sum(bool(calls(r)) and len(calls(r)[0]) >= 2 for r in eligible)
            batch.append(f'{n}/{len(eligible)} ({n / len(eligible):.1%})')
            failed = [r for r in group.values() if r['category'] == 'multi_turn' and not passed(r)]
            n = sum(recovered(r, tasks[r['task']]) for r in failed)
            rec.append(f'{n}/{len(failed)}' if failed else '0/0')
        batching.append(batch)
        recovery.append(rec)
    lines += [table(['Label', 'v2', 'v3'], batching), '', '## 4. Recovered failures', '',
              ('Counts / failed multi-step tasks. Expected calls must match arguments exactly '
              '(using task matchers), in order, as a subsequence of all response calls; '
              'extra calls remain strict failures. This does not prove correct timing.'), '',
              table(['Label', 'v2', 'v3'], recovery), '', '## 5. Side effects', '',
              ('Losses and gains are task counts, not percentage points. A drop is supported '
              'only when losses exceed gains and p < 0.05.'), '',
              table(['Label', 'Category', 'v2 correct', 'v3 correct', 'Losses', 'Gains', 'p'],
                    [[b, c, sum(passed(r) for r in old[a].values() if r['category'] == c),
                      sum(passed(r) for r in new[b].values() if r['category'] == c), x, y, pcell(p)]
                     for a, b in pairs for c in CATEGORIES[:-1]
                     for x, y, p in [paired(old[a], new[b], c)]]), '',
              '## 6. Precision and improvement', '',
              ('Compare Q3 thinking-off with each higher precision thinking-off on the same '
              'multi-step tasks. Delta = v3 success − v2 success. The exact paired sign-flip '
              'test swaps the two precision deltas independently per task, conditional on '
              'their absolute differences; it assumes exchangeability under the null. '
              'No pooling of correlated precision runs or causal precision claim.'), '']
    base = 'Q3_K_M-nothink'
    ids = [t for t, r in old[base].items() if r['category'] == 'multi_turn']
    def delta(label, t):
        return int(passed(new[label + '-wait'][t])) - int(passed(old[label][t]))
    rows = []
    for a, _ in pairs:
        if a == base or not a.endswith('-nothink'):
            continue
        differences = [delta(base, t) - delta(a, t) for t in ids]
        rows.append([a, f'{sum(delta(base, t) for t in ids)}/{len(ids)}',
                     f'{sum(delta(a, t) for t in ids)}/{len(ids)}',
                     f'{sum(differences)/len(ids):+.3f}', pcell(interaction(differences))])
    lines += [table(['Higher precision', 'Q3 net gain', 'Higher net gain',
                     'Q3 minus higher (share)', 'p'], rows), '', '## 7. Findings', '',
              findings(old, new, tasks, pairs), '']
    return '\n'.join(lines)


def build_enforcement_report(v3, v4, tasks):
    """Paired strict/executed scores, costs, and audited rejection counts."""
    from .enforcement import call_matches

    old, new = index(v3), index(v4)
    lines = ['# Tool-calling v4: harness enforcement', '',
             ('Sources: `toolcall-v3-wait.jsonl`, `toolcall-v4-enforce.jsonl`, '
             'and `finance_v1_wait.jsonl`. One observation per task. Exact two-sided '
             'sign tests exclude ties; exploratory, unadjusted p values. Strict scoring '
             'includes every attempted call. Executed scoring excludes rejected calls '
             'and measures only whether the executed trajectory matches; cap hits, completion, '
             'and transport failures are recorded separately. '
             'False rejections match a complete expected call (any expected position); '
             'counts are rejection events, so multiple bad arguments may count separately. '
             'Model turns count completion requests in enforced runs (including failed '
             'requests), and returned responses in v3; medians include failed tasks.'), '']
    score_rows, pair_rows, audit_rows, findings = [], [], [], []
    for label, group in new.items():
        base = label.removesuffix('-enforce').removesuffix('-prov') + '-wait'
        baseline = old[base]
        if baseline.keys() != group.keys():
            raise ValueError('Unmatched task support')
        for category in (None, *CATEGORIES):
            ids = [t for t in group if category is None or group[t]['category'] == category]
            if not ids:
                continue
            subset = category or 'all'
            before = [passed(baseline[t]) for t in ids]
            strict = [passed(group[t]) for t in ids]
            after = [bool(group[t]['success_after_enforcement']) for t in ids]
            score_rows.append([label, subset, len(ids), f'{sum(before)/len(ids):.3f}',
                               f'{sum(strict)/len(ids):.3f}', f'{sum(after)/len(ids):.3f}',
                               f'{statistics.median(baseline[t]["latency_seconds"] for t in ids):.2f}',
                               f'{statistics.median(group[t]["latency_seconds"] for t in ids):.2f}',
                               statistics.median(len(baseline[t]['responses']) for t in ids),
                               statistics.median(group[t]['model_turns'] for t in ids),
                               sum(group[t]['turn_cap_hit'] for t in ids)])
            for metric, values in [('strict', strict), ('executed', after)]:
                a = sum(x and not y for x, y in zip(before, values, strict=True))
                b = sum(y and not x for x, y in zip(before, values, strict=True))
                pair_rows.append([label, subset, metric, a, b, pcell(sign_test(a, b))])
            if category in ('single', 'parallel'):
                for metric, values in [('strict', strict), ('executed', after)]:
                    direction = 'lowers' if sum(values) < sum(before) else (
                        'raises' if sum(values) > sum(before) else 'does not change')
                    findings.append(f'{label} {direction} {category} scores ({metric}); '
                                    'enforced run vs v3; differences include the harness limits listed separately.')
        counts, false = Counter(), Counter()
        for t, row in group.items():
            for event in row['enforcement']:
                key = (event['rule'], row['category'])
                counts[key] += 1
                batch = calls(row)[event['turn'] - 1]
                call = batch[event['call_index']]
                false[key] += any(call_matches(call, expected) for expected in tasks[t]['expected'])
        for rule in ('one_call', 'reject_placeholders', 'provenance'):
            for category in CATEGORIES:
                key = rule, category
                audit_rows.append([label, rule, category, counts[key], false[key]])
            audit_rows.append([label, rule, 'all',
                               sum(v for (r, _), v in counts.items() if r == rule),
                               sum(v for (r, _), v in false.items() if r == rule)])
    for label, group in new.items():
        if not label.endswith('-prov'):
            continue
        control = new.get(label.removesuffix('-prov') + '-enforce')
        if control is None:
            continue
        if control.keys() != group.keys():
            raise ValueError('Unmatched task support')
        for category in CATEGORIES:
            ids = [t for t in group if group[t]['category'] == category]
            if not ids:
                continue
            for metric in ('strict', 'executed'):
                def success(row, metric=metric):
                    return passed(row) if metric == 'strict' else row['success_after_enforcement']
                before, after = [success(control[t]) for t in ids], [success(group[t]) for t in ids]
                a = sum(x and not y for x, y in zip(before, after, strict=True))
                b = sum(y and not x for x, y in zip(before, after, strict=True))
                pair_rows.append([label + ' vs enforce', category, metric, a, b, pcell(sign_test(a, b))])
                if category in ('single', 'parallel'):
                    direction = 'lowers' if sum(after) < sum(before) else (
                        'raises' if sum(after) > sum(before) else 'does not change')
                    findings.append(f'Adding provenance {direction} {category} scores ({metric}) '
                                    f'for {label}; paired p={pcell(sign_test(a, b))}.')
    lines += [table(['Run', 'Subset', 'N', 'v3 strict', 'v4 strict', 'Success after enforcement',
                     'v3 Median seconds', 'v4 Median seconds', 'v3 Median turns', 'v4 Median turns',
                     'Cap hits'], score_rows), '',
              table(['Run', 'Subset', 'Metric', 'Only v3 right', 'Only v4 right', 'p'], pair_rows), '',
              table(['Run', 'Rule', 'Category', 'Rejections', 'False rejections'], audit_rows), '',
              ' '.join(findings), '',
              ('Remaining harness differences (enforced run vs v3): dependent tasks have an '
              'eight-reply cap instead of expected-call count + 2; rejected calls receive '
              'error results and do not consume fixture ordinals. Single/parallel tasks '
              'stop after the first reply in both runners; accepted dependent calls use '
              'ordinal fixture dispatch in both. Executed success is an additional '
              'trajectory metric, independent of transport failure or completion. Enforced '
              'turn counts include failed completion requests; v3 counts returned responses.'), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS / 'toolcall-v3-analysis.md')
    parser.add_argument('--v4', action='store_true', help='Compare enforcement against v3')
    args = parser.parse_args()
    tasks = {t['id']: t for t in load(ROOT / 'inference/toolcall/tasks/finance_v1_wait.jsonl')}
    if args.v4:
        output = (RESULTS / 'toolcall-v4-analysis.md' if args.output == RESULTS / 'toolcall-v3-analysis.md'
                  else args.output)
        output.write_text(build_enforcement_report(load(RESULTS / 'toolcall-v3-wait.jsonl'),
                                                   load(RESULTS / 'toolcall-v4-enforce.jsonl'), tasks))
        return
    args.output.write_text(build_report(load(RESULTS / 'toolcall-v2.jsonl'),
                                       load(RESULTS / 'toolcall-v3-wait.jsonl'), tasks))


if __name__ == '__main__':
    main()
