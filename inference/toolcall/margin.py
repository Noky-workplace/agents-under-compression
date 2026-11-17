"""Q8 margins: owner-only measurement, pre-registered standard-library analysis.

python -m inference.toolcall.margin              # analyze completed ASUS output
python -m inference.toolcall.margin --measure    # running localhost Q8 server only
Missing probabilities (including terminal tokens) are fatal, never approximated.
"""

import argparse
import json
import math
import random
import re
import statistics
import urllib.error
import urllib.request
from pathlib import Path

from .compare_runs import RESULTS, load, passed
from .score import score
from .tasks import load_tasks

TASKS = Path(__file__).with_name('tasks') / 'finance_v1_wait.jsonl'
BASELINE = RESULTS / 'toolcall-v3-wait.jsonl'
OUTPUT = RESULTS / 'margin-q8.jsonl'


def auc(scores, positives):
    """Mann–Whitney AUC with half credit for ties, higher score predicts positive."""
    n = sum(positives)
    if not n or n == len(positives):
        return None
    ordered = sorted(zip(scores, positives, strict=True))
    rank_sum = 0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(p for _, p in ordered[i:j])
        i = j
    return (rank_sum - n * (n + 1) / 2) / (n * (len(scores) - n))


def permutation(scores, positives):
    observed = auc(scores, positives)
    n = sum(positives)
    p = None
    if observed is not None:
        # Precompute tied ranks once; shuffled labels preserve the positive count.
        ranks = {s: statistics.mean(i + 1 for i, x in enumerate(sorted(scores)) if x == s)
                 for s in set(scores)}
        values = [ranks[s] for s in scores]
        threshold = sum(v for v, label in zip(values, positives, strict=True) if label)
        rng = random.Random(0)
        labels = list(positives)
        extreme = 0
        for _ in range(10000):
            rng.shuffle(labels)
            extreme += sum(v for v, label in zip(values, labels, strict=True) if label) >= threshold
        p = (extreme + 1) / 10001
    decision = ('not informative' if n < 8 else
                'supported' if observed is not None and observed > .5 and p < .05 else 'not supported')
    return {'n': len(scores), 'positives': n, 'auc': observed, 'p_value': p, 'decision': decision,
                'permutations': 10000, 'seed': 0}


def positions(response):
    """Normalize OpenAI logprobs and old/new llama.cpp native probabilities."""
    items = response.get('completion_probabilities')
    if items is None:
        items = (response['choices'][0].get('logprobs') or {}).get('content')
    if not items:
        raise ValueError('No generated-token probabilities returned')
    result = []
    for item in items:
        if 'top_logprobs' in item:
            result.append({k: item[k] for k in ('token', 'logprob', 'top_logprobs')})
        else:
            candidates = [{'token': p['tok_str'], 'logprob': math.log(p['prob'])}
                          for p in item['probs'][:10] if p['prob'] > 0]
            token = item['content']
            selected = next((p['logprob'] for p in candidates if p['token'] == token), None)
            if selected is None:
                raise ValueError('Generated token absent from native candidates')
            result.append({'token': token, 'logprob': selected, 'top_logprobs': candidates})
        if item.get('bytes') is not None:
            result[-1]['bytes'] = item['bytes']
        result[-1]['top_logprobs'] = result[-1]['top_logprobs'][:10]
    token_margins(result)
    return result


def token_margins(tokens):
    margins = []
    for pos in tokens:
        other = [p['logprob'] for p in pos['top_logprobs'] if p['token'] != pos['token']]
        if not other or not all(math.isfinite(v) for v in [pos['logprob'], *other]):
            raise ValueError('Missing alternative or nonfinite token probability')
        margins.append(pos['logprob'] - max(other))
    if not margins:
        raise ValueError('Empty trajectory')
    return margins


def validate_coverage(response, tokens, raw=False):
    """Bind probabilities to reply text before parsing or persisting a turn."""
    count = response.get('tokens_predicted') if raw else response.get('usage', {}).get('completion_tokens')
    # Decode once across positions: a UTF-8 character can span token boundaries.
    try:
        text = b''.join(bytes(p['bytes']) if p.get('bytes') is not None
                        else p['token'].encode('utf-8') for p in tokens).decode('utf-8')
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError('Probabilities cannot be decoded as UTF-8 reply text') from exc
    if count != len(tokens) or not text.endswith(('<|im_end|>', '<|endoftext|>')):
        raise ValueError('Probabilities omit generated/template/end-of-turn positions')
    # The server may omit a terminal marker from content. Preserve all other
    # characters, including whitespace and tool/template text, on both sides.
    def normalized(value):
        return value.removesuffix('<|im_end|>').removesuffix('<|endoftext|>')

    if raw:
        if normalized(text) != normalized(response['content']):
            raise ValueError('Raw probabilities do not match response content')
    else:
        message = response['choices'][0]['message']
        parsed = raw_message(text)
        if canonical([parsed['tool_calls']]) != canonical([message.get('tool_calls') or []]):
            raise ValueError('Chat probabilities do not cover complete tool-call text')
        if parsed['content'] != normalized(message.get('content') or ''):
            raise ValueError('Chat probabilities do not match response content')


def raw_message(text):
    """Decode Qwen JSON tool blocks into the envelopes consumed by score/evaluate."""
    calls = []
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if text.count('<tool_call>') != len(blocks) or text.count('</tool_call>') != len(blocks):
        raise ValueError('Unclosed raw tool call')
    for i, block in enumerate(blocks):
        fn = json.loads(block)
        if not isinstance(fn['name'], str) or not isinstance(fn['arguments'], dict):
            raise TypeError('Invalid raw tool call')
        calls.append({'id': f'margin-{i}', 'type': 'function', 'function': {
            'name': fn['name'], 'arguments': json.dumps(fn['arguments'])}})
    content = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    content = content.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    return {'role': 'assistant', 'content': content, 'tool_calls': calls}


def canonical(turns):
    return [[(c['function']['name'], json.dumps(json.loads(c['function']['arguments']),
                                               sort_keys=True, separators=(',', ':')))
             for c in turn] for turn in turns]


def mismatch_share(current, baseline):
    if current.keys() != baseline.keys():
        raise ValueError('Unmatched task support')
    share = sum(canonical(current[t]) != canonical(baseline[t]) for t in baseline) / len(baseline)
    if share > .05:
        raise ValueError(f'Trajectory mismatch {share:.6f} exceeds 5% stop rule')
    return share


def baseline_groups(rows):
    groups = {}
    for row in rows:
        # Historical files use the short label; the model path supplies the full prefix.
        label = row['label'].removeprefix('Qwen_Qwen3-8B-own-')
        if label in ('Q8_0-nothink-wait', 'Q3_K_M-nothink-wait', 'Q4_K_M-nothink-wait'):
            group = groups.setdefault(label.split('-')[0], {})
            if row['task'] in group or row.get('seed', 0) != 0:
                raise ValueError('Duplicate task or nonzero seed in baseline')
            group[row['task']] = row
    return groups


def baseline_calls(group):
    return {t: [r['choices'][0]['message'].get('tool_calls') or [] for r in row['responses']]
            for t, row in group.items()}


def request(base, endpoint, payload):
    req = urllib.request.Request(base.rstrip('/') + endpoint, json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def generate(base, task, messages, mode):
    payload = {'messages': messages, 'tools': task['tools'], 'temperature': 0, 'seed': 0}
    if mode != 'raw':
        payload.update({'logprobs': True, 'top_logprobs': 10} if mode == 'chat' else {'n_probs': 10})
        response = request(base, '/v1/chat/completions', payload)
        tokens = positions(response)
        validate_coverage(response, tokens)
        message = response['choices'][0]['message']
    else:
        rendered = request(base, '/apply-template', {
            'messages': messages, 'tools': task['tools'], 'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': False}})
        response = request(base, '/completion', {
            'prompt': rendered['prompt'], 'temperature': 0, 'seed': 0, 'n_probs': 10,
            'n_predict': 4096, 'post_sampling_probs': False})
        if response.get('truncated') or response.get('stopped_limit'):
            raise ValueError('Raw trajectory truncated')
        tokens = positions(response)
        validate_coverage(response, tokens, raw=True)
        message = raw_message(response['content'])
    if response.get('choices', [{}])[0].get('finish_reason') == 'length':
        raise ValueError('Chat trajectory truncated')
    return message, tokens


def measure(base, output=OUTPUT):
    tasks = load_tasks(TASKS)
    groups = baseline_groups(load(BASELINE))
    if set(groups['Q8_0']) != {t['id'] for t in tasks}:
        raise ValueError('Baseline task support differs from fixed task set')
    output = Path(output)
    partial = output.with_suffix('.partial.jsonl')
    if output.exists() or partial.exists():
        raise ValueError('One-run stop rule: existing output/partial; inspect before doing anything')
    diagnostics = []
    mode = None
    for candidate in ('chat', 'native-chat', 'raw'):
        try:
            message, _ = generate(base, tasks[0], tasks[0]['messages'], candidate)
            if not message.get('tool_calls'):
                raise ValueError('Preflight did not generate a tool call')
            mode = candidate
            diagnostics.append(f'{candidate}: token count, terminal marker and reply-text agreement accepted')
            break
        except (ValueError, KeyError, TypeError, urllib.error.HTTPError) as exc:
            diagnostics.append(f'{candidate}: {exc}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.preflight.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    if mode is None:
        raise ValueError('Neither chat nor raw provides complete token probabilities; see preflight JSON')
    current = {}
    size = 0
    with partial.open('x') as stream:
        for task in tasks:
            messages = list(task['messages'])
            current[task['id']] = []
            calls = []
            limit = len(task['expected']) + 2 if task['category'] == 'multi_turn' else 1
            for turn in range(limit):
                message, tokens = generate(base, task, messages, mode)
                turn_calls = message.get('tool_calls') or []
                current[task['id']].append(turn_calls)
                row = {'task': task['id'], 'turn': turn, 'category': task['category'], 'mode': mode,
                           'tokens': tokens, 'calls': turn_calls}
                line = json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n'
                size += len(line.encode())
                if size >= 20_000_000:
                    raise ValueError('20 MB output limit exceeded')
                stream.write(line)
                stream.flush()
                offset = len(calls)
                calls.extend(turn_calls)
                # Exactly the original v3 evaluate stopping and scripted tool-result policy.
                if not score(task, turn_calls)['schema_valid']:
                    break
                if any(not c.get('id') or c.get('type') != 'function' for c in turn_calls):
                    break
                if task['category'] != 'multi_turn' or not turn_calls:
                    break
                messages.append(message)
                if len(calls) > len(task['tool_results']):
                    break
                for i, call in enumerate(turn_calls):
                    messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                         'content': json.dumps(task['tool_results'][offset + i])})
    try:
        share = mismatch_share(current, baseline_calls(groups['Q8_0']))
    except ValueError as exc:
        output.with_suffix('.validation.json').write_text(json.dumps(
            {'status': 'rejected', 'error': str(exc)}, indent=2) + '\n')
        raise
    output.with_suffix('.validation.json').write_text(json.dumps({
        'tasks': len(tasks), 'mismatch_share': share, 'mode': mode, 'bytes': size}, indent=2) + '\n')
    # The shell promotes only after its final thermal/telemetry checks.
    return partial


def analyze(rows, baseline):
    groups = baseline_groups(baseline)
    by_task = {}
    replay = {}
    for row in rows:
        task = row['task']
        turns = replay.setdefault(task, [])
        if row['turn'] != len(turns):
            raise ValueError('Duplicate or missing turn')
        turns.append(row['calls'])
        by_task.setdefault(task, []).extend(token_margins(row['tokens']))
    share = mismatch_share(replay, baseline_calls(groups['Q8_0']))
    if any(set(g) != set(by_task) for g in groups.values()) or len(groups) != 3:
        raise ValueError('Unmatched baseline support')
    population = sorted(t for t in by_task if passed(groups['Q8_0'][t]))
    negatives = lambda ts, metric: [-metric(by_task[t]) for t in ts]
    labels = [not passed(groups['Q3_K_M'][t]) for t in population]
    primary = permutation(negatives(population, min), labels)
    q4 = permutation(negatives(population, min),
                     [not passed(groups['Q4_K_M'][t]) for t in population])
    first = permutation(negatives(population, lambda x: x[0]), labels)
    mean = permutation(negatives(population, statistics.mean), labels)
    categories = {}
    for category in sorted({groups['Q8_0'][t]['category'] for t in population}):
        ts = [t for t in population if groups['Q8_0'][t]['category'] == category]
        categories[category] = permutation(negatives(ts, min),
                                          [not passed(groups['Q3_K_M'][t]) for t in ts])
    # Ties at the quarter boundary receive fractional inclusion, independent of task IDs.
    margins = sorted(min(by_task[t]) for t in population)
    k = math.ceil(len(population) / 4)
    cutoff = margins[k - 1] if k else None
    below = sum(m < cutoff for m in margins)
    tied = sum(m == cutoff for m in margins)
    captured = sum((1 if min(by_task[t]) < cutoff else (k - below) / tied
                    if min(by_task[t]) == cutoff else 0)
                   for t, label in zip(population, labels, strict=True) if label)
    all_tasks = sorted(by_task)
    confidence = permutation(negatives(all_tasks, min),
                             [not passed(groups['Q8_0'][t]) for t in all_tasks])
    confidence['mean_min_margin_success'] = statistics.mean(
        min(by_task[t]) for t in population) if population else None
    failed = [t for t in all_tasks if t not in population]
    confidence['mean_min_margin_failure'] = statistics.mean(
        min(by_task[t]) for t in failed) if failed else None
    return {'primary': primary, 'mismatch_share': share, 'secondary': {
        'q4': q4, 'first_position': first, 'mean_margin': mean, 'by_category': categories,
        'lowest_quarter': {'tasks': k, 'cutoff': cutoff, 'captured_failures': captured,
                            'share': captured / sum(labels) if sum(labels) else None,
                            'ties': 'fractional inclusion at boundary; quarter size rounded up'},
        'q8_confidence': confidence}}


def render_report(report):
    p = report['primary']
    secondary = report['secondary']
    lines = [
        '# Pre-registered Q8 margin experiment', '',
        (f"Decision: **{p['decision']}**. Q8-success population: {p['n']}; "
        f"Q3 failures: {p['positives']}. AUC (negative minimum margin): {p['auc']}; "
        f"one-sided permutation p: {p['p_value']} (10,000 permutations, seed 0)."), '',
        f"Trajectory call mismatch share: {report['mismatch_share']}.", '',
        'Secondary tables (excluded from the primary decision):', '',
        '| Analysis | Tasks | Positives | AUC | One-sided p | Decision |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for title, key in [('Q4 failure, minimum margin', 'q4'),
                       ('Q3 failure, first generated position', 'first_position'),
                       ('Q3 failure, mean margin', 'mean_margin'),
                       ('Q8 failure, minimum margin', 'q8_confidence')]:
        cell = secondary[key]
        lines.append(f"| {title} | {cell['n']} | {cell['positives']} | {cell['auc']} | "
                     f"{cell['p_value']} | {cell['decision']} |")
    lines += ['', 'Q3 failure by category, among Q8 successes:', '',
              '| Category | Tasks | Positives | AUC | One-sided p | Decision |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for title, cell in secondary['by_category'].items():
        lines.append(f"| {title} | {cell['n']} | {cell['positives']} | {cell['auc']} | "
                     f"{cell['p_value']} | {cell['decision']} |")
    quarter = secondary['lowest_quarter']
    confidence = secondary['q8_confidence']
    lines += ['', (f"Lowest-margin quarter: {quarter['tasks']} tasks; cutoff {quarter['cutoff']}; "
              f"Q3 failures captured {quarter['captured_failures']}, share {quarter['share']}. "
              f"Tie rule: {quarter['ties']}."), '',
              ('Mean task-minimum margin among Q8 successes: '
              f"{confidence['mean_min_margin_success']}; among Q8 failures: "
              f"{confidence['mean_min_margin_failure']}."), '',
              ('AUC uses half credit for ties. Permutation p uses the add-one correction. '
              'Undefined AUC/p values are null in JSON and None in these tables. '
              'Fewer than eight positives is not informative, as pre-registered.'), '',
              ('One run; no alternative margin definitions promoted to the headline. '
              'Any follow-up requires a new task set and separate registration.'), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--base-url', default='http://127.0.0.1:8089')
    args = parser.parse_args()
    if args.measure:
        measure(args.base_url)
        return
    report = analyze(load(OUTPUT), load(BASELINE))
    target = RESULTS / 'margin-analysis'
    target.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    target.with_suffix('.md').write_text(render_report(report))


if __name__ == '__main__':
    main()
