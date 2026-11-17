"""Opt-in harness rules and deterministic fixture-tool execution (no network)."""

import http.client
import json
import re
import time
import urllib.error

from .score import METRICS, matches, score

PLACEHOLDER = re.compile(r'result_of|previous_|placeholder|to_be_|unknown|tbd', re.IGNORECASE)
NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)')


def leaves(value, path=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, f'{path}.{key}' if path else key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, f'{path}[{i}]')
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield path, value


def schema_values(schema):
    """Only enum/default/example annotations confer provenance, not descriptions."""
    if not isinstance(schema, dict):
        return
    for key in ('enum', 'default', 'example', 'examples', 'const'):
        if key in schema:
            yield from (v for _, v in leaves(schema[key]))
    # These mappings contain property names, not schema keywords.
    for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
        for child in schema.get(key, {}).values():
            yield from schema_values(child)
    for key in ('items', 'additionalProperties', 'contains', 'not', 'if', 'then', 'else',
                'propertyNames', 'unevaluatedProperties', 'unevaluatedItems'):
        child = schema.get(key)
        if isinstance(child, list):
            for node in child:
                yield from schema_values(node)
        else:
            yield from schema_values(child)
    for key in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
        for child in schema.get(key, []):
            yield from schema_values(child)


def known(value, sources):
    for source in sources:
        if isinstance(value, str):
            # User messages are prose; tool/schema strings are exact values.
            if isinstance(source, tuple) and value in source[0]:
                return True
            if isinstance(source, str) and value == source:
                return True
        elif isinstance(source, (int, float)) and not isinstance(source, bool) and value == source:
            return True
    return False


def call_matches(call, expected):
    try:
        fn = call['function']
        args = json.loads(fn['arguments'])
        return (fn['name'] == expected['name'] and isinstance(args, dict)
                and args.keys() == expected['args'].keys()
                and all(matches(args[k], matcher) for k, matcher in expected['args'].items()))
    except (KeyError, TypeError, ValueError, RecursionError):
        return False


def evaluate_enforced(task, base_url, seed, timeout, temperature, echo_reasoning,
                      one_call, reject_placeholders, provenance, complete):
    messages = list(task['messages'])
    all_calls, executed, responses, events, texts = [], [], [], [], []
    latency, failure, timed_out, batched, cap = 0.0, None, False, False, False
    sources = []
    for message in messages:
        if message['role'] == 'user':
            content = message.get('content', '')
            sources.append((content,))
            sources.extend(int(m.group()) if not any(c in m.group() for c in '.eE')
                           else float(m.group()) for m in NUMBER.finditer(content))
    completed = False
    transport_failure = None
    for turn in range(1, 9):
        started = time.perf_counter()
        try:
            response, elapsed = complete(base_url, messages, task['tools'], seed, timeout, temperature)
        except (OSError, ValueError, http.client.HTTPException, RecursionError) as exc:
            latency += time.perf_counter() - started
            timed_out = isinstance(exc, TimeoutError) or (
                isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError))
            failure = f'response_error: {type(exc).__name__}: {exc}'
            transport_failure = failure
            break
        responses.append(response)
        latency += elapsed
        try:
            message = response['choices'][0]['message']
            content = message.get('content')
            if content is not None and not isinstance(content, str):
                raise ValueError('content must be text or null')
            batch = message.get('tool_calls')
            batch = [] if batch is None else batch
            if not isinstance(batch, list):
                raise TypeError('tool_calls must be a list')
            if not batch and not (content or '').strip():
                raise ValueError('empty content')
            texts.append(content or '')
            all_calls.extend(batch)
            if any(not isinstance(c, dict) for c in batch):
                failure = 'invalid_tool_call'
                break
            if any(not isinstance(c.get('id'), str) or not c['id']
                   or c.get('type') != 'function' for c in batch):
                failure = ('invalid_tool_call_envelope' if score(task, batch)['schema_valid']
                           else 'invalid_tool_call')
                break
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            failure = f'invalid_response: {exc}'
            break
        if not batch:
            completed = True
            break
        echoed = dict(message)
        if not echo_reasoning:
            echoed.pop('reasoning_content', None)
        messages.append(echoed)
        pending_sources = []
        for i, call in enumerate(batch):
            rules = []
            if one_call and i:
                rules.append(('one_call', None))
            else:
                try:
                    fn = call['function']
                    args = json.loads(fn['arguments'])
                    schema = next(t['function']['parameters'] for t in task['tools']
                                  if t['function']['name'] == fn['name'])
                    if not isinstance(args, dict):
                        raise TypeError('arguments must be an object')
                except (KeyError, TypeError, ValueError, StopIteration, RecursionError):
                    failure = 'invalid_tool_call'
                    break
                allowed = sources + list(schema_values(schema))
                for argument, value in leaves(args):
                    if reject_placeholders and isinstance(value, str) and (
                            ('<' in value and '>' in value) or PLACEHOLDER.search(value)):
                        rules.append(('reject_placeholders', argument))
                    elif provenance and not known(value, allowed):
                        rules.append(('provenance', argument))
            if rules:
                for rule, argument in rules:
                    events.append({'turn': turn, 'rule': rule, 'argument': argument, 'call_index': i})
                if rules[0][0] == 'one_call':
                    result = {'error': 'not executed: one tool call per reply; send it again '
                              'after reading this result'}
                else:
                    result = {'error': '; '.join(f'{arg}: value must come from an earlier result '
                                                'or the user' for _, arg in rules)}
            else:
                if not score(task, [call])['schema_valid']:
                    failure = 'invalid_tool_call'
                    break
                executed.append(call)
                # Historical fixture dispatch is ordinal, irrespective of arguments.
                ordinal = len(executed) - 1
                results = task.get('tool_results', [])
                result = results[ordinal] if ordinal < len(results) else None
            pending_sources.extend(v for _, v in leaves(result))
            if result is not None or rules:
                messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                 'content': json.dumps(result)})
        if failure:
            break
        batched |= task['category'] == 'multi_turn' and len(batch) > 1
        if task['category'] != 'multi_turn':
            completed = True
            break
        if len(executed) > len(task['tool_results']):
            completed = True
            break
        # Same-reply calls cannot cite results they have not yet read.
        sources.extend(pending_sources)
    else:
        cap = True
    scores = score(task, all_calls, '\n'.join(texts))
    if batched:
        scores['args_correct'] = False
        scores['silent_failure'] = scores['schema_valid'] and scores['tool_selection']
    failure = failure or scores['failure_reason'] or ('dependent_calls_batched' if batched else None)
    if failure and failure not in {
        'wrong_tool_selection', 'incorrect_arguments', 'dependent_calls_batched',
        'invalid_tool_call', 'invalid_tool_call_envelope'
    }:
        scores.update({metric: False for metric in METRICS})
    if failure == 'invalid_tool_call_envelope':
        scores['args_correct'] = False
        scores['silent_failure'] = False
    scores['failure_reason'] = failure
    final = score(task, executed, '\n'.join(texts))
    success = final['abstained_correctly'] if task['category'] in ('abstain', 'clarify') else final['args_correct']
    return {'timed_out': timed_out, 'failure_reason': scores['failure_reason'], 'responses': responses,
                'latency_seconds': latency, 'scores': scores, 'enforcement': events, 'executed_calls': executed,
                'success_after_enforcement': bool(success),
                'model_turns': turn, 'turn_cap_hit': cap, 'completed': completed,
                'transport_failure': transport_failure}
