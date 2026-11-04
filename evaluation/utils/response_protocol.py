"""Answer validation and request handling.

One retry rule is shared by every evaluation setting (zero-shot, four-shot,
CoT, reasoning-effort, fine-tuned, and Agents API; Supplementary Material
S.1-E): each sample allows up to five attempts (one initial attempt and up to
four retries). A transient request failure that returns no model response
(e.g. a network error or server timeout) is retried with the same task
prompt. A
response whose final answer fails task-specific format validation is followed
by a repair prompt that specifies only the required answer format, without
ground truth or correctness feedback. The first response that passes
validation is accepted; otherwise the task's failure rules apply. For CoT,
validation applies to the final answer after the reasoning block; a missing
reasoning block alone does not trigger a retry.
"""
from __future__ import annotations

import time
from agents_benchmark.catalog import DATASET_TASKS, Sample
from evaluation.utils.cot_prompting import extract_visible_content, parse_reasoning_response
from evaluation.utils.scoring import parse

MAX_ATTEMPTS = 5
# Statuses for which the provider returned no model response.
NO_RESPONSE_STATUSES = frozenset({'request_failed', 'error', 'failed'})


def task_name(task, context):
    task = {'neon-tree': 'neon', 'dior-rsvg': 'dior_localization',
            'dior-rsvg-recognition': 'dior_classification'}.get(task, task)
    if task == 'geobench-refdet':
        return str(context['source_dataset']).lower()
    return next((name for name, source in DATASET_TASKS.items() if source == task), task)


def parse_answer(task, text, context=None):
    context = context or {}
    name = task_name(task, context)
    public = {key: context[key] for key in ('options', 'prompt', 'image_width', 'image_height') if key in context}
    return parse(Sample(name, '', DATASET_TASKS[name], 0, public), text)


def answer_valid(value):
    return value is not None and (not isinstance(value, dict) or
                                  bool(value) and all(v is not None for v in value.values()))


def format_valid(task, text, context=None):
    return answer_valid(parse_answer(task, text, context))


def sample_format_valid(sample, text):
    """Format validation for an already-built benchmark Sample (Agents API arms)."""
    return answer_valid(parse(sample, text or ''))


def repair_instruction(task, context=None):
    name = task_name(task, context or {})
    formats = {
        'landmarks': "Return exactly one line in the form '<number>. <option>'.",
        'patternnet': 'Return a JSON object with a category field containing one category name.',
        'dior_classification': 'Return a JSON object with a category field containing one category name.',
        'fpcd': 'Return a JSON object with an answer field containing one option letter.',
        'rareplanes': 'Return a JSON object with an Answer field containing a non-negative integer.',
        'cowc': 'Return a JSON object with a count field containing a non-negative integer.',
        'neon': 'Return only a non-negative integer without explanation.',
        'animals': 'Return a JSON object with elephants, zebras, and giraffes fields, each a non-negative integer.',
        'xview2': 'Return a JSON object with count_before, no_damage, minor_damage, major_damage, and destroyed fields, each a non-negative integer.',
        'dior_localization': 'Return a JSON object with a bbox field containing [xmin, ymin, xmax, ymax] in valid pixel coordinates within the image.',
        'dota': 'Return a JSON object with bboxes containing a list of [xmin, ymin, xmax, ymax] pixel boxes within the image.',
        'fair1m': 'Return a JSON object with bboxes containing a list of four-corner polygons [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], in boundary order within the image.',
    }
    return formats[name]


def transient_error(exc):
    last_attempt = getattr(exc, 'last_attempt', None)
    if last_attempt is not None and callable(getattr(last_attempt, 'exception', None)):
        cause = last_attempt.exception()
        if cause is not None and cause is not exc:
            return transient_error(cause)
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    return (isinstance(exc, (ConnectionError, TimeoutError)) or
            type(exc).__name__ in {'RequestFailed', 'APIConnectionError', 'APITimeoutError'} or
            status in {408, 409, 425, 429, 500, 502, 503, 504})


def validated_call(call, reasoning_method, *, task, context=None, prompt='', max_attempts=MAX_ATTEMPTS):
    attempts = min(MAX_ATTEMPTS, max(0, max_attempts))
    repair = False
    response, content = {}, ''
    parsed = {'answer_text': '', 'reasoning': None}
    started = time.monotonic()
    usage = {}
    for attempt in range(attempts):
        current_prompt = prompt + ('\n\n' + repair_instruction(task, context) if repair else '')
        try:
            response = call(current_prompt)
        except Exception as exc:
            if not transient_error(exc):
                raise
            # No model response: retry with the same task prompt.
            response = {'answer_text': '', 'status': 'request_failed', 'error': str(exc)}
            content = ''
            parsed = {'answer_text': '', 'reasoning': None}
            repair = False
            continue
        for key, value in (response.get('token_usage', {}) if isinstance(response, dict) else {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        status = response.get('status') if isinstance(response, dict) else None
        if status in NO_RESPONSE_STATUSES:
            content = ''
            parsed = {'answer_text': '', 'reasoning': None}
            repair = False
            continue
        if status == 'incomplete':
            # A truncated response has no complete final answer: it is never
            # accepted and, like any invalid answer, is followed by a repair.
            content = ''
            parsed = {'answer_text': '', 'reasoning': None}
            repair = True
            continue
        content = extract_visible_content(response)
        parsed = parse_reasoning_response(content, reasoning_method)
        if format_valid(task, parsed['answer_text'], context):
            break
        repair = True
    valid = format_valid(task, parsed['answer_text'], context)
    if isinstance(response, dict):
        response = dict(response)
        response.update(answer_text=parsed['answer_text'], attempt_count=attempt + 1 if attempts else 0,
                        retry_exhausted=not valid,
                        elapsed_seconds=time.monotonic() - started, token_usage=usage)
        response['status'] = 'ok' if valid else response.get('status', 'parse_failed')
        if not valid and response['status'] == 'ok':
            response['status'] = 'parse_failed'
    return response, content, parsed


def query_validated(client, task, prepared, images, *, examples=(), before_attempt=None, max_attempts=MAX_ATTEMPTS):
    def call(prompt):
        if before_attempt is not None:
            before_attempt()
        kwargs = {'examples': examples} if examples else {}
        return client.query(prepared['system_message'], prompt, images,
                            prepared.get('generation_settings', {}), **kwargs)
    response, _, parsed = validated_call(
        call, prepared.get('reasoning_method', 'baseline'), task=task, context=prepared,
        prompt=prepared['prompt'], max_attempts=max_attempts)
    response.update(parsed)
    return response


