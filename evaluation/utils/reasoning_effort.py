"""Reasoning effort."""
from __future__ import annotations
import http.client
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
GPT_MODEL = 'gpt-6.1-sol'
GEMINI_MODEL = 'gemini-3.8-flash'
RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
API_EFFORTS = {GPT_MODEL: ('low', 'medium', 'high', 'xhigh', 'max'), GEMINI_MODEL: ('low', 'medium', 'high')}
DEFAULT_ONLY_MODELS = ()
API_MODELS = (*API_EFFORTS, *DEFAULT_ONLY_MODELS)
CHAT_COMPLETIONS_MODELS = (GPT_MODEL,)
STREAMING_MODELS = CHAT_COMPLETIONS_MODELS
ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / '.env'
OUTPUT_ROOT = ROOT / 'results/inference/reasoning_effort'

class ProviderUnavailable(RuntimeError):
    pass

class RequestFailed(RuntimeError):
    pass

@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    timeout: float = 600
    max_attempts: int = 5  # S.1-E: one initial attempt + up to four retries

    def __post_init__(self):
        allowed = (*API_EFFORTS[self.model], 'default') if self.model in API_EFFORTS else ('default',) if self.model in DEFAULT_ONLY_MODELS else ()
        if self.effort not in allowed:
            raise ValueError(f'{self.model} requires an explicit effort in {allowed}')
        if self.timeout <= 0 or self.max_attempts < 1:
            raise ValueError('timeout and max_attempts must be positive')

def register_reasoning_models(registry, *, explicit=False):
    import sys
    if not explicit and (not any((arg in API_MODELS or arg in tuple(('--model=' + name for name in API_MODELS)) for arg in sys.argv[1:]))):
        return
    registry.update({GPT_MODEL: {'model_name': GPT_MODEL, 'family': 'GPT', 'api_type': 'openai'}, GEMINI_MODEL: {'model_name': GEMINI_MODEL, 'family': 'Gemini', 'api_type': 'google_native'}})

def run_reasoning_task_cli(task, argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    model = next((a.split('=', 1)[1] for a in args if a.startswith('--model=')), None)
    if '--model' in args:
        pos = args.index('--model')
        model = args[pos + 1] if pos + 1 < len(args) else None
    if model not in API_MODELS and '--reasoning-effort' not in args:
        return False
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from evaluation.experiments.reasoning_comparison import main
    main(['--task', task, *args])
    return True

def credentials(model, path=ENV_PATH):
    from benchmark_api import credentials as official_credentials
    return official_credentials(model, path)

def request_payload(settings, system, prompt, image_uris, generation=None, examples=()):
    if settings.model in CHAT_COMPLETIONS_MODELS:
        payload = {'model': settings.model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': [*({'type': 'image_url', 'image_url': {'url': uri}} for uri in image_uris), {'type': 'text', 'text': prompt}]}], 'max_completion_tokens': 65536, 'stream': True, 'stream_options': {'include_usage': True}}
        if settings.effort != 'default':
            payload['reasoning_effort'] = settings.effort
        demonstrations = []
        for example in examples:
            demonstrations.extend([{'role': 'user', 'content': [*({'type': 'image_url', 'image_url': {'url': uri}} for uri in example['images']), {'type': 'text', 'text': example['prompt']}]}, {'role': 'assistant', 'content': example['answer']}])
        payload['messages'][1:1] = demonstrations
        return payload
    if settings.model != GEMINI_MODEL:
        raise ValueError('API payload requested for a local model')
    parts = []
    for uri in image_uris:
        header, data = uri.split(',', 1)
        if not header.startswith('data:image/') or not header.endswith(';base64'):
            raise ValueError('Expected an inline base64 image')
        parts.append({'inlineData': {'mimeType': header[5:-7], 'data': data}})
    parts.append({'text': prompt})
    payload = {'systemInstruction': {'parts': [{'text': system}]}, 'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': {**(generation or {}), 'maxOutputTokens': 65536}}
    for control in ('thinkingConfig', 'thinkingLevel', 'thinkingBudget', 'reasoning_effort'):
        payload['generationConfig'].pop(control, None)
    if settings.effort != 'default':
        payload['generationConfig']['thinkingConfig'] = {'thinkingLevel': settings.effort.upper()}
    demonstrations = []
    for example in examples:
        example_parts = []
        for uri in example['images']:
            header, data = uri.split(',', 1)
            if not header.startswith('data:image/') or not header.endswith(';base64'):
                raise ValueError('Expected an inline base64 image')
            example_parts.append({'inlineData': {'mimeType': header[5:-7], 'data': data}})
        example_parts.append({'text': example['prompt']})
        demonstrations.extend([{'role': 'user', 'parts': example_parts}, {'role': 'model', 'parts': [{'text': example['answer']}]}])
    payload['contents'][0:0] = demonstrations
    return payload

def numeric_usage(value):
    value = value or {}
    details = value.get('completion_tokens_details') or value.get('output_tokens_details') or {}
    return {'input_tokens': int(value.get('prompt_tokens', value.get('promptTokenCount', value.get('input_tokens', 0))) or 0), 'output_tokens': int(value.get('completion_tokens', value.get('candidatesTokenCount', value.get('output_tokens', 0))) or 0), 'reasoning_tokens': int(details.get('reasoning_tokens', value.get('thoughtsTokenCount', 0)) or 0), 'total_tokens': int(value.get('total_tokens', value.get('totalTokenCount', 0)) or 0)}

def model_matches(expected, returned):
    return returned == expected

def normalize_response(model, data):
    if model in STREAMING_MODELS:
        returned = data.get('model')
        if not model_matches(model, returned):
            raise ProviderUnavailable(f'Model mismatch: requested {model}; received {returned!r}')
        choices = data.get('choices') or []
        if not choices:
            raise ProviderUnavailable('Response has no completion choices')
        choice = choices[0]
        content = choice.get('message', {}).get('content') or ''
        if isinstance(content, list):
            content = ''.join((p.get('text', '') for p in content if p.get('type') == 'text'))
        finish = choice.get('finish_reason')
        usage = numeric_usage(data.get('usage'))
        complete = finish == 'stop' and bool(str(content).strip())
    else:
        returned = data.get('modelVersion') or data.get('model')
        if returned != model:
            raise ProviderUnavailable(f'Model mismatch: requested {model}; received {returned!r}')
        candidate = (data.get('candidates') or [{}])[0]
        content = ''.join((str(p.get('text', '')) for p in candidate.get('content', {}).get('parts', []) if not p.get('thought', False)))
        finish = candidate.get('finishReason')
        usage = numeric_usage(data.get('usageMetadata'))
        complete = finish == 'STOP' and bool(content.strip())
    return {'answer_text': str(content).strip() if complete else '', 'output_text': str(content), 'finish_reason': finish, 'actual_model': returned, 'token_usage': usage, 'status': 'ok' if complete else 'incomplete'}

class RemoteClient:

    def __init__(self, settings, env_path=ENV_PATH, transport=None, sleep=time.sleep):
        self.settings = settings
        self.host, self._key = credentials(settings.model, env_path)
        self.transport = transport or self._send
        self.sleep = sleep

    def _send(self, payload):
        from vleo_runtime.api import openai_request, gemini_request, json_response
        if self.settings.model.startswith('gemini-'):
            return json_response(lambda: gemini_request(self.settings.model, payload, self._key, self.settings.timeout))
        return json_response(lambda: openai_request(payload, self._key, self.settings.timeout))

    def query(self, system, prompt, image_uris, generation=None, before_attempt=None, examples=()):
        payload = request_payload(self.settings, system, prompt, image_uris, generation, examples)
        start = time.monotonic()
        last = 'transport error'
        for attempt in range(1):
            if before_attempt is not None:
                before_attempt()
            try:
                attempt_start = time.monotonic()
                status, body = self.transport(payload)
                response_seconds = time.monotonic() - attempt_start
                if status == 200:
                    try:
                        result = normalize_response(self.settings.model, json.loads(body))
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ProviderUnavailable('Malformed successful provider response') from exc
                    return {**result, 'elapsed_seconds': time.monotonic() - start, 'successful_response_seconds': response_seconds if result['status'] == 'ok' else None, 'attempt_count': attempt + 1, 'provider': self.host}
                last = f'HTTP {status}'
                if status not in RETRYABLE_HTTP_STATUSES:
                    raise ProviderUnavailable(f'{self.host}: {last}')
                lowered = body.lower()
                if any((p in lowered for p in (b'insufficient_quota', b'model_not_found'))):
                    raise ProviderUnavailable(f'{self.host}: model unavailable or quota exhausted ({last})')
            except RequestFailed as exc:
                last = str(exc)
            except (OSError, socket.timeout, http.client.HTTPException) as exc:
                last = type(exc).__name__
            if last.endswith('input rejected'):
                break
        raise RequestFailed(f'{self.host}: {last}; bounded retries exhausted')


