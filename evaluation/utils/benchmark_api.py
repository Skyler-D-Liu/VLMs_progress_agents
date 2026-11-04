"""Benchmark api."""
from __future__ import annotations

import http.client
import json
import socket
import time
from pathlib import Path

from benchmark_evaluation import AdaptiveLimiter, ROOT
from benchmark_environment import _read_simple_dotenv
from reasoning_effort import ProviderUnavailable, RequestFailed


RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def credentials(model, path):
    import os
    values = _read_simple_dotenv(Path(path))
    if model.startswith('gemini-'):
        host, name = 'generativelanguage.googleapis.com', 'GEMINI_API_KEY'
    elif model.startswith(('gpt-', 'o1', 'o3', 'o4')):
        host, name = 'api.openai.com', 'OPENAI_API_KEY'
    else:
        raise ValueError('Unsupported official API model')
    key = os.environ.get(name) or values.get(name)
    if not key:
        raise ProviderUnavailable(f'Set {name}')
    return host, key


def request_payload(model, system, prompt, images, generation):
    if model.startswith(("gpt-", "o1-", "o3-")):
        payload = {"model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                *({"type": "image_url", "image_url": {"url": uri}} for uri in images),
                {"type": "text", "text": prompt}]},
        ], "stream": True, "stream_options": {"include_usage": True}}
        payload.update({key: value for key, value in generation.items()
                        if key in {"temperature", "top_p", "max_completion_tokens", "reasoning_effort"}})
        return payload
    parts = []
    for uri in images:
        header, encoded = uri.split(",", 1)
        if not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("Expected an inline image")
        parts.append({"inlineData": {"mimeType": header[5:-7], "data": encoded}})
    parts.append({"text": prompt})
    payload = {"contents": [{"role": "user", "parts": parts}]}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if generation:
        payload["generationConfig"] = generation
    return payload


def response_fields(model, data):
    if model.startswith(("gpt-", "o1-", "o3-")):
        actual = data.get("model")
        choices = data.get("choices") or []
        if not choices:
            raise RequestFailed("Successful response has no completion choice")
        value = choices[0].get("message", {}).get("content") or ""
        if isinstance(value, list):
            value = "".join(part.get("text", "") for part in value if part.get("type") == "text")
        finish = choices[0].get("finish_reason")
        usage = data.get("usage") or {}
        complete = finish == "stop"
        token_usage = {"input_tokens": int(usage.get("prompt_tokens") or 0),
                       "output_tokens": int(usage.get("completion_tokens") or 0),
                       "total_tokens": int(usage.get("total_tokens") or 0)}
    else:
        actual = data.get("modelVersion") or data.get("model")
        candidate = (data.get("candidates") or [{}])[0]
        value = "".join(str(part.get("text", "")) for part in candidate.get("content", {}).get("parts", [])
                        if not part.get("thought", False))
        finish = candidate.get("finishReason")
        usage = data.get("usageMetadata") or {}
        complete = finish == "STOP"
        token_usage = {"input_tokens": int(usage.get("promptTokenCount") or 0),
                       "output_tokens": int(usage.get("candidatesTokenCount") or 0),
                       "total_tokens": int(usage.get("totalTokenCount") or 0)}
    if actual != model:
        raise ProviderUnavailable(f"Model mismatch for {model}: received {actual!r}")
    return {"answer_text": str(value).strip() if complete else "", "output_text": str(value),
            "finish_reason": finish, "status": "ok" if complete else "incomplete",
            "refused": complete and not bool(str(value).strip()), "token_usage": token_usage}


class RemoteClient:
    def __init__(self, model, env_path=None, maximum=8, timeout=600, attempts=3, transport=None, sleep=time.sleep, *, ramp_up=False):
        self.model = model
        self.host, self.key = credentials(model, env_path or ROOT / ".env")
        self.limiter = (AdaptiveLimiter(maximum, initial=4, recovery_successes=16, backoff_errors=True)
                        if ramp_up else AdaptiveLimiter(maximum))
        self.timeout, self.attempts = timeout, attempts
        self.transport = transport or self._send
        self.sleep = sleep

    def _send(self, payload):
        from vleo_runtime.api import openai_request, gemini_request, json_response
        if self.model.startswith('gemini-'):
            return json_response(lambda: gemini_request(self.model, payload, self.key, self.timeout))
        return json_response(lambda: openai_request(payload, self.key, self.timeout))

    def query(self, system, prompt, images, generation=None):
        payload = request_payload(self.model, system, prompt, images, generation or {})
        start = time.monotonic()
        last = "transport error"
        for attempt in range(self.attempts):
            try:
                with self.limiter:
                    status, body = self.transport(payload)
                self.limiter.report(status)
                if status == 200:
                    result = response_fields(self.model, json.loads(body))
                    result["elapsed_seconds"] = time.monotonic() - start
                    result["attempt_count"] = attempt + 1
                    return result
                last = f"HTTP {status}"
                if status not in RETRYABLE:
                    raise ProviderUnavailable(f"{self.model}: {last}")
                lower = body.lower()
                if any(term in lower for term in (b"model_not_found", b"insufficient_quota")):
                    raise ProviderUnavailable(f"{self.model}: {last}")
            except (OSError, socket.timeout, http.client.HTTPException, ValueError, RequestFailed) as error:
                last = type(error).__name__
                self.limiter.report(None)
            if attempt + 1 < self.attempts:
                self.sleep(min(30, 2 ** (attempt + 1)))
        raise RequestFailed(f"{self.model}: {last}; retries exhausted")
