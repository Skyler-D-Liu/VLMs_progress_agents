"""Api."""
import base64
import json


def openai_client(api_key, timeout=180):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                  timeout=timeout, max_retries=0)


def openai_request(payload, api_key, timeout=180):
    body = dict(payload)
    body.pop('stream_options', None)
    body['stream'] = False
    with openai_client(api_key, timeout) as client:
        return client.chat.completions.create(**body).model_dump(mode='json')


def gemini_request(model, payload, api_key, timeout=180):
    from google import genai
    from google.genai import types
    contents = []
    for content in payload.get('contents', []):
        parts = []
        for part in content.get('parts', []):
            if 'text' in part:
                parts.append(types.Part.from_text(text=part['text']))
            elif 'inlineData' in part or 'inline_data' in part:
                inline = part.get('inlineData', part.get('inline_data'))
                parts.append(types.Part.from_bytes(
                    data=base64.b64decode(inline['data']),
                    mime_type=inline.get('mimeType', inline.get('mime_type'))))
            elif 'fileData' in part:
                data = part['fileData']
                parts.append(types.Part.from_uri(file_uri=data['fileUri'], mime_type=data.get('mimeType')))
        contents.append(types.Content(role=content.get('role', 'user'), parts=parts))
    config = dict(payload.get('generationConfig') or {})
    instruction = payload.get('systemInstruction')
    if instruction:
        config['system_instruction'] = '\n'.join(p.get('text', '') for p in instruction.get('parts', []))
    with genai.Client(api_key=api_key, vertexai=False, http_options=types.HttpOptions(
        base_url='https://generativelanguage.googleapis.com', timeout=int(timeout * 1000),
        retry_options=types.HttpRetryOptions(attempts=1))) as client:
        result = client.models.generate_content(model=model, contents=contents,
                                               config=types.GenerateContentConfig(**config))
        return result.model_dump(mode='json', by_alias=True, exclude_none=True)


def json_response(call):
    try:
        return 200, json.dumps(call()).encode('utf-8')
    except Exception as exc:
        import httpx
        from openai import APIConnectionError

        if isinstance(exc, (APIConnectionError, httpx.NetworkError, httpx.TimeoutException)):
            raise ConnectionError('Official API connection failed') from exc
        status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
        if not isinstance(status, int):
            raise
        return status, json.dumps({'error': {'message': str(exc)}}).encode('utf-8')
