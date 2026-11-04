"""Open vlm runtime."""
from __future__ import annotations
import base64
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

class UnsupportedModel(RuntimeError):
    pass

class ValidateGenerationScores:

    def __call__(self, input_ids, scores):
        import torch
        if torch.isnan(scores).any() or torch.isposinf(scores).any() or (not torch.isfinite(scores).any(dim=-1).all()):
            raise FloatingPointError('Generation scores do not define a finite sampling distribution')
        return scores

class AllocationStop:

    def __init__(self, requested):
        self.requested = requested
        self.log_progress = os.environ.get('OPEN_VLM_LOG_TOKEN_PROGRESS') == '1'
        self.last_report = time.monotonic()
        self.initial_length = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.log_progress:
            length = input_ids.shape[-1]
            if self.initial_length is None:
                self.initial_length = length - 1
            if time.monotonic() - self.last_report >= 60:
                print(f'Generation progress: {length - self.initial_length} new tokens', flush=True)
                self.last_report = time.monotonic()
        if self.requested():
            raise InterruptedError('Allocation ending; unfinished sample will resume')
        return False

def final_answer(text, *, truncated=False, expects_thinking=False):
    if truncated:
        return ('', 'incomplete')
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    elif '<think>' in text:
        return ('', 'incomplete')
    match = re.search('<answer>\\s*(.*?)\\s*</answer>', text, re.DOTALL)
    if match:
        text = match.group(1)
    for marker in ('<|im_end|>', '<|endoftext|>', '</s>'):
        text = text.replace(marker, '')
    return (text.strip(), 'ok' if text.strip() else 'incomplete')

def native_generation(directory, spec):
    path = Path(directory) / 'generation_config.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    length = config.get('max_new_tokens') or spec.output_tokens
    return {'max_new_tokens': int(length)}

def images_from_uris(uris):
    from PIL import Image
    return [Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1]))).convert('RGB') for uri in uris]

def weight_directory(spec):
    source = spec.directory
    root = os.environ.get('OPEN_VLM_WEIGHT_CACHE')
    if not root:
        return source
    mirror = Path(root) / spec.name
    if not (mirror / '.ready').is_file():
        return source
    files = [path for path in source.rglob('*') if path.is_file()]
    if not files or any((not (mirror / path.relative_to(source)).is_file() or (mirror / path.relative_to(source)).stat().st_size != path.stat().st_size for path in files)):
        return source
    return mirror

def messages_for(system, prompt, images, examples=()):
    messages = []
    if system:
        messages.append({'role': 'system', 'content': [{'type': 'text', 'text': system}]})
    for example in examples:
        messages.append({'role': 'user', 'content': [*({'type': 'image', 'image': image} for image in example['images']), {'type': 'text', 'text': example['prompt']}]})
        messages.append({'role': 'assistant', 'content': example['answer']})
    messages.append({'role': 'user', 'content': [*({'type': 'image', 'image': image} for image in images), {'type': 'text', 'text': prompt}]})
    return messages

def assert_gpu_only(model, spec=None):
    mapping = getattr(model, 'hf_device_map', {})
    if any((str(v) in ('cpu', 'disk', 'meta') for v in mapping.values())):
        raise UnsupportedModel('Checkpoint requires CPU/disk offload; excluded by the requested policy')
    if getattr(model, 'is_quantized', False):
        raise UnsupportedModel('Quantized checkpoint is outside the requested original-precision comparison')
    if any((p.device.type != 'cuda' for p in model.parameters())):
        raise UnsupportedModel('Some checkpoint parameters are not resident on GPUs')

def cpu_offload_budget(spec):
    budget = int(os.environ.get('QWEN38_CPU_OFFLOAD_GIB', '0'))
    if budget < 0 or (budget and spec.name != 'Qwen3.8-Flash-Next'):
        raise UnsupportedModel('CPU offload is enabled only for Qwen3.8-Flash-Next')
    return budget * 2 ** 30

def assert_cpu_offload(model):
    mapping = getattr(model, 'hf_device_map', {})
    if not mapping:
        raise UnsupportedModel('CPU offload needs a mixed GPU/CPU device map; all-CPU placement is unsupported')
    if any((str(v) in ('disk', 'meta') for v in mapping.values())):
        raise UnsupportedModel('CPU offload requires a device map without disk placement')
    if getattr(model, 'is_quantized', False):
        raise UnsupportedModel('CPU offload must preserve original-precision weights')
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            if parameter.device.type == 'meta' and (not getattr(module, '_hf_hook', None)):
                raise UnsupportedModel('Unmaterialized parameter lacks an offload hook')

class NativeClient:

    def __init__(self, spec, *, stop_requested=lambda: False):
        import torch
        import transformers
        self.spec, self.directory = (spec, weight_directory(spec))
        self.cpu_offload = cpu_offload_budget(spec)
        self.stopping_criteria = transformers.StoppingCriteriaList([AllocationStop(stop_requested)])
        if not torch.cuda.is_available():
            raise RuntimeError('Inference must run on an allocated cluster GPU node')
        required_gpus = 1 if self.cpu_offload else spec.gpus
        if torch.cuda.device_count() < required_gpus:
            raise UnsupportedModel(f'{spec.name} requires {required_gpus} visible GPUs')
        self.weight_temporary = None
        if spec.name == 'Qwen3.8-Flash-Next' and self.directory == spec.directory:
            from open_vlm_weights import stage_weights
            import shutil
            required = sum((p.stat().st_size for p in spec.directory.rglob('*') if p.is_file()))
            if shutil.disk_usage('/tmp').free >= required + 20 * 2 ** 30:
                self.weight_temporary, self.directory = stage_weights(spec.directory, stop_requested=stop_requested)
            else:
                print('Local disk cannot hold the checkpoint; loading original weights directly from shared storage', flush=True)
        print(f'Loading {spec.name} from {self.directory}', flush=True)
        load_start = time.monotonic()
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}; free={free / 2 ** 30:.1f} GiB; total={total / 2 ** 30:.1f} GiB', flush=True)
        self.generation = native_generation(self.directory, spec)
        self.maximum_memory = {i: int(torch.cuda.mem_get_info(i)[0] * 0.9) for i in range(torch.cuda.device_count())}
        self.maximum_memory['cpu'] = self.cpu_offload
        if self.cpu_offload:
            print(f'CPU offload budget: {self.cpu_offload / 2 ** 30:.0f} GiB; original precision retained', flush=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='open-vlm-images-')
        config = json.loads((self.directory / 'config.json').read_text())
        if config.get('quantization_config'):
            raise UnsupportedModel('Official checkpoint is quantized')
        names = config.get('architectures', [])
        model_class = next((getattr(transformers, name) for name in names if hasattr(transformers, name)), None)
        if model_class is None:
            model_class = transformers.AutoModelForCausalLM if spec.backend == 'llava-rl' else transformers.AutoModelForImageTextToText
        kwargs = dict(local_files_only=True, trust_remote_code=True, torch_dtype='auto', device_map='auto', max_memory=self.maximum_memory)
        attention = os.environ.get('OPEN_VLM_ATTENTION_IMPLEMENTATION')
        if attention:
            if attention not in ('eager', 'sdpa'):
                raise ValueError('Unsupported attention implementation')
            kwargs['attn_implementation'] = attention
        if self.cpu_offload:
            from qwen38_cpu_offload import embedding_cpu_device_map
            model_config = transformers.AutoConfig.from_pretrained(str(self.directory), local_files_only=True, trust_remote_code=True)
            kwargs['device_map'] = embedding_cpu_device_map(model_class, model_config, self.maximum_memory)
        if spec.name.startswith('Qwen3.8-'):
            kwargs['disable_mmap'] = self.weight_temporary is None
        if spec.backend == 'llava-rl':
            kwargs['key_mapping'] = {'^visual': 'model.visual', '^model(?!\\.(language_model|visual))': 'model.language_model'}
        self.model = model_class.from_pretrained(str(self.directory), **kwargs)
        if self.cpu_offload:
            from qwen38_cpu_offload import keep_ngram_lookup_on_cpu
            keep_ngram_lookup_on_cpu(self.model)
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.directory), trust_remote_code=True, local_files_only=True)
        if self.cpu_offload:
            assert_cpu_offload(self.model)
            print(f'Model placement: {self.model.hf_device_map}', flush=True)
        else:
            assert_gpu_only(self.model, spec)
        self.model.eval()
        print(f"Model placement: {getattr(self.model, 'hf_device_map', {})}", flush=True)
        print(f'Model load seconds={time.monotonic() - load_start:.1f}', flush=True)

    def query(self, system, prompt, image_uris, generation=None, examples=()):
        import torch
        start = time.monotonic()
        for criterion in self.stopping_criteria:
            if isinstance(criterion, AllocationStop):
                criterion.initial_length = None
                criterion.last_report = start
        images = images_from_uris(image_uris)
        examples = [{**example, 'images': images_from_uris(example['images'])} for example in examples]
        kwargs = dict(self.generation)
        kwargs['stopping_criteria'] = self.stopping_criteria
        if self.spec.backend == 'transformers':
            from transformers import LogitsProcessorList
            kwargs['logits_processor'] = LogitsProcessorList([ValidateGenerationScores()])
        messages = messages_for(system, prompt, images, examples)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        device = 'cuda:0' if self.cpu_offload else next(self.model.parameters()).device
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = getattr(outputs, 'sequences', outputs)
        trimmed = outputs[0][inputs['input_ids'].shape[-1]:]
        text = self.processor.decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return self._response(text, trimmed, inputs['input_ids'].shape[-1], start)

    def _response(self, text, tokens, input_tokens, start):
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        truncated = len(tokens) >= self.generation['max_new_tokens'] and int(tokens[-1]) not in eos
        requires_separator = self.spec.thinking and self.spec.name.startswith('Qwen3')
        answer, status = final_answer(text, truncated=truncated, expects_thinking=requires_separator)
        return {'answer_text': answer, 'output_text': text, 'status': status, 'finish_reason': 'length' if truncated else 'stop', 'elapsed_seconds': time.monotonic() - start, 'token_usage': {'input_tokens': int(input_tokens), 'output_tokens': len(tokens), 'total_tokens': int(input_tokens) + len(tokens)}}
