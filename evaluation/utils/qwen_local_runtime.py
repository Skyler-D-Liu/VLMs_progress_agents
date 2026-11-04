"""Qwen local runtime."""
from __future__ import annotations
import base64
import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
BASE_MODEL_NAME = 'Qwen3-VL-32B-Instruct'
BASE_MODEL_RELEASE_DATE = '2025-10-21'
LORA_MODEL_PREFIX = f'{BASE_MODEL_NAME}-LoRA-'
TASK_ADAPTER_DIRS = {'dior-rsvg-recognition': 'dior-rsvg-recognition', 'geobench-refdet': 'geobench-refdet', 'aerial-animal': 'aerial-animal', 'cowc': 'cowc', 'dior-rsvg': 'dior-rsvg', 'fpcd': 'fpcd', 'landmarks': 'landmarks', 'neon-tree': 'neon-tree', 'pattern-net': 'pattern-net', 'rare-planes': 'rare-planes', 'xview2': 'xview2'}
_TASK_ALIASES = {'dior-rsvg-recognition': ('dior-rsvg-recognition',), 'geobench-refdet': ('geobench-refdet',), 'aerial-animal': ('aerial-animal-population-4tu', 'aerial-animal'), 'cowc': ('cowc-m', 'cowc'), 'dior-rsvg': ('DIOR-RSVG', 'dior-rsvg'), 'fpcd': ('FPCD', 'farm-pond-change-detection', 'fpcd'), 'landmarks': ('landmarks',), 'neon-tree': ('neon-trees', 'neon-tree'), 'pattern-net': ('pattern-net', 'PatternNet'), 'rare-planes': ('RarePlanes', 'Rare Planes', 'rare_planes', 'rare-planes', 'aircraft-counting'), 'xview2': ('xView2', 'xview2')}

def canonical_lora_model_name(task: str) -> str:
    _require_known_task(task)
    return f'{LORA_MODEL_PREFIX}{task}'

def _require_known_task(task: str) -> None:
    if task not in TASK_ADAPTER_DIRS:
        raise ValueError(f'Unknown Qwen LoRA task: {task}')

def _selection_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for task, aliases in _TASK_ALIASES.items():
        names = (*aliases, canonical_lora_model_name(task))
        for name in names:
            lookup[name.casefold()] = task
    return lookup
_LORA_SELECTIONS = _selection_lookup()
_QWEN_MODEL_CACHE: Dict[tuple, Dict[str, Any]] = {}
_LEGACY_QWEN_VL_RUNTIME = 'qwen-vl-chat-legacy'
_LEGACY_GENERATION_COMPAT_CLASSES: Dict[type, type] = {}
_RESOLUTION_MAX_PIXELS = {'processor-default': 16777216, 'visual-tokens-4096': 4194304, 'visual-tokens-2304': 2359296, 'visual-tokens-1280': 1310720}

def resolve_lora_task(model_or_alias: str) -> str:
    task = _LORA_SELECTIONS.get(model_or_alias.strip().casefold())
    if task is None:
        supported = ', '.join(sorted(TASK_ADAPTER_DIRS))
        raise ValueError(f'Unknown Qwen LoRA model or adapter alias: {model_or_alias!r}. Supported tasks: {supported}')
    return task

def normalize_lora_model_selection(model_arg: Optional[str], expected_task: str) -> Optional[str]:
    _require_known_task(expected_task)
    if model_arg is None:
        return None
    try:
        task = resolve_lora_task(model_arg)
    except ValueError:
        return model_arg
    if task != expected_task:
        raise ValueError(f'LoRA task mismatch: this evaluation script expects {expected_task!r}, but --model selected the {task!r} adapter. Use {canonical_lora_model_name(expected_task)!r} for this script.')
    return canonical_lora_model_name(task)

def build_lora_model_configs(task: str) -> Dict[str, Dict[str, str]]:
    _require_known_task(task)
    model_key = canonical_lora_model_name(task)
    return {model_key: {'api_type': 'hf_local', 'model_name': model_key, 'base_model_name': BASE_MODEL_NAME, 'release_date': BASE_MODEL_RELEASE_DATE, 'family': 'Qwen', 'model_kind': 'lora', 'adapter_name': task, 'adapter_dir': TASK_ADAPTER_DIRS[task]}}

def is_lora_config(config: Mapping[str, Any]) -> bool:
    return config.get('model_kind') == 'lora'

@dataclass(frozen=True)
class ResolvedQwenModel:
    model_key: str
    base_path: Path
    adapter_path: Optional[Path] = None
    adapter_name: Optional[str] = None

    @property
    def is_lora(self) -> bool:
        return self.adapter_path is not None

    def __str__(self) -> str:
        if self.adapter_path is None:
            return str(self.base_path)
        return f'base={self.base_path}, adapter={self.adapter_path}'

def _configured_root(explicit: Optional[str], *environment_names: str) -> Optional[Path]:
    value = explicit
    if not value:
        for name in environment_names:
            value = os.environ.get(name)
            if value:
                break
    return Path(value).expanduser() if value else None

def _resolve_checkpoint(root: Path, model_name: str, *, single_model: bool) -> Path:
    if not root.exists():
        raise FileNotFoundError(f'Qwen model path does not exist: {root}')
    candidate = root / model_name
    if candidate.is_dir():
        return candidate
    if single_model and root.is_dir():
        return root
    raise FileNotFoundError(f'Unable to resolve local checkpoint for {model_name!r}. Expected directory: {candidate}')

def _resolve_lora_base_checkpoint(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f'Qwen model path does not exist: {root}')
    candidate = root / BASE_MODEL_NAME
    if candidate.is_dir():
        return candidate
    if root.is_dir() and root.name.casefold() == BASE_MODEL_NAME.casefold():
        return root
    raise FileNotFoundError(f'LoRA evaluation requires the {BASE_MODEL_NAME!r} base checkpoint. Expected {candidate}, or pass that checkpoint directory directly with --qwen-model-path.')

def _adapter_base_name(config: Mapping[str, Any]) -> str:
    raw_name = str(config.get('base_model_name_or_path') or '').rstrip('/\\')
    return raw_name.replace('\\', '/').rsplit('/', 1)[-1]

def validate_adapter_directory(adapter_path: Path) -> None:
    if not adapter_path.is_dir():
        raise FileNotFoundError(f'LoRA adapter directory does not exist: {adapter_path}. Final adapters must be saved at QWEN_FINETUNED_ROOT/<task>, not left only under checkpoint-* directories.')
    config_path = adapter_path / 'adapter_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing PEFT adapter config: {config_path}')
    weight_paths = (adapter_path / 'adapter_model.safetensors', adapter_path / 'adapter_model.bin')
    if not any((path.is_file() for path in weight_paths)):
        expected = ' or '.join((str(path) for path in weight_paths))
        raise FileNotFoundError(f'Missing PEFT adapter weights; expected {expected}')
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read valid JSON from {config_path}: {exc}') from exc
    configured_base = _adapter_base_name(config)
    if configured_base.casefold() != BASE_MODEL_NAME.casefold():
        shown = configured_base or '<missing base_model_name_or_path>'
        raise ValueError(f'Adapter {adapter_path} targets {shown!r}, but evaluation requires {BASE_MODEL_NAME!r}. Refusing to load a mismatched adapter (for example an old 8B adapter).')
    peft_type = str(config.get('peft_type') or '').upper()
    if peft_type != 'LORA':
        shown = peft_type or '<missing peft_type>'
        raise ValueError(f"Adapter {adapter_path} declares peft_type={shown!r}; expected 'LORA'.")

def load_lora_inference_config(adapter_path: Path) -> Dict[str, Any]:
    path = adapter_path / 'inference_config.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing promoted-adapter inference contract: {path}. Final evaluation requires an atomically promoted adapter.')
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read valid JSON from {path}: {exc}') from exc
    expected = {'base_model': BASE_MODEL_NAME, 'few_shot_k': 0}
    for field, wanted in expected.items():
        if config.get(field) != wanted:
            raise ValueError(f'Promoted adapter inference contract has {field}={config.get(field)!r}; expected {wanted!r}: {path}')
    resolution = config.get('resolution_candidate')
    if resolution not in _RESOLUTION_MAX_PIXELS:
        raise ValueError(f'Unknown promoted-adapter resolution_candidate={resolution!r}: {path}')
    generation = config.get('generation')
    if not isinstance(generation, dict) or type(generation.get('max_new_tokens')) is not int or generation['max_new_tokens'] not in {128, 2048} or (generation.get('do_sample') is not False) or (set(generation) != {'max_new_tokens', 'do_sample'}):
        raise ValueError('Promoted adapter generation contract requires max_new_tokens=128 or 2048 and do_sample=false')
    prompt_sha256 = config.get('prompt_sha256')
    if prompt_sha256 is not None and (not isinstance(prompt_sha256, str) or len(prompt_sha256) != 64):
        raise ValueError(f'Invalid prompt_sha256 in {path}')
    return config

def apply_lora_inference_resolution(processor: Any, inference_config: Mapping[str, Any]) -> None:
    resolution = str(inference_config['resolution_candidate'])
    max_pixels = _RESOLUTION_MAX_PIXELS[resolution]
    image_processor = getattr(processor, 'image_processor', None)
    if image_processor is None:
        raise ValueError('The Qwen processor has no image_processor')
    current_size = dict(getattr(image_processor, 'size', {}) or {})
    min_pixels = int(current_size.get('shortest_edge', 65536))
    if min_pixels > max_pixels:
        raise ValueError(f'Processor min_pixels={min_pixels} exceeds promoted max_pixels={max_pixels}')
    image_processor.size = {'shortest_edge': min_pixels, 'longest_edge': max_pixels}

def resolve_qwen_model_specs(model_keys: Iterable[str], model_configs: Mapping[str, Mapping[str, Any]], qwen_model_path: Optional[str]=None, qwen_finetuned_root: Optional[str]=None) -> Dict[str, ResolvedQwenModel]:
    local_keys = [key for key in model_keys if model_configs[key].get('api_type') == 'hf_local' and model_configs[key].get('family') == 'Qwen']
    if not local_keys:
        return {}
    base_root = _configured_root(qwen_model_path, 'QWEN_MODEL_PATH', 'QWEN_MODEL_ROOT')
    if base_root is None:
        names = ', '.join(local_keys)
        raise ValueError(f'Selected local Qwen model(s): {names}. Provide --qwen-model-path or set QWEN_MODEL_PATH in .env.')
    lora_keys = [key for key in local_keys if is_lora_config(model_configs[key])]
    adapter_root = None
    if lora_keys:
        adapter_root = _configured_root(qwen_finetuned_root, 'QWEN_FINETUNED_ROOT')
        if adapter_root is None:
            raise ValueError('A Qwen LoRA model was selected. Provide --qwen-finetuned-root or set QWEN_FINETUNED_ROOT in .env.')
        if not adapter_root.is_dir():
            raise FileNotFoundError(f'Qwen LoRA root does not exist: {adapter_root}')
    resolved: Dict[str, ResolvedQwenModel] = {}
    single_model = len(local_keys) == 1
    for model_key in local_keys:
        config = model_configs[model_key]
        if is_lora_config(config):
            base_path = _resolve_lora_base_checkpoint(base_root)
            adapter_name = str(config['adapter_name'])
            adapter_path = adapter_root / str(config['adapter_dir'])
            validate_adapter_directory(adapter_path)
            resolved[model_key] = ResolvedQwenModel(model_key=model_key, base_path=base_path, adapter_path=adapter_path, adapter_name=adapter_name)
        else:
            model_name = str(config.get('model_name') or model_key)
            base_path = _resolve_checkpoint(base_root, model_name, single_model=single_model)
            resolved[model_key] = ResolvedQwenModel(model_key=model_key, base_path=base_path)
    return resolved

def _peft_model_class():
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError('Qwen LoRA inference requires PEFT. Install the project requirements or run `python -m pip install peft`.') from exc
    return PeftModel

def attach_lora_adapter(model: Any, runtime: ResolvedQwenModel) -> Any:
    if not runtime.is_lora:
        return model
    return _peft_model_class().from_pretrained(model, str(runtime.adapter_path), is_trainable=False, local_files_only=True)

def _iter_qwen_model_classes(transformers_module: Any, checkpoint_path: Path, model_key: str):
    config = _checkpoint_config(checkpoint_path)
    architecture_names = [str(name) for name in config.get('architectures', [])]
    model_type_classes = {'qwen2_vl': 'Qwen2VLForConditionalGeneration', 'qwen2_5_vl': 'Qwen2_5_VLForConditionalGeneration', 'qwen3_vl': 'Qwen3VLForConditionalGeneration'}
    model_type_class = model_type_classes.get(str(config.get('model_type', '')))
    if model_type_class:
        architecture_names.append(model_type_class)
    if not architecture_names:
        normalized_key = model_key.casefold()
        if 'qwen2.5-vl' in normalized_key:
            architecture_names.append('Qwen2_5_VLForConditionalGeneration')
        elif 'qwen2-vl' in normalized_key:
            architecture_names.append('Qwen2VLForConditionalGeneration')
        elif 'qwen3-vl' in normalized_key:
            architecture_names.append('Qwen3VLForConditionalGeneration')
    class_names = (*architecture_names, 'AutoModelForImageTextToText', 'AutoModelForVision2Seq')
    seen = set()
    for class_name in class_names:
        if class_name in seen:
            continue
        seen.add(class_name)
        model_class = getattr(transformers_module, class_name, None)
        if model_class is not None:
            yield (class_name, model_class)

def _checkpoint_config(checkpoint_path: Path) -> Dict[str, Any]:
    config_path = checkpoint_path / 'config.json'
    try:
        return json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}

def _validate_loaded_model_architecture(model: Any, checkpoint_path: Path, model_key: str) -> None:
    checkpoint_config = _checkpoint_config(checkpoint_path)
    expected_model_type = str(checkpoint_config.get('model_type') or '').strip()
    loaded_config = getattr(model, 'config', None)
    loaded_model_type = str(getattr(loaded_config, 'model_type', '') or '').strip()
    if expected_model_type and loaded_model_type and (loaded_model_type != expected_model_type):
        raise RuntimeError(f'Refusing to run {model_key}: checkpoint model_type is {expected_model_type!r}, but Transformers instantiated {loaded_model_type!r} ({model.__class__.__name__}). This would load incompatible or randomly initialized weights.')

def _is_legacy_qwen_vl_checkpoint(model_key: str, checkpoint_path: Path) -> bool:
    config = _checkpoint_config(checkpoint_path)
    architectures = {str(name) for name in config.get('architectures', [])}
    auto_map = config.get('auto_map') or {}
    causal_lm_target = str(auto_map.get('AutoModelForCausalLM') or '')
    return model_key.casefold() == 'qwen-vl-chat' or 'QWenLMHeadModel' in architectures or causal_lm_target.endswith('.QWenLMHeadModel')

def is_legacy_qwen_vl_bundle(bundle: Mapping[str, Any]) -> bool:
    return bundle.get('runtime_kind') == _LEGACY_QWEN_VL_RUNTIME

def _transformers_major_version(transformers_module: Any) -> int:
    version = str(getattr(transformers_module, '__version__', '0'))
    try:
        return int(version.split('.', 1)[0])
    except ValueError:
        return 0

def _load_model_with_attempts(model_classes: Iterable[tuple[str, Any]], checkpoint_path: str) -> tuple[Any, list[str]]:
    max_memory = None
    raw_max_memory = os.environ.get('QWEN_MAX_MEMORY', '').strip()
    if raw_max_memory:
        parsed: dict[int | str, str] = {}
        for item in raw_max_memory.split(','):
            key, separator, value = item.partition(':')
            if not separator or not key.strip() or (not value.strip()):
                raise ValueError('QWEN_MAX_MEMORY must be comma-separated entries such as 0:115GiB,cpu:256GiB')
            key = key.strip()
            normalized_key: int | str = int(key) if key.isdigit() else key
            parsed[normalized_key] = value.strip()
        max_memory = parsed
    load_attempts = ({'device_map': 'auto', 'torch_dtype': 'auto', 'trust_remote_code': True}, {'device_map': 'auto', 'dtype': 'auto', 'trust_remote_code': True}, {'device_map': 'auto', 'trust_remote_code': True})
    if max_memory is not None:
        load_attempts = tuple(({**kwargs, 'max_memory': max_memory} for kwargs in load_attempts))
    errors = []
    for class_name, model_class in model_classes:
        for kwargs in load_attempts:
            try:
                model = model_class.from_pretrained(checkpoint_path, local_files_only=True, **kwargs)
                return (model, errors)
            except Exception as exc:
                errors.append(f'{class_name}: {exc}')
    return (None, errors)

def _install_legacy_stream_generator_stub() -> None:
    package_name = 'transformers_stream_generator'
    main_name = f'{package_name}.main'
    existing = sys.modules.get(package_name)
    if getattr(existing, '__qwen_legacy_stub__', False):
        return

    def unsupported_streaming(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("Streaming is unavailable for legacy Qwen-VL-Chat on Transformers 5; use the benchmark's non-streaming model.chat() path.")

    class NewGenerationMixin:
        generate = unsupported_streaming
        sample_stream = unsupported_streaming

    class StreamGenerationConfig:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    main_module = types.ModuleType(main_name)
    main_module.NewGenerationMixin = NewGenerationMixin
    main_module.StreamGenerationConfig = StreamGenerationConfig
    package_module = types.ModuleType(package_name)
    package_module.__path__ = []
    package_module.__qwen_legacy_stub__ = True
    package_module.main = main_module
    package_module.init_stream_support = unsupported_streaming
    sys.modules[package_name] = package_module
    sys.modules[main_name] = main_module

def _ensure_legacy_generation_mixin(model: Any, transformers_module: Any) -> Any:
    generation_mixin = getattr(transformers_module, 'GenerationMixin', None)
    if generation_mixin is None:
        try:
            from transformers.generation.utils import GenerationMixin
        except ImportError as exc:
            raise ImportError('Qwen-VL-Chat requires transformers GenerationMixin support.') from exc
        generation_mixin = GenerationMixin
    if isinstance(model, generation_mixin):
        return model
    original_class = model.__class__
    compat_class = _LEGACY_GENERATION_COMPAT_CLASSES.get(original_class)
    if compat_class is None:
        compat_class = type(f'{original_class.__name__}WithGenerationMixin', (original_class, generation_mixin), {'__module__': original_class.__module__})
        _LEGACY_GENERATION_COMPAT_CLASSES[original_class] = compat_class
    try:
        model.__class__ = compat_class
    except TypeError as exc:
        raise RuntimeError('Unable to attach GenerationMixin to the legacy Qwen-VL-Chat remote-code model.') from exc
    return model

def _legacy_supports_default_dynamic_cache(model: Any) -> bool:
    del model
    return False

def _configure_legacy_kv_cache(model: Any) -> None:
    for config_name in ('config', 'generation_config'):
        config = getattr(model, config_name, None)
        if config is not None:
            config.use_cache = True
    generation_config = getattr(model, 'generation_config', None)
    if generation_config is not None:
        generation_config.cache_implementation = None
    model._supports_default_dynamic_cache = types.MethodType(_legacy_supports_default_dynamic_cache, model)

def _legacy_convert_head_mask_to_5d(model: Any, head_mask: Any, num_hidden_layers: int) -> Any:
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if head_mask.dim() != 5:
        raise ValueError(f'head_mask must have dimension 1 or 2, but has dimension {head_mask.dim()}')
    return head_mask.to(dtype=model.dtype)

def _legacy_get_head_mask(model: Any, head_mask: Any, num_hidden_layers: int, is_attention_chunked: bool=False) -> Any:
    if head_mask is None:
        return [None] * num_hidden_layers
    head_mask = model._convert_head_mask_to_5d(head_mask, num_hidden_layers)
    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask

def _install_legacy_modeling_compat(model: Any) -> None:
    transformer = getattr(model, 'transformer', None)
    if transformer is None:
        raise RuntimeError('Legacy Qwen-VL-Chat compatibility expected model.transformer, but it is missing.')
    if not hasattr(transformer, '_convert_head_mask_to_5d'):
        transformer._convert_head_mask_to_5d = types.MethodType(_legacy_convert_head_mask_to_5d, transformer)
    if not hasattr(transformer, 'get_head_mask'):
        transformer.get_head_mask = types.MethodType(_legacy_get_head_mask, transformer)

def _is_meta_tensor(value: Any) -> bool:
    return getattr(getattr(value, 'device', None), 'type', None) == 'meta'

def _materialize_legacy_plain_tensors(model: Any) -> None:
    import math
    import torch
    transformer = getattr(model, 'transformer', None)
    if transformer is None:
        return
    rotary = getattr(transformer, 'rotary_emb', None)
    inv_freq = getattr(rotary, 'inv_freq', None)
    if rotary is not None and _is_meta_tensor(inv_freq):
        embedding_weight = getattr(getattr(transformer, 'wte', None), 'weight', None)
        device = getattr(embedding_weight, 'device', None)
        if device is None or device.type == 'meta':
            raise RuntimeError('Unable to materialize legacy Qwen rotary embeddings on a real device.')
        rotary.inv_freq = 1.0 / rotary.base ** (torch.arange(0, rotary.dim, 2, device=device).float() / rotary.dim)
        rotary._rotary_pos_emb_cache = None
        rotary._seq_len_cached = 0
        rotary._ntk_alpha_cached = 1.0
    for block in getattr(transformer, 'h', ()):
        attention = getattr(block, 'attn', None)
        if attention is None:
            continue
        attention_weight = getattr(getattr(attention, 'c_attn', None), 'weight', None)
        device = getattr(attention_weight, 'device', None)
        if device is None or device.type == 'meta':
            continue
        if _is_meta_tensor(getattr(attention, 'logn_tensor', None)):
            logn_values = [math.log(index, attention.seq_length) if index > attention.seq_length else 1 for index in range(1, 32768)]
            attention.logn_tensor = torch.tensor(logn_values, device=device)[None, :, None, None]
        if _is_meta_tensor(getattr(attention, 'masked_bias', None)):
            attention.masked_bias = torch.tensor(-10000.0, device=device)

def _is_cuda_device(device: Any) -> bool:
    if isinstance(device, int):
        return device >= 0
    normalized = str(device).strip().lower()
    return normalized.isdigit() or normalized.startswith('cuda')

def _model_device(model: Any) -> Any:
    device = getattr(model, 'device', None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return 'cpu'

def load_qwen_local_model(model_key: str, runtime: ResolvedQwenModel) -> Dict[str, Any]:
    if not isinstance(runtime, ResolvedQwenModel):
        raise TypeError(f'Expected ResolvedQwenModel for {model_key}, got {type(runtime).__name__}')
    cache_key = (model_key, runtime)
    if cache_key in _QWEN_MODEL_CACHE:
        return _QWEN_MODEL_CACHE[cache_key]
    if runtime.is_lora:
        _peft_model_class()
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError('Running Qwen local inference requires transformers and torch. Install the project requirements before evaluation.') from exc
    allow_cpu = os.environ.get('QWEN_ALLOW_CPU', '').strip().lower() in {'1', 'true', 'yes'}
    if not torch.cuda.is_available() and (not allow_cpu):
        cuda_detail = f" PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}."
        raise RuntimeError(f"CUDA is unavailable, so {model_key} would silently fall back to CPU and appear to hang on the first sample. This checkpoint requires a working CUDA GPU for practical inference.{cuda_detail} Check the job's GPU allocation with `nvidia-smi`. Set QWEN_ALLOW_CPU=1 only if the extremely slow CPU fallback is intentional.")
    checkpoint_path = str(runtime.base_path)
    legacy_qwen_vl = _is_legacy_qwen_vl_checkpoint(model_key, runtime.base_path)
    if legacy_qwen_vl:
        if runtime.is_lora:
            raise ValueError('The original Qwen-VL-Chat runtime does not support these Qwen3-VL LoRA adapters.')
        if _transformers_major_version(transformers) >= 5:
            raise RuntimeError('Qwen-VL-Chat cannot produce valid answers with Transformers 5. Use the isolated Transformers 4.37 runtime: install requirements/qwen_vl_chat_legacy.txt in a separate environment and run this evaluator there. Modern Qwen models continue to use the main Transformers 5 environment.')
        _install_legacy_stream_generator_stub()
        tokenizer_class = getattr(transformers, 'AutoTokenizer', None)
        model_class = getattr(transformers, 'AutoModelForCausalLM', None)
        if tokenizer_class is None or model_class is None:
            raise ImportError('Qwen-VL-Chat requires transformers.AutoTokenizer and transformers.AutoModelForCausalLM.')
        try:
            tokenizer = tokenizer_class.from_pretrained(checkpoint_path, trust_remote_code=True, local_files_only=True)
        except ImportError as exc:
            raise ImportError('Loading Qwen-VL-Chat requires its legacy tokenizer dependencies. Install the updated project requirements (notably tiktoken).') from exc
        model, errors = _load_model_with_attempts((('AutoModelForCausalLM', model_class),), checkpoint_path)
        processor = tokenizer
    else:
        processor_class = getattr(transformers, 'AutoProcessor', None)
        if processor_class is None:
            raise ImportError('Modern Qwen-VL inference requires transformers.AutoProcessor.')
        processor = processor_class.from_pretrained(checkpoint_path, trust_remote_code=True, local_files_only=True)
        if runtime.is_lora:
            inference_config = load_lora_inference_config(runtime.adapter_path)
            apply_lora_inference_resolution(processor, inference_config)
        model, errors = _load_model_with_attempts(_iter_qwen_model_classes(transformers, runtime.base_path, model_key), checkpoint_path)
    if model is None:
        raise RuntimeError(f'Failed to load local checkpoint for {model_key} at {checkpoint_path}:\n' + '\n'.join(errors))
    _validate_loaded_model_architecture(model, runtime.base_path, model_key)
    if legacy_qwen_vl:
        model = _ensure_legacy_generation_mixin(model, transformers)
        _configure_legacy_kv_cache(model)
        _install_legacy_modeling_compat(model)
        _materialize_legacy_plain_tensors(model)
    model = attach_lora_adapter(model, runtime).eval()
    device_map = getattr(model, 'hf_device_map', None)
    if device_map:
        raw_devices = set(device_map.values())
        devices = {str(device) for device in raw_devices}
        if not any((_is_cuda_device(device) for device in raw_devices)) and (not allow_cpu):
            raise RuntimeError(f'{model_key} loaded without any CUDA layers (device map: {sorted(devices)}). Refusing an accidental CPU-only run; fix CUDA/GPU availability first.')
        print(f"[{model_key}] Loaded on devices: {', '.join(sorted(devices))}")
    else:
        print(f'[{model_key}] Loaded on device: {_model_device(model)}')
    bundle = {'model': model, 'processor': processor}
    if legacy_qwen_vl:
        bundle.update({'runtime_kind': _LEGACY_QWEN_VL_RUNTIME, 'tokenizer': processor})
    _QWEN_MODEL_CACHE[cache_key] = bundle
    return bundle

def _few_shot_image_refs(example: Mapping[str, Any]) -> list[Any]:
    if 'image_urls' in example:
        return list(example['image_urls'])
    if 'image_url' in example:
        return [example['image_url']]
    raise ValueError('Few-shot example is missing image_url/image_urls.')

def _few_shot_prompt_text(example: Mapping[str, Any], default_user: str) -> str:
    return str(example.get('user_prompt') or example.get('prompt') or default_user)

def _data_uri_bytes(image_ref: str) -> tuple[bytes, str]:
    header, separator, encoded = image_ref.partition(',')
    if not separator or ';base64' not in header:
        raise ValueError('Qwen-VL-Chat received an invalid base64 image data URI.')
    mime_type = header[5:].split(';', 1)[0].lower()
    suffix = {'image/jpeg': '.jpg', 'image/jpg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/gif': '.gif', 'image/bmp': '.bmp', 'image/tiff': '.tiff'}.get(mime_type, '.img')
    try:
        return (base64.b64decode(encoded), suffix)
    except (ValueError, TypeError) as exc:
        raise ValueError('Qwen-VL-Chat received invalid base64 image data.') from exc

def _materialize_legacy_image_ref(image_ref: Any, root: Path, index: int) -> str:
    if isinstance(image_ref, str):
        if image_ref.startswith('data:'):
            raw_bytes, suffix = _data_uri_bytes(image_ref)
            image_path = root / f'image-{index}{suffix}'
            image_path.write_bytes(raw_bytes)
            return str(image_path)
        return image_ref
    save = getattr(image_ref, 'save', None)
    if callable(save):
        image_path = root / f'image-{index}.png'
        save(image_path, format='PNG')
        return str(image_path)
    raise TypeError(f'Unsupported image reference type for Qwen-VL-Chat: {type(image_ref)!r}')

def _legacy_query(tokenizer: Any, image_paths: Sequence[str], prompt_text: str) -> str:
    list_format = [{'image': image_path} for image_path in image_paths]
    list_format.append({'text': prompt_text})
    return tokenizer.from_list_format(list_format)

def run_legacy_qwen_vl_chat(bundle: Mapping[str, Any], image_refs: Sequence[Any], system: Optional[str], user: str, few_shot_examples: Optional[Sequence[Mapping[str, Any]]]=None, *, max_new_tokens: int=128, allow_repeated_output: bool=False) -> str:
    if not is_legacy_qwen_vl_bundle(bundle):
        raise ValueError('run_legacy_qwen_vl_chat requires a legacy Qwen-VL bundle.')
    import torch
    model = bundle['model']
    tokenizer = bundle['tokenizer']
    with tempfile.TemporaryDirectory(prefix='qwen-vl-chat-') as temporary:
        temporary_root = Path(temporary)
        next_image_index = 0

        def materialize(refs: Sequence[Any]) -> list[str]:
            nonlocal next_image_index
            paths = []
            for image_ref in refs:
                paths.append(_materialize_legacy_image_ref(image_ref, temporary_root, next_image_index))
                next_image_index += 1
            return paths
        history = []
        for example in few_shot_examples or []:
            example_paths = materialize(_few_shot_image_refs(example))
            example_query = _legacy_query(tokenizer, example_paths, _few_shot_prompt_text(example, user))
            history.append((example_query, str(example.get('answer', ''))))
        query = _legacy_query(tokenizer, materialize(image_refs), user)
        generation_limit = int(max_new_tokens)
        if generation_limit < 1:
            raise ValueError('max_new_tokens must be positive')
        chat_kwargs = {'query': query, 'history': history or None, 'do_sample': False, 'max_new_tokens': generation_limit, 'use_cache': True}
        if system:
            chat_kwargs['system'] = system
        with torch.inference_mode():
            response, _ = model.chat(tokenizer, **chat_kwargs)
    response_text = str(response).strip()
    if not response_text:
        raise RuntimeError('Qwen-VL-Chat returned an empty response. This is not a valid benchmark answer; verify that the isolated Transformers 4.37 runtime is active.')
    if not allow_repeated_output and len(response_text) >= 32:
        most_common_count = max((response_text.count(char) for char in set(response_text)))
        if most_common_count / len(response_text) >= 0.6:
            raise RuntimeError('Qwen-VL-Chat returned a pathologically repetitive response, which indicates an incompatible or corrupted runtime rather than a normal model answer. Verify the isolated Transformers 4.37 environment.')
    return response_text

def local_response_metadata(runtime: ResolvedQwenModel) -> Dict[str, str]:
    metadata = {'local_model_path': str(runtime.base_path)}
    try:
        import transformers
        metadata['transformers_version'] = str(transformers.__version__)
    except (ImportError, AttributeError):
        pass
    if runtime.is_lora:
        metadata['adapter_path'] = str(runtime.adapter_path)
        metadata['adapter_name'] = str(runtime.adapter_name)
    return metadata

def result_model_metadata(model_key: str, model_configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    metadata = {'model': model_key}
    config = model_configs[model_key]
    if is_lora_config(config):
        metadata.update({'base_model': str(config['base_model_name']), 'model_kind': 'lora', 'adapter_name': str(config['adapter_name'])})
    return metadata

def format_qwen_model_listing(model_key: str, config: Mapping[str, Any], qwen_finetuned_root: Optional[str]=None) -> str:
    listing = f"  {model_key} (released: {config['release_date']})"
    if not is_lora_config(config):
        return listing
    root = qwen_finetuned_root or os.environ.get('QWEN_FINETUNED_ROOT')
    adapter_dir = str(config['adapter_dir'])
    adapter_path = str(Path(root).expanduser() / adapter_dir) if root else f'$QWEN_FINETUNED_ROOT/{adapter_dir}'
    return f'{listing} [LoRA adapter: {adapter_path}]'
