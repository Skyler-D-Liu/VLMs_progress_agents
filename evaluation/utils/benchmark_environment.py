"""Benchmark environment."""
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List
EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALUATION_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from vleo_runtime.api import openai_request, gemini_request, json_response
RESULTS_ROOT = REPO_ROOT / "results/inference"
ENV_PATH = REPO_ROOT / ".env"
EXTERNAL_VOLUME = Path.home() / ".cache"
DEFAULT_HF_CACHE_DIR = EXTERNAL_VOLUME / "huggingface"
DEFAULT_TORCH_CACHE_DIR = EXTERNAL_VOLUME / "torch"
HF_CACHE_DIR = DEFAULT_HF_CACHE_DIR
TORCH_CACHE_DIR = DEFAULT_TORCH_CACHE_DIR
PORTABLE_CACHE_ROOT = EVALUATION_ROOT / ".cache"
PORTABLE_HF_CACHE_DIR = PORTABLE_CACHE_ROOT / "huggingface"
PORTABLE_TORCH_CACHE_DIR = PORTABLE_CACHE_ROOT / "torch"
_LOADED = False


def _load_simple_dotenv(path: Path, override: bool = False) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()

        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if not override and key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value

def _read_simple_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values

def require_external_volume(*paths: Path) -> None:
    normalized_paths = [Path(path).expanduser() for path in paths if path is not None]
    uses_external_volume = any(
        path.is_absolute() and path.as_posix().startswith(EXTERNAL_VOLUME.as_posix())
        for path in normalized_paths
    )
    if uses_external_volume and not EXTERNAL_VOLUME.is_dir():
        raise RuntimeError(
            f"Required cache directory is not mounted: {EXTERNAL_VOLUME}"
        )

def configure_cache_environment() -> None:
    cache_root = Path(
        os.environ.get("HF_CACHE_DIR") or os.environ.get("HF_HOME") or DEFAULT_HF_CACHE_DIR
    ).expanduser()
    datasets_cache = Path(
        os.environ.get("HF_DATASETS_CACHE", cache_root / "datasets")
    ).expanduser()
    hub_cache = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            os.environ.get("HUGGINGFACE_HUB_CACHE", cache_root / "hub"),
        )
    ).expanduser()
    torch_cache = Path(
        os.environ.get("TORCH_HOME", DEFAULT_TORCH_CACHE_DIR)
    ).expanduser()

    require_external_volume(cache_root, datasets_cache, hub_cache, torch_cache)
    for path in (cache_root, datasets_cache, hub_cache, torch_cache):
        path.mkdir(parents=True, exist_ok=True)

    global HF_CACHE_DIR, TORCH_CACHE_DIR
    HF_CACHE_DIR = cache_root
    TORCH_CACHE_DIR = torch_cache

    cache_values = {
        "HF_HOME": str(cache_root),
        "HF_DATASETS_CACHE": str(datasets_cache),
        "HF_HUB_CACHE": str(hub_cache),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache),
        "TORCH_HOME": str(torch_cache),
        "HF_CACHE_DIR": str(cache_root),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_XET": "1",
        "HF_ENDPOINT": "https://huggingface.co",
    }
    for key, value in cache_values.items():
        os.environ[key] = value

def _looks_like_unavailable_mac_volume(path_value: str) -> bool:
    if not path_value:
        return False
    path = Path(path_value).expanduser()
    return path.is_absolute() and path.as_posix().startswith("/Volumes/") and not path.exists()

def _apply_portable_cache_defaults() -> None:
    cache_values = {
        "HF_HOME": str(PORTABLE_HF_CACHE_DIR),
        "HF_CACHE_DIR": str(PORTABLE_HF_CACHE_DIR),
        "HF_DATASETS_CACHE": str(PORTABLE_HF_CACHE_DIR / "datasets"),
        "HF_HUB_CACHE": str(PORTABLE_HF_CACHE_DIR / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(PORTABLE_HF_CACHE_DIR / "hub"),
        "TORCH_HOME": str(PORTABLE_TORCH_CACHE_DIR),
    }
    for key, value in cache_values.items():
        os.environ[key] = value

def sanitize_cache_environment() -> None:
    configured_paths = {
        key: os.environ.get(key, "")
        for key in (
            "HF_HOME",
            "HF_CACHE_DIR",
            "HF_DATASETS_CACHE",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TORCH_HOME",
        )
    }
    stale_paths = {
        key: value
        for key, value in configured_paths.items()
        if _looks_like_unavailable_mac_volume(value)
    }
    if not stale_paths:
        return

    _apply_portable_cache_defaults()
    stale_details = ", ".join(f"{key}={value}" for key, value in stale_paths.items())
    print(
        "Configured cache paths are unavailable on this machine; "
        f"falling back to project-local caches under {PORTABLE_CACHE_ROOT} "
        f"({stale_details})"
    )

def load_benchmark_environment() -> None:
    global _LOADED
    if _LOADED:
        return


    override = os.environ.get(
        "VLEO_BENCHMARK_MODELTEST_ENV_OVERRIDE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=override)
    else:
        _load_simple_dotenv(ENV_PATH, override=override)

    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    sanitize_cache_environment()
    configure_cache_environment()
    _LOADED = True


def is_unavailable_model_error(message: str) -> bool:
    normalized = message.casefold()
    return any(
        marker in normalized
        for marker in (
            "model_not_found",
            "model not found",
        )
    )

def is_fatal_provider_error(message: str) -> bool:
    normalized = message.casefold()
    return is_unavailable_model_error(message) or any(
        marker in normalized
        for marker in (
            "local:insufficient_quota",
            "insufficient_quota",
            "insufficient quota",
            "unauthorized_client_error",
            "unauthenticated",
            "invalid api key",
            "incorrect api key",
        )
    )

def model_family(model_key: str, model_configs: Dict[str, Dict[str, str]]) -> str:
    config = model_configs[model_key]
    family = config.get("family")
    if family:
        return family
    name = config.get("model_name", model_key).lower()
    if name.startswith("gemini"):
        return "Gemini"
    return "GPT"

def first_api_key_for_model(model_key: str, model_configs: Dict[str, Dict[str, str]]) -> str:
    keys = api_keys_for_model(model_key, model_configs)
    return keys[0] if keys else ""

def random_api_key_for_model(
    model_key: str,
    model_configs: Dict[str, Dict[str, str]],
    fallback: str = "",
) -> str:
    keys = api_keys_for_model(model_key, model_configs)
    if keys:
        return random.choice(keys)
    return fallback

def api_keys_for_models(model_keys: List[str], model_configs: Dict[str, Dict[str, str]]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for model_key in model_keys:
        for key in api_keys_for_model(model_key, model_configs):
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys

def key_for_model_index(model_key: str, idx: int, model_configs: Dict[str, Dict[str, str]]) -> str:
    keys = api_keys_for_model(model_key, model_configs)
    if not keys:
        raise ValueError(
            f"No API key configured for model family {model_family(model_key, model_configs)}."
        )
    return keys[idx % len(keys)]

def key_index_for_model(model_key: str, idx: int, model_configs: Dict[str, Dict[str, str]]) -> int:
    keys = api_keys_for_model(model_key, model_configs)
    return (idx % len(keys)) + 1 if keys else 0

def safe_key_display(api_key: str) -> str:
    if not api_key:
        return "missing"
    if len(api_key) <= 14:
        return "<redacted>"
    return f"{api_key[:8]}...{api_key[-4:]}"

def print_provider_summary() -> None:
    hosts = provider_hosts()
    keys = provider_keys()
    print(
        "Loaded provider API keys: "
        f"GPT={len(keys['GPT'])} via {hosts['GPT']}, "
        f"Gemini={len(keys['Gemini'])} via {hosts['Gemini']}"
    )

def print_model_key_summary(model_keys: List[str], model_configs: Dict[str, Dict[str, str]]) -> None:
    hosts = provider_hosts()
    keys = provider_keys()
    families = []
    for model_key in model_keys:
        family = model_family(model_key, model_configs)
        if family not in families:
            families.append(family)
    summary = ", ".join(
        f"{family}={len(keys.get(family, []))} via {hosts.get(family, 'unset')}"
        for family in families
    )
    print(f"Available provider API keys: {summary}")


def official_keys(family):
    name = "OPENAI_API_KEY" if family == "GPT" else "GEMINI_API_KEY"
    key = os.environ.get(name) or (os.environ.get("GOOGLE_API_KEY") if family == "Gemini" else None)
    return [key] if key else []


def provider_hosts():
    return {"GPT": "api.openai.com", "Gemini": "generativelanguage.googleapis.com"}


def provider_keys():
    return {family: official_keys(family) for family in provider_hosts()}


def api_host_for_model(model_key, model_configs):
    return provider_hosts()[model_family(model_key, model_configs)]


def api_keys_for_model(model_key, model_configs):
    return official_keys(model_family(model_key, model_configs))


class APIStatusError(IOError):
    """Non-200 provider reply. Keeps the HTTP status so that transient failures
    (e.g. 429 or 5xx) are retried under the shared five-attempt rule."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def post_openai_json(*, model_key, model_configs, payload, api_key, timeout=None):
    import json
    key = api_key or first_api_key_for_model(model_key, model_configs)
    return json_response(lambda: openai_request(json.loads(payload), key, timeout or 180))


def post_gemini_json(*, model_key, model_configs, payload, api_key, timeout=None):
    import json
    key = api_key or first_api_key_for_model(model_key, model_configs)
    model = model_configs.get(model_key, {}).get("model_name", model_key)
    return json_response(lambda: gemini_request(model, json.loads(payload), key, timeout or 180))
