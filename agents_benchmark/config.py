"""Config."""
from pathlib import Path
import os
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "agents_benchmark" / ".env"
OUTPUT = ROOT / "results" / "agents" / "autonomous"
DATA_ROOT = Path("./raw_dataset")
RAREPLANES_ROOT = ROOT / "raw_dataset" / "rare_planes"
MODEL = "gpt-6-astra"
COMMON_INSTRUCTIONS = """Complete the provided imagery task independently. You may decide whether to use tools or other Agents' functions except web searching. Do not access external data, reference answers, annotation files. Do not request human assistance.

Do not search the web for these datasets or data."""


def initialize_key_file(path: Path = KEY_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.is_symlink():
            raise ValueError("The key file must not be a symbolic link")
        path.chmod(0o600)
        return
    with os.fdopen(fd, "w") as handle:
        handle.write("OPENAI_API_KEY=在这里填写你的OpenAI官方项目API密钥\n")


def read_key(path: Path = KEY_FILE) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Set an official project key in {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"Key file permissions must be 600: {path}")
    values = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if name.strip() != "OPENAI_API_KEY" or not sep:
            raise ValueError("The isolated .env may contain only OPENAI_API_KEY")
        values.append(value.strip().strip("\"'"))
    if len(values) != 1 or not values[0].startswith("sk-") or len(values[0]) < 20:
        raise ValueError(f"Official API key is not configured in {path}")
    return values[0]


def make_client(path: Path = KEY_FILE):
    from vleo_runtime.api import openai_client
    client = openai_client(read_key(path), timeout=60.0)
    if not hasattr(client.beta, "agents"):
        client.close()
        raise RuntimeError("Install the official OpenAI SDK with beta.agents support")
    return client


def build_input(prompt: str, specs: list[dict], guidance: str | None = None) -> list[dict]:
    if not prompt or not 1 <= len(specs) <= 2:
        raise ValueError("Expected a task and one or two current-sample images")
    locations = []
    for number, spec in enumerate(specs, 1):
        if not re.fullmatch(rf"image_{number}\.(png|jpg|jpeg|webp)", spec["name"]):
            raise ValueError("Invalid input image name")
        locations.append(f"Image {number}: /workspace/input/" + spec["name"])
    content = [{"type":"input_text", "text":prompt},
               {"type":"input_text", "text":"Input files (in image order):\n" + "\n".join(locations)}]
    if guidance is not None:
        if not guidance.strip():
            raise ValueError("Guidance must be non-empty text")
        content.append({"type":"input_text", "text":"Procedure:\n" + guidance})
    return [{"role":"user", "content":content}]


def zero_shot_system_message(system_message: str) -> str:
    """Standard API arm: the unchanged zero-shot task system message (Supplement S.1)."""
    if not isinstance(system_message, str) or not system_message.strip():
        raise ValueError('A task system message is required')
    return system_message.strip()


def agent_instructions(system_message: str) -> str:
    """Agents API arms: task system message plus the agent-level instruction (Supplement S.4)."""
    return zero_shot_system_message(system_message) + '\n\n' + COMMON_INSTRUCTIONS


def build_hosted_request(command: str, system_message: str) -> dict:
    return {
        "agent":{"model":MODEL, "instructions":agent_instructions(system_message), "multi_agent":{"enabled":True}},

        "environment":{"type":"openai_hosted", "network":{"access":"disabled"},
                       "capability_directories":[],
                       "setup_commands":[{"command":command}]},
    }


