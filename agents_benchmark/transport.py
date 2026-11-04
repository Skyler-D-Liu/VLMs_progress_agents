"""Transport."""
import base64
import re
import shlex
import zlib

CHUNK_BYTES = 4 * 1024 * 1024


def input_manifest(directory, specs):
    manifest, uploads = [], []
    for number, spec in enumerate(specs, 1):
        name = spec["name"]
        if not re.fullmatch(rf"image_{number}\.(png|jpg|jpeg|webp)", name):
            raise ValueError("Invalid input name")
        data = (directory / "inputs" / name).read_bytes()
        parts = []
        for offset in range(0, len(data), CHUNK_BYTES):
            content = data[offset:offset+CHUNK_BYTES]
            part = f"image_{number}_part_{len(parts)}.bin"
            parts.append({"name":part, "size":len(content)})
            uploads.append({"path":"/workspace/input_transfer/" + part,
                            "source":name, "offset":offset, "size":len(content)})
        manifest.append({"name":name, "size":len(data), "crc":zlib.crc32(data), "parts":parts})
    if not 1 <= len(manifest) <= 2:
        raise ValueError("Expected one or two current-sample images")
    return manifest, uploads


def assembler_code(manifest, root="/workspace"):
    return f'''
from pathlib import Path
import shutil
import time
import zlib
root = Path({str(root)!r})
staging = root / "input_transfer"
destination = root / "input"
staging.mkdir(parents=True, exist_ok=True)
destination.mkdir(parents=True, exist_ok=True)
manifest = {manifest!r}
stage = "receiving"
try:
    parts = [p for image in manifest for p in image["parts"]]
    while not all((staging / p["name"]).is_file() and (staging / p["name"]).stat().st_size == p["size"] for p in parts):
        time.sleep(0.2)
    stage = "integrity"
    images = []
    for image in manifest:
        data = b"".join((staging / p["name"]).read_bytes() for p in image["parts"])
        if len(data) != image["size"] or zlib.crc32(data) != image["crc"]:
            raise ValueError("Input transfer integrity check failed")
        images.append((image["name"], data))
    stage = "cleanup"
    shutil.rmtree(staging)
    stage = "publishing"
    for name, data in images:
        temporary = destination / (name + ".partial")
        temporary.write_bytes(data)
        if temporary.read_bytes() != data:
            raise ValueError("Written input differs from received bytes")
        temporary.replace(destination / name)
except Exception as exc:
    (destination / ("transfer_failed_" + stage + "_" + type(exc).__name__ + ".txt")).write_bytes(b"failed")
    raise
'''


def setup_command(manifest):
    return "python3 -c " + shlex.quote(assembler_code(manifest)) + " </dev/null >/dev/null 2>&1 &"


def encoded_part(directory, upload):
    with (directory / "inputs" / upload["source"]).open("rb") as handle:
        handle.seek(upload["offset"])
        part = handle.read(upload["size"])
    if len(part) != upload["size"]:
        raise ValueError("Local input changed during transfer")
    return base64.b64encode(part).decode("ascii")
