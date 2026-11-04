"""Open vlm weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import tempfile


def stage_weights(source, *, stop_requested=lambda: False, workers=4,
                  local_root='/tmp', reserve_bytes=20 * 2**30):
    source = Path(source)
    files = [path for path in source.rglob('*') if path.is_file()]
    required = sum(path.stat().st_size for path in files)
    if not files or shutil.disk_usage(local_root).free < required + reserve_bytes:
        raise OSError('Insufficient node-local disk for the complete official checkpoint')
    temporary = tempfile.TemporaryDirectory(prefix='open-vlm-weights-', dir=local_root)
    destination = Path(temporary.name) / source.name
    destination.mkdir()

    def copy_file(path):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as reader, target.open('wb') as writer:
            while True:
                if stop_requested():
                    raise InterruptedError('Node-local weight staging interrupted')
                block = reader.read(8 * 2**20)
                if not block:
                    break
                writer.write(block)
            if target.suffix == '.safetensors':


                writer.flush()
                os.fsync(writer.fileno())
                if hasattr(os, 'posix_fadvise'):
                    os.posix_fadvise(writer.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if target.stat().st_size != path.stat().st_size:
            raise OSError(f'Incomplete node-local weight file: {path.name}')
        return target.stat().st_size

    try:
        copied = 0
        print(f'Staging {required / 2**30:.1f} GiB to node-local disk', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, size in enumerate(executor.map(copy_file, files), 1):
                copied += size
                if index % 16 == 0 or index == len(files):
                    print(f'Weight staging: {copied / 2**30:.1f}/{required / 2**30:.1f} GiB', flush=True)
        return temporary, destination
    except BaseException:
        temporary.cleanup()
        raise
