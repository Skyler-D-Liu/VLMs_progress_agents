"""Download open vlm models."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from open_vlm_registry import MODEL_BY_NAME, LLAVA_ROOT, OUTPUT_ROOT
from evaluation.experiments.open_vlm import atomic_json
from runtime_lock import operation_lock
FLASH = 'Qwen3.8-Flash-Next'
RESERVE = 20 * 2 ** 30

def ensure_space(required, destination):
    if shutil.disk_usage(destination.parent).free < required + RESERVE:
        raise RuntimeError('Insufficient model storage')

def download(spec):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint='https://huggingface.co', token=False)
    info = api.model_info(spec.repo, files_metadata=True)
    expected = {s.rfilename: s.size for s in info.siblings if not s.rfilename.endswith(('.bin', '.pt', '.pth')) and (not s.rfilename.startswith('.git'))}
    if not any((name.endswith('.safetensors') for name in expected)):
        expected = {s.rfilename: s.size for s in info.siblings if not s.rfilename.startswith('.git')}
    if any((size is None for size in expected.values())):
        raise RuntimeError('Official repository did not report checkpoint file sizes')
    spec.directory.parent.mkdir(parents=True, exist_ok=True)
    present = sum((min((spec.directory / name).stat().st_size, size) for name, size in expected.items() if (spec.directory / name).is_file()))
    ensure_space(sum(expected.values()) - present, spec.directory)
    snapshot_download(spec.repo, local_dir=spec.directory, allow_patterns=list(expected), endpoint='https://huggingface.co', token=False, max_workers=4)
    bad = [name for name, size in expected.items() if not (spec.directory / name).is_file() or (spec.directory / name).stat().st_size != size]
    if bad:
        raise RuntimeError(f'Incomplete official download: {bad}')
    for index in spec.directory.glob('*.safetensors.index.json'):
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards.issubset(expected):
            raise RuntimeError('Official weight index references absent shards')
    config = json.loads((spec.directory / 'config.json').read_text())
    tower = config.get('mm_vision_tower')
    if tower and '/' in tower and (not tower.startswith('/')):
        encoder_dir = LLAVA_ROOT / 'vision_encoders' / tower.split('/')[-1]
        encoder_info = api.model_info(tower, files_metadata=True)
        encoder_files = {s.rfilename: s.size for s in encoder_info.siblings if not s.rfilename.endswith(('.bin', '.h5', '.msgpack', '.onnx'))}
        ensure_space(sum((x or 0 for x in encoder_files.values())), spec.directory)
        snapshot_download(tower, local_dir=encoder_dir, allow_patterns=list(encoder_files), endpoint='https://huggingface.co', token=False, max_workers=4)
        atomic_json(OUTPUT_ROOT / spec.name / 'vision_encoder.json', {'official_repo': tower, 'directory': str(encoder_dir)})
    atomic_json(OUTPUT_ROOT / spec.name / 'download_status.json', {'model': spec.name, 'official_repo': spec.repo, 'state': 'ready'})
    print(f'Download verified: {spec.name}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=tuple(MODEL_BY_NAME))
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / 'download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(3):
            try:
                download(MODEL_BY_NAME[args.model])
                return
            except (OSError, ConnectionError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(15)
if __name__ == '__main__':
    main()
