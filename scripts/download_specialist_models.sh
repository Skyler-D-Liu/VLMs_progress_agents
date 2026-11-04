#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname "${BASH_SOURCE[0]}")/../specialist_models"
cd "$(dirname "${BASH_SOURCE[0]}")/../specialist_models"
PY=${PYTHON:-python3}
"$PY" - "${1:-mmclip}" <<'PY'
from huggingface_hub import snapshot_download
import sys
selection = sys.argv[1]
if selection != 'mmclip':
    raise SystemExit('Usage: download_specialist_models.sh [mmclip]')
snapshot_download('openai/clip-vit-large-patch14-336', local_dir='hf/clip-vit-large-patch14-336', max_workers=2)
snapshot_download('OskarK1/MMLandmarks_baseline', local_dir='MMLandmarks/pretrained_model', max_workers=2)
PY
