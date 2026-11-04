"""Open vlm registry."""
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
QWEN_ROOT = Path('./downloaded_qwen')
LLAVA_ROOT = Path('./downloaded_LLaVA')
OUTPUT_ROOT = ROOT / 'results/inference/open_vlm'
EVALUATION_TASKS = ('landmarks', 'dior-recognition', 'pattern-net', 'rare-planes', 'cowc', 'xview2', 'neon-trees', 'aerial-animal', 'fpcd', 'dior-refdet', 'geobench-refdet')

@dataclass(frozen=True)
class Model:
    name: str
    repo: str
    backend: str = 'transformers'
    thinking: bool = False
    gpu: str = 'l40s'
    gpus: int = 1
    memory_gib: int = 64
    output_tokens: int = 4096
    directory_override: Path | None = None

    @property
    def family(self):
        return 'Qwen' if self.repo.startswith('Qwen/') else 'LLaVA'

    @property
    def directory(self):
        return self.directory_override or (QWEN_ROOT if self.family == 'Qwen' else LLAVA_ROOT) / self.name

def _qwen(name, *, backend='transformers', thinking=False, large=False, tokens=None):
    if 'Flash-Next' in name:
        count, memory = (4, 512)
    else:
        count, memory = (1, 128 if large else 64)
    return Model(name, 'Qwen/' + name, backend, thinking, 'h200' if large or count > 1 else 'l40s', count, memory, tokens or (32768 if thinking else 4096))
MODELS = (_qwen('Qwen3.5-9B', thinking=True), _qwen('Qwen3.5-27B', thinking=True, large=True), _qwen('Qwen3.6-27B', thinking=True, large=True), _qwen('Qwen3.8-27B', thinking=True, large=True), _qwen('Qwen3.8-Flash-Next', thinking=True, large=True, tokens=81920), Model('LLaVA-OneVision-1.5-8B-RL', 'mvp-lab/LLaVA-OneVision-1.5-8B-RL', 'llava-rl', thinking=True, output_tokens=32768))
MODEL_BY_NAME = {m.name: m for m in MODELS}
