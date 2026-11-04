"""Images."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from PIL import Image

@dataclass(frozen=True)
class InputImage:
    data: bytes
    suffix: str
    width: int
    height: int

    def filename(self, position):
        return f'image_{position}{self.suffix}'

class ImageStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def load(self, sample):
        images = []
        for item in sample.row['image_files']:
            path = (self.root / item['path']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Image path leaves the experiment directory')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Experiment input bytes changed')
            with Image.open(BytesIO(raw)) as image:
                images.append(InputImage(raw, path.suffix, *image.size))
        return images
