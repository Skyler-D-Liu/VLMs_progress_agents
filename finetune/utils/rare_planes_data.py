"""Rare planes data."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import random
import struct
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import zlib
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
DATASET_ID = 's3://rareplanes-public/real'
DATASET_CONFIG = 'instances_aircraft_tiled'
DATASET_DIRECTORY_NAME = 'rare_planes'
DEFAULT_RAW_DATASET_ROOT = Path('./raw_dataset')
S3_BASE_URL = 'https://rareplanes-public.s3.amazonaws.com'
OFFICIAL_SPLITS = ('train', 'test')
ANNOTATION_OBJECTS = {'train': 'real/metadata_annotations/instances_train_aircraft.json', 'test': 'real/metadata_annotations/instances_test_aircraft.json'}
SPLIT_SEED = 0
MAX_TILES_PER_SOURCE_SCENE = 5
DOWNLOAD_WORKERS = 8
DOWNLOAD_ATTEMPTS = 5
_SELECTION_MANIFEST = 'selection_manifest.json'
_OFFICIAL_TEST_MANIFEST = 'official_test_selection_manifest.json'

@dataclass(frozen=True)
class RarePlanesRecord:
    source_id: str
    canonical_split: str
    official_split: str
    raw_index: int
    coco_image_id: int
    file_name: str
    source_scene_id: str
    image_relative_path: str
    count: int
    bboxes: tuple[tuple[float, float, float, float], ...]

    def to_manifest_entry(self) -> dict[str, Any]:
        value = asdict(self)
        value['bboxes'] = [list(box) for box in self.bboxes]
        return value

@dataclass(frozen=True)
class RarePlanesBundle:
    dataset: Path
    records_by_split: Mapping[str, tuple[RarePlanesRecord, ...]]
    audit: Mapping[str, Any]

    def records(self, split: str) -> tuple[RarePlanesRecord, ...]:
        normalized = 'validation' if split in {'val', 'valid'} else split
        try:
            return self.records_by_split[normalized]
        except KeyError as exc:
            raise ValueError(f'unknown RarePlanes canonical split: {split!r}') from exc

    def read_image_bytes(self, record: RarePlanesRecord) -> bytes:
        path = (self.dataset / record.image_relative_path).resolve()
        try:
            path.relative_to(self.dataset.resolve())
        except ValueError as exc:
            raise ValueError(f'RarePlanes image escapes dataset root: {path}') from exc
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f'RarePlanes image is unavailable: {record.source_id}') from exc
        if not _valid_png(path):
            raise ValueError(f'RarePlanes image is not a valid PNG payload: {path}')
        return raw

    def manifest(self) -> dict[str, Any]:
        return {'dataset_id': DATASET_ID, 'dataset_config': DATASET_CONFIG, 'dataset_root': str(self.dataset), 'imagery': 'real_observed_satellite_only', 'synthetic_imagery_used': False, 'label_authority': 'count_of_all_official_aircraft_annotations_per_tile', 'split_seed': SPLIT_SEED, 'audit': dict(self.audit), 'records': {split: [record.to_manifest_entry() for record in records] for split, records in self.records_by_split.items()}}

def record_to_candidate_record(candidate: _Candidate, *, canonical_split: str) -> RarePlanesRecord:
    relative = Path('real') / candidate.official_split / 'PS-RGB_tiled' / candidate.file_name
    return RarePlanesRecord(source_id=f'rare-planes:real:{candidate.official_split}:{candidate.file_name}', canonical_split=canonical_split, official_split=candidate.official_split, raw_index=candidate.raw_index, coco_image_id=candidate.coco_image_id, file_name=candidate.file_name, source_scene_id=f'{candidate.official_split}:{candidate.source_scene_id}', image_relative_path=relative.as_posix(), count=candidate.count, bboxes=candidate.bboxes)

@dataclass(frozen=True)
class _Candidate:
    official_split: str
    raw_index: int
    coco_image_id: int
    file_name: str
    source_scene_id: str
    count: int
    bboxes: tuple[tuple[float, float, float, float], ...]

def dataset_directory(cache_dir: str | Path | None=None) -> Path:
    root = Path(cache_dir or DEFAULT_RAW_DATASET_ROOT).expanduser().resolve()
    return root if root.name == DATASET_DIRECTORY_NAME else root / DATASET_DIRECTORY_NAME

def _safe_file_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f'RarePlanes COCO file_name is invalid: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 1:
        raise ValueError(f'unsafe RarePlanes COCO file_name: {value!r}')
    if path.suffix.casefold() != '.png':
        raise ValueError(f'RarePlanes tiled image is not PNG: {value!r}')
    return path.name

def _source_scene_id(file_name: str) -> str:
    marker = '_tile_'
    if marker not in file_name:
        raise ValueError(f'RarePlanes tiled filename lacks {marker!r}: {file_name!r}')
    scene, tile = file_name.rsplit(marker, 1)
    if not scene or not tile.removesuffix('.png').isdigit():
        raise ValueError(f'RarePlanes tiled filename is malformed: {file_name!r}')
    return scene

def _valid_download(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix.casefold() == '.json':
        try:
            return isinstance(json.loads(path.read_text(encoding='utf-8')), dict)
        except (OSError, json.JSONDecodeError):
            return False
    if path.suffix.casefold() == '.png':
        return _valid_png(path)
    return True

def _valid_png(path: Path) -> bool:
    signature = b'\x89PNG\r\n\x1a\n'
    try:
        with path.open('rb') as handle:
            if handle.read(8) != signature:
                return False
            saw_ihdr = False
            saw_idat = False
            while True:
                header = handle.read(8)
                if len(header) != 8:
                    return False
                length, chunk_type = struct.unpack('>I4s', header)
                if length > 512 * 1024 * 1024:
                    return False
                payload = handle.read(length)
                crc_bytes = handle.read(4)
                if len(payload) != length or len(crc_bytes) != 4:
                    return False
                expected_crc = struct.unpack('>I', crc_bytes)[0]
                if zlib.crc32(chunk_type + payload) & 4294967295 != expected_crc:
                    return False
                if chunk_type == b'IHDR':
                    if saw_ihdr or length != 13:
                        return False
                    width, height = struct.unpack('>II', payload[:8])
                    if width != 512 or height != 512:
                        return False
                    saw_ihdr = True
                elif chunk_type == b'IDAT':
                    if not saw_ihdr:
                        return False
                    saw_idat = True
                elif chunk_type == b'IEND':
                    if length != 0 or not saw_ihdr or (not saw_idat):
                        return False
                    return handle.read(1) == b''
                elif not saw_ihdr:
                    return False
    except (OSError, struct.error, ValueError):
        return False

def _download(url: str, destination: Path) -> None:
    if destination.is_file():
        if _valid_download(destination):
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.downloading-{os.getpid()}')
    last_error: BaseException | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        request = Request(url, headers={'User-Agent': 'VLEO-Benchmark'})
        temporary.unlink(missing_ok=True)
        try:
            with urlopen(request, timeout=300) as response, temporary.open('wb') as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if not _valid_download(temporary):
                raise RuntimeError(f'RarePlanes download is unreadable: {url}')
            os.replace(temporary, destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), 8))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    assert last_error is not None
    raise RuntimeError(f'RarePlanes download failed after {DOWNLOAD_ATTEMPTS} attempts for {url}: {type(last_error).__name__}: {last_error}') from last_error

def _annotation_paths(root: Path) -> dict[str, Path]:
    metadata = root / 'metadata'
    return {split: metadata / f'instances_{split}_aircraft.json' for split in OFFICIAL_SPLITS}

def _ensure_annotations(root: Path) -> dict[str, Path]:
    paths = _annotation_paths(root)
    lock_path = root / '.metadata.download.lock'
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for split, path in paths.items():
            _download(f'{S3_BASE_URL}/{ANNOTATION_OBJECTS[split]}', path)
    return paths

def _integer(value: Any, *, field: str, split: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'RarePlanes {split} {field} must be an integer: {value!r}')
    return value

def _bbox(value: Any, *, split: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f'RarePlanes {split} bbox must have four values: {value!r}')
    box = tuple((float(item) for item in value))
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(f'RarePlanes {split} bbox has non-positive size: {value!r}')
    return box

def _parse_annotation_file(split: str, path: Path) -> tuple[_Candidate, ...]:
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'cannot read RarePlanes COCO annotations: {path}') from exc
    if not isinstance(document, Mapping):
        raise ValueError(f'RarePlanes {split} COCO document is not an object')
    images = document.get('images')
    annotations = document.get('annotations')
    categories = document.get('categories')
    if not isinstance(images, list) or not isinstance(annotations, list) or (not isinstance(categories, list)):
        raise ValueError(f'RarePlanes {split} COCO collections are invalid')
    if len(categories) != 1 or str(categories[0].get('name', '')).casefold() != 'aircraft':
        raise ValueError(f'RarePlanes {split} must contain only the aircraft category')
    aircraft_category_id = _integer(categories[0].get('id'), field='category.id', split=split)
    image_rows: dict[int, tuple[int, str]] = {}
    file_names: set[str] = set()
    for raw_index, image in enumerate(images):
        if not isinstance(image, Mapping):
            raise ValueError(f'RarePlanes {split} image row is not an object')
        image_id = _integer(image.get('id'), field='image.id', split=split)
        if image_id in image_rows:
            raise ValueError(f'duplicate RarePlanes {split} image id: {image_id}')
        if _integer(image.get('width'), field='image.width', split=split) != 512:
            raise ValueError(f'RarePlanes {split} image width is not 512: {image_id}')
        if _integer(image.get('height'), field='image.height', split=split) != 512:
            raise ValueError(f'RarePlanes {split} image height is not 512: {image_id}')
        file_name = _safe_file_name(image.get('file_name'))
        if file_name in file_names:
            raise ValueError(f'duplicate RarePlanes {split} file_name: {file_name}')
        file_names.add(file_name)
        image_rows[image_id] = (raw_index, file_name)
    boxes_by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError(f'RarePlanes {split} annotation row is not an object')
        annotation_id = _integer(annotation.get('id'), field='annotation.id', split=split)
        if annotation_id in annotation_ids:
            raise ValueError(f'duplicate RarePlanes {split} annotation id: {annotation_id}')
        annotation_ids.add(annotation_id)
        image_id = _integer(annotation.get('image_id'), field='annotation.image_id', split=split)
        if image_id not in image_rows:
            raise ValueError(f'RarePlanes {split} annotation references unknown image: {image_id}')
        if _integer(annotation.get('category_id'), field='annotation.category_id', split=split) != aircraft_category_id:
            raise ValueError(f'RarePlanes {split} annotation is not aircraft: {annotation_id}')
        boxes_by_image[image_id].append(_bbox(annotation.get('bbox'), split=split))
    candidates = []
    for image_id, (raw_index, file_name) in image_rows.items():
        boxes = tuple(boxes_by_image.get(image_id, ()))
        if not boxes:
            raise ValueError(f'RarePlanes {split} positive-tile COCO image has no aircraft: {image_id}')
        candidates.append(_Candidate(official_split=split, raw_index=raw_index, coco_image_id=image_id, file_name=file_name, source_scene_id=_source_scene_id(file_name), count=len(boxes), bboxes=boxes))
    return tuple(sorted(candidates, key=lambda item: (item.file_name, item.coco_image_id)))


def select_canonical_records(candidates: Iterable[_Candidate], *, seed: int=SPLIT_SEED) -> dict[str, tuple[RarePlanesRecord, ...]]:
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[f'{candidate.official_split}:{candidate.source_scene_id}'].append(candidate)
    scene_keys = sorted(grouped)
    random.Random(seed).shuffle(scene_keys)
    train_end = int(len(scene_keys) * 0.8)
    validation_end = int(len(scene_keys) * 0.9)
    selected = {'train': [], 'validation': [], 'test': []}
    for index, key in enumerate(scene_keys):
        split = 'train' if index < train_end else 'validation' if index < validation_end else 'test'
        selected[split].extend(grouped[key])
    output: dict[str, tuple[RarePlanesRecord, ...]] = {}
    for canonical_split, values in selected.items():
        rows = []
        for candidate in values:
            relative = Path('real') / candidate.official_split / 'PS-RGB_tiled' / candidate.file_name
            rows.append(RarePlanesRecord(source_id=f'rare-planes:real:{candidate.official_split}:{candidate.file_name}', canonical_split=canonical_split, official_split=candidate.official_split, raw_index=candidate.raw_index, coco_image_id=candidate.coco_image_id, file_name=candidate.file_name, source_scene_id=f'{candidate.official_split}:{candidate.source_scene_id}', image_relative_path=relative.as_posix(), count=candidate.count, bboxes=candidate.bboxes))
        output[canonical_split] = tuple(rows)
    return output

def record_image_url(record: RarePlanesRecord) -> str:
    key = f'real/{record.official_split}/PS-RGB_tiled/{quote(record.file_name)}'
    return f'{S3_BASE_URL}/{key}'

def _download_selected_images(root: Path, records: Iterable[RarePlanesRecord]) -> None:
    unique = {record.source_id: record for record in records}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download, record_image_url(record), root / record.image_relative_path): record for record in unique.values()}
        for future in as_completed(futures):
            record = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append(f'{record.source_id}: {exc}')
    if failures:
        preview = '\n'.join(failures[:10])
        raise RuntimeError(f'RarePlanes image downloads failed ({len(failures)}):\n{preview}')

def download_rare_planes_images(root: str | Path, records: Iterable[RarePlanesRecord]) -> None:
    records = tuple(records)
    dataset_root = dataset_directory(root)
    _download_selected_images(dataset_root, records)
    missing = [record.source_id for record in records if not (dataset_root / record.image_relative_path).is_file()]
    if missing:
        raise RuntimeError(f'RarePlanes selected image downloads are incomplete ({len(missing)} missing): {missing[:5]}')

def _write_official_test_manifest(root: Path, *, seed: int, selected_test_records: Sequence[RarePlanesRecord]) -> Path:
    manifest = {'dataset_id': DATASET_ID, 'dataset_config': DATASET_CONFIG, 'official_split': 'test', 'selection_seed': int(seed), 'selected_records': [record.to_manifest_entry() for record in selected_test_records]}
    path = root / _OFFICIAL_TEST_MANIFEST
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if not path.is_file() or path.read_text(encoding='utf-8') != serialized:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    return path

def select_official_test_records(candidates: Iterable[_Candidate], *, seed: int=SPLIT_SEED) -> tuple[RarePlanesRecord, ...]:
    all_records = tuple((record_to_candidate_record(candidate, canonical_split='test') for candidate in candidates))
    if any((record.official_split != 'test' for record in all_records)):
        raise ValueError('official test selection received a non-test candidate')
    if len({record.source_id for record in all_records}) != len(all_records):
        raise ValueError('official test selection contains duplicate source identities')
    all_records = tuple(sorted(all_records, key=lambda record: (record.file_name, record.coco_image_id)))
    positions = list(range(len(all_records)))
    random.Random(int(seed)).shuffle(positions)
    return tuple((all_records[position] for position in positions))

def build_official_test_bundle(cache_dir: str | Path | None=None, *, seed: int=SPLIT_SEED, download_images: bool=True, strict: bool=True) -> RarePlanesBundle:
    root = dataset_directory(cache_dir)
    annotation_paths = _ensure_annotations(root)
    train_candidates = _parse_annotation_file('train', annotation_paths['train'])
    test_candidates = _parse_annotation_file('test', annotation_paths['test'])
    train_records = tuple((record_to_candidate_record(candidate, canonical_split='train') for candidate in train_candidates))
    selected_test_records = select_official_test_records(test_candidates, seed=seed)
    if download_images:
        download_rare_planes_images(root, selected_test_records)
    if strict and download_images:
        missing = [record.source_id for record in selected_test_records if not (root / record.image_relative_path).is_file()]
        if missing:
            raise RuntimeError(f'RarePlanes official test images missing: {missing[:5]}')
    _write_official_test_manifest(root, seed=int(seed), selected_test_records=selected_test_records)
    audit = {'selection_seed': int(seed), 'selected_official_split': 'test', 'synthetic_imagery_used': False}
    return RarePlanesBundle(root, {'train': train_records, 'test': selected_test_records}, audit)


def build_rare_planes_bundle(cache_dir: str | Path | None=None, *, download_images: bool=True, strict: bool=True) -> RarePlanesBundle:
    root = dataset_directory(cache_dir)
    annotation_paths = _ensure_annotations(root)
    candidates = tuple((candidate for split in OFFICIAL_SPLITS for candidate in _parse_annotation_file(split, annotation_paths[split])))
    records_by_split = select_canonical_records(candidates)
    audit = {}
    if download_images:
        _download_selected_images(root, (record for values in records_by_split.values() for record in values))
    if strict and download_images:
        for values in records_by_split.values():
            for record in values:
                root_path = root / record.image_relative_path
                if not root_path.is_file():
                    raise RuntimeError(f'RarePlanes selected image is missing: {root_path}')
    bundle = RarePlanesBundle(root, records_by_split, audit)
    manifest_path = root / _SELECTION_MANIFEST
    manifest = bundle.manifest()
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if not manifest_path.is_file() or manifest_path.read_text(encoding='utf-8') != serialized:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=manifest_path.parent, delete=False) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, manifest_path)
    return bundle
