"""Fpcd data."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile
from finetune.utils.dataset_splits import build_leakage_aware_source_id_splits
DATASET_ID = 'ctundia/FPCD'
KNOWN_INCORRECT_ZENODO_RECORD_ID = '17317117'
ZENODO_API_URL = 'https://zenodo.org/api/records/{record_id}'
HF_RESOLVE_URL = 'https://huggingface.co/datasets/{dataset}/resolve/main/{filename}?download=true'
GEOBENCH_DATASET_ID = 'aialliance/GEOBench-VLM'
SPLIT_SEED = 0
ARCHIVE_FILENAMES = ('T0.zip', 'T1.zip', 'multi_class_masks.zip')
LIST_FILENAMES = ('cd_dataset_train.txt', 'cd_dataset_test.txt')
REQUIRED_SOURCE_FILENAMES = (*ARCHIVE_FILENAMES, *LIST_FILENAMES)
HF_SOURCE_FILE_METADATA = {'T0.zip': {'size': 227134211}, 'T1.zip': {'size': 202995596}, 'multi_class_masks.zip': {'size': 557985}, 'cd_dataset_train.txt': {'size': 21707}, 'cd_dataset_test.txt': {'size': 2291}}
MASK_TO_OPTION = {0: 'D', 1: 'E', 2: 'A', 3: 'C', 4: 'B'}
OPTION_TO_CLASS = {'A': 'Farm Pond Demolished', 'B': 'Farm Pond Wetted', 'C': 'Farm Pond Dried', 'D': 'No Change', 'E': 'Farm Pond Constructed'}
_DATE_COMPONENT = re.compile('^\\d{6}$')
_IMAGE_SUFFIXES = frozenset(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'))

@dataclass(frozen=True)
class FPCDRecord:
    source_id: str
    canonical_split: str
    raw_split: str
    raw_index: int
    t0_member: str
    t1_member: str
    mask_member: str
    mask_classes: tuple[int, ...]
    answer: str

    @property
    def class_name(self) -> str:
        return OPTION_TO_CLASS[self.answer]

    def to_manifest_entry(self) -> dict[str, Any]:
        value = asdict(self)
        value['mask_classes'] = list(self.mask_classes)
        value['class_name'] = self.class_name
        return value

@dataclass(frozen=True)
class FPCDBundle:
    records_by_split: Mapping[str, tuple[FPCDRecord, ...]]
    archive_paths: Mapping[str, Path]
    audit: Mapping[str, Any]

    def records(self, split: str) -> tuple[FPCDRecord, ...]:
        normalized = 'validation' if split == 'val' else split
        try:
            return self.records_by_split[normalized]
        except KeyError as exc:
            raise ValueError(f'unknown canonical FPCD split: {split!r}') from exc

    def read_image_bytes(self, record: FPCDRecord) -> tuple[bytes, bytes]:
        return (read_zip_member(self.archive_paths['T0.zip'], record.t0_member), read_zip_member(self.archive_paths['T1.zip'], record.t1_member))

    def manifest(self) -> dict[str, Any]:
        zenodo_record_id = os.environ.get('FPCD_ZENODO_RECORD_ID')
        source = {'type': 'zenodo', 'record_id': zenodo_record_id, 'record_url': f'https://zenodo.org/records/{zenodo_record_id}'} if zenodo_record_id else {'type': 'huggingface_direct_files', 'dataset_id': DATASET_ID, 'files': list(REQUIRED_SOURCE_FILENAMES)}
        return {'dataset_id': DATASET_ID, 'source': source, 'geobench_prompt_provenance': {'dataset_id': GEOBENCH_DATASET_ID, 'task': 'FPCD', 'prompt_index': 5}, 'split_seed': SPLIT_SEED, 'mask_to_option': {str(key): value for key, value in MASK_TO_OPTION.items()}, 'audit': dict(self.audit), 'records': {split: [record.to_manifest_entry() for record in records] for split, records in self.records_by_split.items()}}

def normalize_source_id(filename: str) -> str:
    name = Path(str(filename).replace('\\', '/')).name
    stem = Path(name).stem
    components = stem.split('_')
    if len(components) >= 2 and _DATE_COMPONENT.fullmatch(components[-2]):
        del components[-2]
    if len(components) < 3 or not components[-1]:
        raise ValueError(f'cannot derive FPCD source ID from {filename!r}')
    return '_'.join(components)

class FPCDSourceError(RuntimeError):
    pass

def _read_json_url(url: str, retries: int=5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'VLEO-FPCD'}), timeout=60) as response:
                value = json.loads(response.read().decode('utf-8'))
            if not isinstance(value, dict):
                raise FPCDSourceError(f'Zenodo metadata is not an object: {url}')
            return value
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 10))
    raise FPCDSourceError(f'unable to read Zenodo metadata after {retries} attempts: {url}') from last_error

def zenodo_record_metadata(record_id: str | None=None) -> dict[str, Any]:
    selected = str(record_id or os.environ.get('FPCD_ZENODO_RECORD_ID') or '')
    if not selected:
        raise FPCDSourceError('an explicit FPCD Zenodo record ID is required')
    metadata = _read_json_url(ZENODO_API_URL.format(record_id=selected))
    title = str((metadata.get('metadata') or {}).get('title') or '')
    normalized = title.casefold()
    if 'farm pond' not in normalized and 'fpcd' not in normalized:
        raise FPCDSourceError(f'Zenodo record {selected} is not FPCD: title={title!r}. Record 17317117 is the EverWatch bird dataset, not Farm Pond Change Detection.')
    return metadata

def resumable_http_download(url: str, destination: str | Path, *, expected_size: int | None=None, retries: int=10) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        size_matches = expected_size is None or path.stat().st_size == expected_size
        if size_matches:
            return path
    partial = path.with_name(path.name + '.part')
    last_error: Exception | None = None
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset > expected_size:
            partial.unlink()
            offset = 0
        if expected_size is not None and offset == expected_size:
            os.replace(partial, path)
            return path
        headers = {'User-Agent': 'VLEO-FPCD'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                status = getattr(response, 'status', response.getcode())
                append = bool(offset and status == 206)
                if offset and (not append):
                    offset = 0
                mode = 'ab' if append else 'wb'
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise IOError(f'incomplete download: {partial.stat().st_size} != {expected_size} bytes')
            os.replace(partial, path)
            return path
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 30))
    raise FPCDSourceError(f'download failed after {retries} resumable attempts: {url}') from last_error

def _extract_required_from_package(package: Path, destination: Path) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    with ZipFile(package) as archive:
        by_basename: dict[str, str] = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if basename in REQUIRED_SOURCE_FILENAMES:
                if basename in by_basename:
                    raise FPCDSourceError(f'Zenodo package repeats required file {basename!r}')
                by_basename[basename] = info.filename
        missing = set(REQUIRED_SOURCE_FILENAMES) - set(by_basename)
        if missing:
            raise FPCDSourceError(f'Zenodo package {package.name!r} lacks FPCD files: {sorted(missing)}')
        for basename, member in by_basename.items():
            output = destination / basename
            temporary = output.with_name(output.name + '.extracting')
            if not output.is_file():
                with archive.open(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                os.replace(temporary, output)
            extracted[basename] = output.resolve()
    return extracted

def download_fpcd_source_files(cache_dir: str | Path | None=None, record_id: str | None=None) -> dict[str, Path]:
    selected = str(record_id or os.environ.get('FPCD_ZENODO_RECORD_ID') or '')
    root = Path(cache_dir).expanduser() if cache_dir else Path.home() / '.cache' / 'vleo'
    if not selected:
        destination = (root / 'fpcd' / 'huggingface').resolve()
        destination.mkdir(parents=True, exist_ok=True)
        resolved: dict[str, Path] = {}
        for filename in REQUIRED_SOURCE_FILENAMES:
            metadata = HF_SOURCE_FILE_METADATA[filename]
            resolved[filename] = resumable_http_download(HF_RESOLVE_URL.format(dataset=DATASET_ID, filename=filename), destination / filename, expected_size=int(metadata['size']))
        return resolved
    metadata = zenodo_record_metadata(selected)
    destination = (root / 'fpcd' / f'zenodo-{selected}').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = metadata.get('files') or []
    entries = {str(item.get('key')): item for item in files if isinstance(item, Mapping)}
    if set(REQUIRED_SOURCE_FILENAMES).issubset(entries):
        resolved = {}
        for filename in REQUIRED_SOURCE_FILENAMES:
            entry = entries[filename]
            url = str((entry.get('links') or {}).get('self') or '')
            resolved[filename] = resumable_http_download(url, destination / filename, expected_size=int(entry['size']))
        return resolved
    packages = [item for item in files if str(item.get('key', '')).casefold().endswith('.zip')]
    if len(packages) != 1:
        raise FPCDSourceError('FPCD Zenodo record must expose the five required files or one containing ZIP')
    entry = packages[0]
    package = resumable_http_download(str((entry.get('links') or {}).get('self') or ''), destination / str(entry['key']), expected_size=int(entry['size']))
    return _extract_required_from_package(package, destination)

def load_official_filename_lists(source_files: Mapping[str, Path]) -> dict[str, list[str]]:
    output = {}
    for split, filename in (('train', 'cd_dataset_train.txt'), ('test', 'cd_dataset_test.txt')):
        output[split] = [line.strip() for line in source_files[filename].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    return output


def index_zip_members(path: str | Path) -> dict[str, str]:
    index: dict[str, str] = {}
    with ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.casefold() not in _IMAGE_SUFFIXES:
                continue
            source_id = normalize_source_id(info.filename)
            previous = index.get(source_id)
            if previous is not None:
                raise ValueError(f'archive {path} has multiple members for {source_id!r}: {previous!r}, {info.filename!r}')
            index[source_id] = info.filename
    if not index:
        raise ValueError(f'archive has no supported images: {path}')
    return index

def read_zip_member(path: str | Path, member: str) -> bytes:
    with ZipFile(path) as archive:
        return archive.read(member)

def mask_classes_from_bytes(raw: bytes) -> tuple[int, ...]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError('FPCD mask decoding requires Pillow') from exc
    with Image.open(BytesIO(raw)) as image:
        image.load()
        pixels = image.get_flattened_data() if hasattr(image, 'get_flattened_data') else image.getdata()
        values = {int(value) for value in pixels}
    unexpected = values - set(MASK_TO_OPTION)
    if unexpected:
        raise ValueError(f'FPCD mask contains values outside 0..4: {sorted(unexpected)}')
    nonzero = tuple(sorted((value for value in values if value != 0)))
    return nonzero or (0,)

def answer_from_mask_classes(mask_classes: Sequence[int]) -> str:
    classes = tuple((int(value) for value in mask_classes))
    nonzero = tuple((value for value in classes if value != 0))
    if len(set(nonzero)) > 1:
        raise ValueError(f'multi-label FPCD mask has classes {sorted(set(nonzero))}')
    label = nonzero[0] if nonzero else 0
    try:
        return MASK_TO_OPTION[label]
    except KeyError as exc:
        raise ValueError(f'unknown FPCD mask class: {label}') from exc


def _validate_native_ids(filenames: Iterable[str], split: str) -> tuple[list[str], dict[str, int]]:
    source_ids = []
    raw_index_by_id = {}
    for raw_index, filename in enumerate(filenames):
        source_id = normalize_source_id(filename)
        if source_id in raw_index_by_id:
            raise ValueError(f'native {split} list repeats source ID {source_id!r}')
        raw_index_by_id[source_id] = raw_index
        source_ids.append(source_id)
    return (source_ids, raw_index_by_id)

def build_fpcd_bundle(cache_dir: str | Path | None=None, *, strict: bool=True, filename_lists: Mapping[str, Sequence[str]] | None=None, archive_paths: Mapping[str, str | Path] | None=None) -> FPCDBundle:
    downloaded = None
    if filename_lists is None or archive_paths is None:
        downloaded = download_fpcd_source_files(cache_dir)
    lists = {key: list(values) for key, values in filename_lists.items()} if filename_lists is not None else load_official_filename_lists(downloaded)
    archives = {key: Path(value).expanduser().resolve() for key, value in archive_paths.items()} if archive_paths is not None else {filename: downloaded[filename] for filename in ARCHIVE_FILENAMES}
    if set(archives) != set(ARCHIVE_FILENAMES):
        raise ValueError(f'FPCD archives must be exactly {list(ARCHIVE_FILENAMES)}')
    train_ids, train_raw_indices = _validate_native_ids(lists['train'], 'train')
    test_ids, test_raw_indices = _validate_native_ids(lists['test'], 'test')
    member_indices = {'T0.zip': index_zip_members(archives['T0.zip']), 'T1.zip': index_zip_members(archives['T1.zip']), 'multi_class_masks.zip': index_zip_members(archives['multi_class_masks.zip'])}
    required_ids = set(train_ids) | set(test_ids)
    missing_masks = required_ids - set(member_indices['multi_class_masks.zip'])
    if missing_masks:
        raise ValueError(f'multi_class_masks.zip is missing {len(missing_masks)} listed source IDs; first={sorted(missing_masks)[0]}')
    mask_classes_by_id = {source_id: mask_classes_from_bytes(read_zip_member(archives['multi_class_masks.zip'], member_indices['multi_class_masks.zip'][source_id])) for source_id in sorted(required_ids)}
    overlap = set(train_ids) & set(test_ids)
    independent_train_ids = [source_id for source_id in train_ids if source_id not in overlap]
    multi_label_ids = {source_id for source_id, classes in mask_classes_by_id.items() if len([value for value in classes if value != 0]) > 1}
    multi_train = multi_label_ids & set(independent_train_ids)
    multi_test = multi_label_ids & set(test_ids)
    source_splits = build_leakage_aware_source_id_splits(train_ids, test_ids, seed=SPLIT_SEED, train_ratio=0.9, val_ratio=0.1, excluded_train_source_ids=multi_train, excluded_test_source_ids=multi_test)
    canonical_ids = {'train': source_splits['train'], 'validation': source_splits['val'], 'test': source_splits['test']}
    canonical_required_ids = {source_id for source_ids in canonical_ids.values() for source_id in source_ids}
    raw_missing_images: dict[str, list[str]] = {}
    for filename in ('T0.zip', 'T1.zip'):
        raw_missing = sorted(required_ids - set(member_indices[filename]))
        raw_missing_images[filename] = raw_missing
        canonical_missing = canonical_required_ids - set(member_indices[filename])
        if canonical_missing:
            raise ValueError(f'{filename} is missing {len(canonical_missing)} canonical source IDs; first={sorted(canonical_missing)[0]}')
    raw_indices = {'train': train_raw_indices, 'test': test_raw_indices}
    records_by_split = {}
    for canonical_split, source_ids in canonical_ids.items():
        raw_split = 'test' if canonical_split == 'test' else 'train'
        records = []
        for source_id in source_ids:
            classes = mask_classes_by_id[source_id]
            records.append(FPCDRecord(source_id=source_id, canonical_split=canonical_split, raw_split=raw_split, raw_index=raw_indices[raw_split][source_id], t0_member=member_indices['T0.zip'][source_id], t1_member=member_indices['T1.zip'][source_id], mask_member=member_indices['multi_class_masks.zip'][source_id], mask_classes=classes, answer=answer_from_mask_classes(classes)))
        records_by_split[canonical_split] = tuple(records)
    audit = {'native_train_contains_all_test': overlap == set(test_ids), 'independent_train_candidate_count': len(independent_train_ids), 'multi_label_train_count': len(multi_train), 'multi_label_train_source_ids': sorted(multi_train), 'raw_missing_image_source_ids': raw_missing_images, 'excluded_missing_image_source_ids': sorted((source_id for source_ids in raw_missing_images.values() for source_id in source_ids if source_id not in canonical_required_ids)), 'eligible_train_candidate_count': len(independent_train_ids) - len(multi_train), 'test_data_used_for_tuning': False}
    return FPCDBundle(records_by_split=records_by_split, archive_paths=archives, audit=audit)

def write_frozen_manifest(bundle: FPCDBundle, path: str | Path, **extra: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = bundle.manifest()
    value.update(extra)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + '\n'
    temporary = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
    temporary.write_text(encoded, encoding='utf-8')
    os.replace(temporary, destination)
    return destination
