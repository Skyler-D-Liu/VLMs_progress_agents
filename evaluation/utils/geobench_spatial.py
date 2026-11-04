"""Geobench spatial."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vleo_runtime.localization import ground_truth_polygons, localization_prompt, polygon_boxes, polygon_iou, source_dataset, source_metadata
HF_REPO_ID = 'aialliance/GEOBench-VLM'
HF_REVISION = 'b3485fe4971213ee13f91960638ddc5c4a098e70'
FEW_SHOT_LEVELS = (4,)
FEW_SHOT_SUPPORT_SIZE = max(FEW_SHOT_LEVELS)
FEW_SHOT_SELECTION_SEED = 0
EVALUATION_SELECTION_SEED = 0
LLAVA_MODEL_KEYS = frozenset({'LLaVA-OneVision-0.5B', 'LLaVA-OneVision-7B', 'LLaVA-OneVision-72B', 'LLaVA-OV-1.5-4B', 'LLaVA-OV-1.5-8B', 'LLaVA-OV-2-8B'})
REFUSAL_RE = re.compile("\\b(?:sorry|cannot|can't|unable|refuse|not able)\\b", re.I)
LOCATION_CUE_RE = re.compile('\\b(?:top|bottom|left|right|above|below|upper|lower)\\b', re.I)

def sha256_file(path: Path, chunk_size: int=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')

def _safe_member_name(name: str) -> str:
    normalized = str(PurePosixPath(name.replace('\\', '/')))
    path = PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe zip member: {name!r}')
    return normalized.lstrip('./')

def _qa_member(archive: zipfile.ZipFile, expected_parent: str) -> str:
    candidates = []
    for info in archive.infolist():
        safe = _safe_member_name(info.filename)
        if PurePosixPath(safe).name.casefold() != 'qa.json':
            continue
        score = 0 if expected_parent.casefold() in safe.casefold() else 1
        candidates.append((score, len(PurePosixPath(safe).parts), safe))
    if not candidates:
        raise ValueError(f'No qa.json found in archive for {expected_parent}')
    return min(candidates)[2]

def _qa_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        for key in ('questions', 'data', 'annotations', 'qa'):
            if isinstance(raw.get(key), list):
                rows = raw[key]
                break
        else:
            if all((isinstance(value, dict) for value in raw.values())):
                rows = list(raw.values())
            else:
                raise ValueError('Unsupported qa.json object schema')
    else:
        raise ValueError('qa.json must contain a list or object')
    return [dict(row) for row in rows if isinstance(row, dict)]

def _read_archive_qa(archive: zipfile.ZipFile, expected_parent: str) -> Tuple[str, List[Dict[str, Any]]]:
    member = _qa_member(archive, expected_parent)
    with archive.open(member) as handle:
        return (member, _qa_records(json.load(io.TextIOWrapper(handle, encoding='utf-8'))))

def _image_ref(row: Mapping[str, Any]) -> str:
    for key in ('image_path', 'image', 'file_name', 'image_name', 'filename'):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _safe_member_name(value.strip())
        if isinstance(value, Mapping) and isinstance(value.get('path'), str):
            return _safe_member_name(value['path'])
    raise ValueError(f'Question has no image path: {row}')

def _member_index(archive: zipfile.ZipFile) -> Tuple[Dict[str, zipfile.ZipInfo], Dict[str, List[str]]]:
    exact: Dict[str, zipfile.ZipInfo] = {}
    basenames: Dict[str, List[str]] = defaultdict(list)
    for info in archive.infolist():
        if info.is_dir():
            continue
        safe = _safe_member_name(info.filename)
        exact[safe.casefold()] = info
        basenames[PurePosixPath(safe).name.casefold()].append(safe)
    return (exact, basenames)

def _resolve_image_member(image_ref: str, exact: Mapping[str, zipfile.ZipInfo], basenames: Mapping[str, Sequence[str]]) -> zipfile.ZipInfo:
    reference = _safe_member_name(image_ref)
    ref_fold = reference.casefold()
    direct = exact.get(ref_fold)
    if direct is not None:
        return direct
    suffixes = [info for name, info in exact.items() if name.endswith('/' + ref_fold)]
    if len(suffixes) == 1:
        return suffixes[0]
    names = list(basenames.get(PurePosixPath(reference).name.casefold(), ()))
    if len(names) == 1:
        return exact[names[0].casefold()]
    raise ValueError(f'Could not uniquely resolve image {image_ref!r}; candidates={names[:5]}')

def _extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, task_dir: Path) -> Tuple[str, Dict[str, Any]]:
    safe = _safe_member_name(info.filename)
    destination = task_dir / 'files' / safe
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == info.file_size:
        pass
    else:
        temporary = destination.with_name(destination.name + '.partial')
        with archive.open(info) as source, temporary.open('wb') as target:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        os.replace(temporary, destination)
    metadata = {'archive_member': safe, 'crc32': f'{info.CRC:08x}', 'size': info.file_size}
    return (str(destination), metadata)

def image_dimensions(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
    return (int(width), int(height))

def points_to_xyxy(value: Any) -> List[float]:
    if isinstance(value, Mapping):
        if 'bbox' in value:
            return points_to_xyxy(value['bbox'])
        keys = ('xmin', 'ymin', 'xmax', 'ymax')
        if all((key in value for key in keys)):
            value = [value[key] for key in keys]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'Invalid bbox: {value!r}')
    if len(value) == 4 and all((isinstance(item, (int, float)) and (not isinstance(item, bool)) for item in value)):
        xmin, ymin, xmax, ymax = map(float, value)
    else:
        raise ValueError(f'Invalid bbox points: {value!r}')
    if not all((math.isfinite(number) for number in (xmin, ymin, xmax, ymax))):
        raise ValueError(f'Non-finite bbox: {value!r}')
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f'Empty bbox: {value!r}')
    return [xmin, ymin, xmax, ymax]

def _ground_truth_boxes(row: Mapping[str, Any]) -> List[List[float]]:
    value = row.get('ground_truth', row.get('bboxes', row.get('bbox')))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid serialized ground truth: {value!r}') from exc
    if isinstance(value, Mapping):
        value = value.get('bboxes', value.get('bbox', value))
    if not isinstance(value, list):
        raise ValueError(f'Invalid Ref-Det ground truth: {value!r}')
    try:
        return [points_to_xyxy(value)]
    except (TypeError, ValueError):
        return [points_to_xyxy(box) for box in value]

def clip_bbox_to_image(box: Sequence[float], width: int, height: int) -> List[float]:
    clipped = [min(float(width), max(0.0, float(box[0]))), min(float(height), max(0.0, float(box[1]))), min(float(width), max(0.0, float(box[2]))), min(float(height), max(0.0, float(box[3])))]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError(f'Ground-truth bbox is empty after clipping: {box}')
    return clipped

def _image_size(row: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    width = row.get('image_width', row.get('width'))
    height = row.get('image_height', row.get('height'))
    size = row.get('image_size')
    if width and height:
        return (int(width), int(height))
    if isinstance(size, (int, float)):
        return (int(size), int(size))
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    return None

def annotation_axis_boxes(row):
    value = row.get('ground_truth', row.get('ground_truth_bboxes'))
    if isinstance(value, list) and value and isinstance(value[0], list) and len(value[0]) == 4 and isinstance(value[0][0], (list, tuple)):
        return [[min(p[0] for p in box), min(p[1] for p in box),
                 max(p[0] for p in box), max(p[1] for p in box)] for box in value]
    return _ground_truth_boxes(row)

def normalize_refdet_row(row: Mapping[str, Any], row_index: int) -> Dict[str, Any]:
    prompts = row.get('prompts', row.get('prompt', row.get('question')))
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f'Ref-Det row {row_index} has no prompt')
    prompt = str(prompts[0])
    image_ref = _image_ref(row)
    question_id = row.get('question_id', row.get('id', row_index))
    source_value = str(row.get('source') or image_ref).casefold()
    source = 'geochat' if 'geochat' in source_value else 'geobench-vlms'
    boxes = (ground_truth_polygons(row) if source_dataset(row) == 'FAIR1M'
             else annotation_axis_boxes(row))
    return {'sample_id': f'refdet:{question_id}', 'question_id': question_id, 'image_ref': image_ref, 'prompt': prompt, 'ground_truth_bboxes_raw': boxes, 'ground_truth_bboxes': boxes, 'declared_image_size': _image_size(row), 'source': source, 'source_dataset': source_dataset(row), 'ground_truth_polygons': ground_truth_polygons(row) if source_dataset(row) == 'FAIR1M' else None, 'has_location_cue': bool(LOCATION_CUE_RE.search(prompt))}

def _write_immutable_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Refusing to replace non-matching immutable manifest: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    path.chmod(292)

def prepare_dataset(data_dir: str, cache_dir: Optional[str]=None) -> Path:
    from huggingface_hub import hf_hub_download
    root = Path(data_dir).expanduser().resolve() / f'GEOBench-VLM-{HF_REVISION}'
    root.mkdir(parents=True, exist_ok=True)
    archive_paths: Dict[str, Path] = {}
    for filename in ('Ref-Det.zip',):
        archive_paths[filename] = Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type='dataset', filename=filename, revision=HF_REVISION, cache_dir=cache_dir))
    all_tasks: Dict[str, List[Dict[str, Any]]] = {}
    archive_metadata: Dict[str, Dict[str, Any]] = {}
    extraction_metadata: Dict[str, Dict[str, Any]] = {}
    specs = (('refdet', 'Ref-Det.zip', 'Ref-Det'),)
    for task_name, archive_name, expected_parent in specs:
        archive_path = archive_paths[archive_name]
        archive_metadata[archive_name] = {'sha256': sha256_file(archive_path), 'size': archive_path.stat().st_size}
        task_dir = root / task_name
        with zipfile.ZipFile(archive_path) as archive:
            qa_member, raw_rows = _read_archive_qa(archive, expected_parent)
            if not raw_rows or any(source_dataset(row) not in {'DOTA', 'FAIR1M', 'DIOR'} for row in raw_rows):
                raise ValueError('Ref-Det requires explicit DOTA/FAIR1M metadata and FAIR1M polygon ground truth in dataset annotations or raw_dataset/geobench_spatial/source_metadata.json (build with evaluation.tasks.prepare_refdet_sources)')
            rows = [normalize_refdet_row(row, index) for index, row in enumerate(raw_rows) if source_dataset(row) in {'DOTA', 'FAIR1M'}]
            exact, basenames = _member_index(archive)
            extracted: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                info = _resolve_image_member(row['image_ref'], exact, basenames)
                image_path, member_meta = _extract_member(archive, info, task_dir)
                row['image_path'] = str(Path(image_path).relative_to(root))
                row['image_width'], row['image_height'] = image_dimensions(Path(image_path))
                declared = row.get('declared_image_size')
                if declared is not None and tuple(declared) != (row['image_width'], row['image_height']):
                    raise ValueError(f"Declared image size {declared} does not match actual {row['image_width']}x{row['image_height']} for {row['image_ref']}")
                raw_boxes = row['ground_truth_bboxes_raw']
                clipped_boxes = (raw_boxes if row['source_dataset'] == 'FAIR1M' else
                                 [clip_bbox_to_image(box, row['image_width'], row['image_height']) for box in raw_boxes])
                row['ground_truth_bboxes'] = clipped_boxes
                row['ground_truth_was_clipped'] = clipped_boxes != raw_boxes
                extracted[member_meta['archive_member']] = member_meta
            qa_info = exact[qa_member.casefold()]
            _, qa_meta = _extract_member(archive, qa_info, task_dir)
            extracted[qa_meta['archive_member']] = qa_meta
            extraction_metadata[task_name] = {'qa_member': qa_member, 'members': [extracted[key] for key in sorted(extracted)]}
            all_tasks[task_name] = rows
    manifest = {'dataset': HF_REPO_ID, 'revision': HF_REVISION, 'archives': archive_metadata, 'extraction': extraction_metadata, 'tasks': all_tasks}
    _write_immutable_json(root / 'manifest.json', manifest)
    return root / 'manifest.json'

def load_manifest(data_dir: str) -> Tuple[Path, Dict[str, Any]]:
    root = Path(data_dir).expanduser().resolve()
    if root.name != f'GEOBench-VLM-{HF_REVISION}':
        root = root / f'GEOBench-VLM-{HF_REVISION}'
    path = root / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Prepared manifest not found: {path}; run with --prepare-data')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('dataset') != HF_REPO_ID or manifest.get('revision') != HF_REVISION:
        raise ValueError(f'Manifest has the wrong source revision: {path}')
    return (root, manifest)


def few_shot_support_rows(rows: Sequence[Mapping[str, Any]], task: str, support_size: int=FEW_SHOT_SUPPORT_SIZE, seed: int=FEW_SHOT_SELECTION_SEED) -> List[Dict[str, Any]]:
    if task not in {'refdet'}:
        raise ValueError('Few-shot support is available only for refdet')
    if support_size < 1:
        raise ValueError('support_size must be positive')
    candidates = [dict(row) for row in rows]
    random.Random(seed).shuffle(candidates)
    selected: List[Dict[str, Any]] = []
    seen_images = set()
    for row in candidates:
        image_key = str(row.get('image_path') or row.get('image_ref') or '')
        if not image_key or image_key in seen_images:
            continue
        selected.append(row)
        seen_images.add(image_key)
        if len(selected) == support_size:
            break
    if len(selected) != support_size:
        raise ValueError(f'Could not reserve {support_size} unique support images for {task}; found {len(selected)}')
    return selected

def exclude_few_shot_support_images(rows: Sequence[Mapping[str, Any]], support_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    support_images = {str(row.get('image_path') or row.get('image_ref') or '') for row in support_rows}
    if '' in support_images:
        raise ValueError('Few-shot support row has no image identity')
    return [dict(row) for row in rows if str(row.get('image_path') or row.get('image_ref') or '') not in support_images]

def _json_candidates(text: str) -> Iterable[str]:
    from evaluation.utils.scoring import json_answer
    try:
        yield json.dumps(json_answer(text))
    except ValueError:
        return

def parse_bbox_response(value: Any, image_width: Optional[int]=None, image_height: Optional[int]=None) -> Dict[str, Any]:
    text = '' if value is None else str(value).strip()
    parsed = None
    parsed_entire_response = False
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and 'bboxes' in parsed:
            parsed_entire_response = candidate == text.strip()
            break
        parsed = None
    strict = False
    boxes: List[List[float]] = []
    error = None
    if parsed is None:
        error = 'invalid_json_schema'
    elif not isinstance(parsed.get('bboxes'), list):
        error = 'bboxes_not_list'
    else:
        try:
            if any(not isinstance(box, list) or len(box) != 4 or
                   any(type(v) not in (int, float) for v in box) for box in parsed['bboxes']):
                raise ValueError('Expected horizontal boxes as four pixel numbers')
            boxes = [points_to_xyxy(box) for box in parsed['bboxes']]
            if image_width is not None and image_height is not None:
                for box in boxes:
                    if box[0] < 0 or box[1] < 0 or box[2] > image_width or (box[3] > image_height):
                        raise ValueError('bbox_out_of_bounds')
            strict = parsed_entire_response
        except (TypeError, ValueError) as exc:
            boxes = []
            error = str(exc)
    return {'bboxes': boxes, 'format_valid': error is None, 'strict_format': strict and error is None, 'refused': not text or bool(REFUSAL_RE.search(text)), 'parse_error': error}

def parse_fair1m_response(value, image_width, image_height):
    text = str(value or '').strip()
    error = 'invalid_json_schema'
    for candidate in _json_candidates(text):
        try:
            decoded = json.loads(candidate)
            if not isinstance(decoded, dict) or 'bboxes' not in decoded:
                continue
            polygons = polygon_boxes(decoded['bboxes'])
            if image_width is not None and image_height is not None:
                if any(x < 0 or x > image_width or y < 0 or y > image_height for box in polygons for x, y in box):
                    raise ValueError('bbox_out_of_bounds')
            return {'bboxes': polygons, 'polygons': polygons, 'format_valid': True, 'strict_format': candidate == text, 'parse_error': None, 'refused': False}
        except (ValueError, TypeError) as exc:
            error = str(exc)
    return {'bboxes': [], 'polygons': [], 'format_valid': False, 'strict_format': False, 'parse_error': error, 'refused': not text or bool(REFUSAL_RE.search(text))}


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = (max(left[0], right[0]), max(left[1], right[1]))
    ix2, iy2 = (min(left[2], right[2]), min(left[3], right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0

def hungarian_match(predicted: Sequence[Sequence[float]], ground_truth: Sequence[Sequence[float]], iou_function=None) -> List[Tuple[int, int, float]]:
    if not predicted or not ground_truth:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    metric = iou_function or bbox_iou
    ious = np.asarray([[metric(pred, truth) for truth in ground_truth] for pred in predicted])
    pred_indices, truth_indices = linear_sum_assignment(-ious)
    return [(int(pi), int(ti), float(ious[pi, ti])) for pi, ti in zip(pred_indices, truth_indices)]

def centroid_grid(box: Sequence[float], width: int, height: int) -> Tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    column = min(2, max(0, int(3 * center_x / width)))
    row = min(2, max(0, int(3 * center_y / height)))
    return (row, column)


def evaluate_bbox_query(predicted: Sequence[Sequence[float]], ground_truth: Sequence[Sequence[float]], width: int, height: int, *, predicted_polygons=None, target_polygons=None) -> Dict[str, Any]:
    if target_polygons is not None:
        if predicted_polygons is None:
            raise ValueError('Oriented scoring requires prediction polygons')
        predicted = predicted_polygons
        ground_truth = target_polygons
        matches = hungarian_match(predicted_polygons, target_polygons, polygon_iou)
    else:
        matches = hungarian_match(predicted, ground_truth)
    denominator = max(len(predicted), len(ground_truth), 1)
    metrics: Dict[str, Any] = {'prediction_count': len(predicted), 'target_count': len(ground_truth), 'mIoU': sum((match[2] for match in matches)) / denominator}
    diagonal = math.hypot(width, height)
    metrics['matched_pair_count'] = len(matches)
    if matches and target_polygons is None:
        metrics['edge_abs_error_sum'] = sum((sum((abs(predicted[pi][edge] - ground_truth[ti][edge]) for edge in range(4))) for pi, ti, _ in matches))
        metrics['edge_mae_px'] = metrics['edge_abs_error_sum'] / (4 * len(matches))
        metrics['normalized_centroid_distance_sum'] = sum((math.hypot((predicted[pi][0] + predicted[pi][2] - ground_truth[ti][0] - ground_truth[ti][2]) / 2, (predicted[pi][1] + predicted[pi][3] - ground_truth[ti][1] - ground_truth[ti][3]) / 2) / diagonal for pi, ti, _ in matches))
        metrics['normalized_centroid_distance'] = metrics['normalized_centroid_distance_sum'] / len(matches)
    else:
        metrics['edge_abs_error_sum'] = None if target_polygons is not None else 0.0
        metrics['edge_mae_px'] = None
        metrics['normalized_centroid_distance_sum'] = 0.0
        metrics['normalized_centroid_distance'] = None
    exact_count = len(predicted) == len(ground_truth) and bool(ground_truth)
    if target_polygons is not None:
        from shapely.geometry import Polygon
        def grid(polygon):
            center = Polygon(polygon).centroid
            return (min(2, max(0, int(3 * center.y / height))), min(2, max(0, int(3 * center.x / width))))
        grid_ok = exact_count and all(grid(predicted[pi]) == grid(ground_truth[ti]) for pi, ti, _ in matches)
    else:
        grid_ok = exact_count and all((centroid_grid(predicted[pi], width, height) == centroid_grid(ground_truth[ti], width, height) for pi, ti, _ in matches))
    metrics['coarse_grid_success'] = bool(grid_ok)
    metrics['matches'] = [{'prediction_index': pi, 'target_index': ti, 'iou': iou} for pi, ti, iou in matches]
    for threshold in (0.25, 0.5, 0.75):
        suffix = str(int(round(threshold * 100)))
        tp = sum((iou >= threshold for _, _, iou in matches))
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(ground_truth) if ground_truth else 0.0
        metrics[f'tp_iou{suffix}'] = tp
        metrics[f'precision_iou{suffix}'] = precision
        metrics[f'recall_iou{suffix}'] = recall
        metrics[f'f1_iou{suffix}'] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f'all_targets_iou{suffix}'] = bool(exact_count and tp == len(ground_truth))
    return metrics

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Malformed JSONL at {path}:{line_number}') from exc
            if not isinstance(value, dict):
                raise ValueError(f'Non-object JSONL row at {path}:{line_number}')
            rows.append(value)
    return rows

def clean_resume_jsonl(path: Path, expected: Mapping[str, Any]) -> set[str]:
    if not path.exists():
        return set()
    entries: List[Tuple[str, Dict[str, Any], bytes]] = []
    needs_cleanup = False
    with path.open('rb') as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode('utf-8')
            except UnicodeDecodeError:
                needs_cleanup = True
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                needs_cleanup = True
                continue
            if not isinstance(row, dict):
                needs_cleanup = True
                continue
            key = row.get('sample_key')
            if key is None:
                needs_cleanup = True
                continue
            for name, value in expected.items():
                if row.get(name) != value:
                    raise ValueError(f'Resume mismatch in {path}: {name}={row.get(name)!r}, expected {value!r}')
            serialized = raw_line
            if not serialized.endswith(b'\n'):
                serialized += b'\n'
                needs_cleanup = True
            entries.append((str(key), row, serialized))
    latest_index: Dict[str, int] = {}
    for index, (key, _, _) in enumerate(entries):
        if key in latest_index:
            needs_cleanup = True
        latest_index[key] = index
    if not needs_cleanup:
        return set(latest_index)
    payload = b''.join((serialized for index, (key, _, serialized) in enumerate(entries) if latest_index[key] == index))
    temporary = path.with_name(path.name + '.cleaning')
    with temporary.open('wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return set(latest_index)

