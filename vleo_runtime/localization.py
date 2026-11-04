"""Localization."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXEL_INSTRUCTION = (
    "Use actual pixel coordinates of the original image. Do not rescale or normalize "
    "coordinates to 0–1000 or 0–1. The origin (0, 0) is at the image's top-left corner; "
    "x increases to the right and y increases downward."
)


def localization_prompt(task, width, height, description):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    prefix = f"You are given a satellite image that is {width} pixels wide and {height} pixels high. "
    task = str(task).strip().casefold()
    if task in {"dior", "dior-refdet", "dior_localization", "dior-rsvg"}:
        instruction = "Locate the object described below and return its axis-aligned bounding box."
        representation = (
            "Represent the box as [xmin, ymin, xmax, ymax], where (xmin, ymin) is its "
            "top-left corner and (xmax, ymax) is its bottom-right corner, both in pixels."
        )
        output = (
            "Return only JSON in the following format, replacing the placeholders with numbers. "
            "Do not include explanations or Markdown:\n"
            '{"bbox": [xmin, ymin, xmax, ymax]}'
        )
    elif task in {"fair1m", "fair1m localization"}:
        instruction = ("Locate every object matching the description below and return one tight, "
                       "oriented rectangular bounding box aligned with each object.")
        representation = (
            "Represent each box by the four distinct corners of the same rotated rectangle:\n"
            "[[x1, y1], [x2, y2], [x3, y3], [x4, y4]].\n"
            "Each [xi, yi] is one corner's pixel coordinate. List the corners consecutively "
            "around the rectangle, either clockwise or counterclockwise, starting at any corner. "
            "Do not repeat the first corner at the end."
        )
        output = (
            'Return only a JSON object with the key "bboxes", whose value is a list containing '
            "one four-corner box per matching object. Do not include extra objects, explanations, or Markdown:\n"
            '{"bboxes": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]}'
        )
    elif task in {"dota", "dota localization"}:
        instruction = ("Locate every object matching the description below and return one "
                       "axis-aligned bounding box per object.")
        representation = (
            "Represent each box as [xmin, ymin, xmax, ymax], where (xmin, ymin) is its "
            "top-left corner and (xmax, ymax) is its bottom-right corner, both in pixels."
        )
        output = (
            'Return only a JSON object with the key "bboxes", whose value is a list containing '
            "one box per matching object. Do not include extra objects, explanations, or Markdown:\n"
            '{"bboxes": [[xmin, ymin, xmax, ymax]]}'
        )
    else:
        raise ValueError(f"Unsupported localization task: {task}")
    return f"{prefix}{instruction}\n\n{PIXEL_INSTRUCTION}\n\n{representation}\n\nDescription: {description}\n\n{output}"


@lru_cache(maxsize=1)
def source_map():
    path = Path(__import__("os").environ.get("GEOBENCH_SOURCE_METADATA", ROOT / "raw_dataset/geobench_spatial/source_metadata.json"))
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text())
    result = {}
    for row in rows:
        key = int(row['question_id'])
        if key in result or row.get('dataset') not in {'DOTA','FAIR1M','DIOR'} or not row.get('image_path'):
            raise ValueError('Invalid or duplicate dataset provenance entry')
        result[key] = row
    return result


def source_metadata(row):
    identifier = next((row[k] for k in ("question_id", "source_index", "source_id", "sample_key", "sample_id")
                       if row.get(k) is not None), None)
    if isinstance(identifier, str) and identifier.startswith("refdet:"):
        identifier = identifier.split(":")[1]
    try:
        meta = source_map().get(int(identifier))
    except (ValueError, TypeError):
        return None
    if meta is None:
        return None
    path = row.get("image_path") or next(iter(row.get("image_paths") or []), None)
    if path and Path(str(path)).name != Path(meta["image_path"]).name:
        return None
    return meta


def source_dataset(row):
    explicit = row.get("source_dataset") or row.get("dataset") or (row.get("provenance") or {}).get("dataset")
    if explicit in {"DOTA", "FAIR1M", "DIOR"}:
        return explicit
    meta = source_metadata(row)
    return meta["dataset"] if meta else None


def quadrilateral(value):
    from shapely.geometry import Polygon
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A bounding box requires four distinct corners")
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or
           any(type(c) not in (int, float) or not math.isfinite(c) for c in p) for p in value):
        raise ValueError("Invalid corner coordinates")
    points = [[float(x), float(y)] for x, y in value]
    polygon = Polygon(points)
    if len({tuple(p) for p in points}) != 4 or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Invalid or degenerate bounding box")
    if not math.isclose(polygon.area, polygon.convex_hull.area, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Bounding box must be convex")
    return points


def polygon_boxes(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        if set(value) != {"bboxes"}:
            raise ValueError("Expected the bboxes field")
        value = value["bboxes"]
    if not isinstance(value, list):
        raise ValueError("Bounding boxes must be a list")
    return [quadrilateral(box) for box in value]


def polygon_iou(left, right):
    from shapely.geometry import Polygon
    a, b = Polygon(quadrilateral(left)), Polygon(quadrilateral(right))
    intersection = a.intersection(b).area
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def ground_truth_polygons(row):
    value = row.get("ground_truth_polygons") or row.get("ground_truth") or row.get("ground_truth_bboxes")
    return polygon_boxes(value)


def image_dimensions(row, root=ROOT):
    width, height = row.get('image_width'), row.get('image_height')
    if width is not None and height is not None:
        return width, height
    from PIL import Image
    name = row.get('image_path') or row.get('path') or next(iter(row.get('image_paths') or []), None)
    if not name:
        raise ValueError('Image dimensions are required')
    path = Path(name)
    with Image.open(path if path.is_absolute() else Path(root) / path) as image:
        return image.size
