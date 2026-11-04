"""Dataset catalog."""
from __future__ import annotations
import importlib
from pathlib import Path

TASK_MODULES = {
    "landmarks": "landmarks", "pattern-net": "pattern_net", "cowc": "cowc",
    "aerial-animal": "aerial_animal", "neon-trees": "neon_trees", "xview2": "xview2",
    "dior-recognition": "dior_rsvg_recognition", "dior-refdet": "dior_rsvg_localization",
    "geobench-refdet": "geobench_localization", "rare-planes": "rare_planes", "fpcd": "fpcd",
}

def dataset_rows(task, cache_root=Path("raw_dataset")):

    import sys
    root = Path(__file__).resolve().parents[2]
    for directory in (root, root / "evaluation/utils", root / "evaluation/tasks"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    m = importlib.import_module(TASK_MODULES[task])
    backend = m.dior if task == "dior-recognition" else m
    configure = getattr(backend, "configure_cache_environment_from_args", None)
    if configure:
        configure(str(cache_root))
    if task == "geobench-refdet":
        from evaluation.tasks.create_geobench_refdet_manifest import partition_rows
        data_root, manifest = m.geo.load_manifest(str(Path(cache_root) / "geobench_spatial"))
        rows = m.build_items(manifest, "refdet")
        chosen = set(partition_rows(rows)["test"])
        rows = [dict(row, official_prompt=row["prompt"], image_root=str(data_root))
                for row in rows if row["sample_id"] in chosen]
    elif task == "fpcd":
        m.pin_canonical_fpcd_source()
        bundle = m.build_fpcd_bundle(Path(cache_root), strict=True)
        rows = [dict(source_id=r.source_id, sample_id=r.source_id, ground_truth=r.answer,
                     raw_split="test", canonical_split="test") for r in bundle.records("test")]
    elif task == "rare-planes":
        _, rows = m.download_and_prepare_dataset("test")
    elif task == "cowc":
        rows = []
        for city in m.DEFAULT_CITIES:
            payload = m.load_canonical_city_payload(city)
            assignment = payload["assignments"]["test"]
            allowed = set(assignment["indices"])
            rows.extend(dict(item, city=city, raw_split=assignment["raw_split"], canonical_split="test")
                        for item in payload["metadata_items"] if item["index"] in allowed)
    else:
        options = {"few_shot_k": 0}
        if task == "dior-refdet":
            options["task"] = "detection"
        rows = m.resolve_canonical_pools(**options)["evaluation_items"]
    result = []
    for original in rows:
        row = dict(original)
        if task == "landmarks":
            row["options"] = m.build_landmark_options(row)
            if not row["options"]:
                continue
        if task == "xview2" and "gt" not in row:
            row["gt"] = m.parse_ground_truth(row["objects1"], row["objects2"])
        if row.get("city") is not None:
            identity = f"{row['city']}:{row['index']}"
        else:
            identity = next((str(row[k]) for k in ("source_id", "sample_id", "sample_key", "question_id", "index") if row.get(k) is not None), None)
        if identity is None:
            raise ValueError(f"Missing dataset identity: {task}")
        row.update(sample_id=identity, canonical_split="test")
        result.append(row)
    result.sort(key=lambda row: row["sample_id"])
    if not result or len({r["sample_id"] for r in result}) != len(result):
        raise ValueError(f"Empty or duplicate dataset identities: {task}")
    return result
