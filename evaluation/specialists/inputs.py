"""Inputs."""
import base64
import json
from pathlib import Path
from io import BytesIO
from PIL import Image
from evaluation.utils.dataset_catalog import dataset_rows


def load_inputs(task, data_root, cohort=None):
    rows = dataset_rows(task, data_root)
    if task in {'pattern-net','dior-refdet'} and cohort is None:
        raise ValueError(f'{task} specialist evaluation requires the checkpoint held-out input manifest')
    if cohort is not None:
        from evaluation.utils.benchmark_evaluation import identity
        manifest = [json.loads(line) for line in Path(cohort).read_text().splitlines() if line.strip()]
        if any(any(key in row for key in ('response', 'answer_text', 'prediction', 'predictions', 'model')) for row in manifest):
            raise ValueError('Use a dataset input manifest, not predictions')
        chosen = {identity(task, row) for row in manifest}
        available = {identity(task, row) for row in rows}
        if not chosen or len(chosen) != len(manifest) or not chosen <= available:
            raise ValueError('Invalid dataset cohort identities')
        rows = [row for row in rows if identity(task, row) in chosen]
    from evaluation.utils.reasoning_tasks import TaskInputs
    return rows, TaskInputs(task, rows, data_root)


def input_images(inputs, row):
    _, images = inputs.prepare(row)
    return [Image.open(BytesIO(base64.b64decode(uri.split(',', 1)[1]))).convert('RGB') for uri in images]


def input_image(inputs, row):
    images = input_images(inputs, row)
    if len(images) != 1:
        raise ValueError('This specialist requires a single-image task')
    return images[0]
