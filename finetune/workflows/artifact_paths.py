"""Artifact paths."""

from __future__ import annotations

from pathlib import Path

from finetune.workflows.selection import learning_rate_slug


def trial_root(base_config, *, task: str) -> Path:
    probe = Path(
        base_config.paths.temporary_trial_template.format(
            task=task,
            learning_rate=learning_rate_slug(1e-5),
            seed=0,
            adapter_name=task,
        )
    ).expanduser().resolve()
    return probe.parent


def validation_root(base_config, *, task: str) -> Path:
    return Path(
        base_config.paths.validation_result_template.format(
            task=task,
            learning_rate=learning_rate_slug(1e-5),
            seed=0,
            adapter_name=task,
        )
    ).expanduser().resolve()


def _render(template: str, *, task: str, adapter_name: str) -> Path:
    try:
        return Path(
            template.format(
                task=task,
                adapter_name=adapter_name,
                learning_rate=learning_rate_slug(1e-5),
                seed=0,
            )
        ).expanduser().resolve()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid model artifact template for task {task!r}") from exc


def model_artifact_paths(base, dataset):
    if base.base_model.model_type.casefold() not in {"llava_onevision2"}:
        return dataset
    from finetune.config.schema import DatasetPaths

    task = dataset.task
    final_adapter_template = base.paths.final_adapter_template


    if base.base_model.model_type.casefold() == "llava_onevision2" and dataset.task in {"dior-rsvg-recognition", "geobench-refdet"}:
        final_adapter_template = "./finetuned_llava/{adapter_name}"
    paths = DatasetPaths.model_validate(
        {
            "raw_dataset_root": dataset.paths.raw_dataset_root,
            "prepared_dir": dataset.paths.prepared_dir,
            "final_adapter_dir": _render(
                final_adapter_template,
                task=task,
                adapter_name=dataset.adapter_name,
            ),
            "validation_result_dir": _render(
                base.paths.validation_result_template,
                task=task,
                adapter_name=dataset.adapter_name,
            ),
            "test_result_dir": _render(
                base.paths.test_result_template,
                task=task,
                adapter_name=dataset.adapter_name,
            ),
            "manifest_dir": _render(
                base.paths.manifest_template,
                task=task,
                adapter_name=dataset.adapter_name,
            ),
        }
    )
    return dataset.model_copy(update={"paths": paths})


def resolution_manifest_root(
    base_config,
    *,
    task: str,
    dataset_manifest_dir: str | Path,
) -> Path:
    if str(getattr(base_config.base_model, "model_type", "")).casefold() in {"llava_onevision2"}:
        return _render(base_config.paths.manifest_template, task=task, adapter_name=task)
    return Path(dataset_manifest_dir).expanduser().resolve()


def resolution_statistics_path(
    base_config,
    *,
    task: str,
    dataset_manifest_dir: str | Path,
) -> Path:
    return resolution_manifest_root(
        base_config,
        task=task,
        dataset_manifest_dir=dataset_manifest_dir,
    ) / "resolution_statistics.json"


