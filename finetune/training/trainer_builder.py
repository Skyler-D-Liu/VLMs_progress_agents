"""Trainer builder."""
from __future__ import annotations
from pathlib import Path
import inspect
import math
from typing import Any, Mapping, Sequence
from finetune.config.schema import BaseLoraConfig
from finetune.datasets import AdapterExample
from finetune.model_backends import get_model_backend
from .checkpoint_manager import CompleteCheckpointCallback
from .training_dataset import AdapterExampleDataset
from .training_state import TrainingBudgetCallback

def create_training_arguments(*, base_config: BaseLoraConfig, output_dir: str | Path, learning_rate: float, max_steps: int, num_train_epochs: float | None=None, epoch_checkpoints: bool=False, save_steps: int | None=None, save_total_limit: int | None=None):
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise ImportError('Transformers TrainingArguments is unavailable. Run the runtime environment setup job.') from exc
    lr_search = base_config.search.learning_rate
    if learning_rate not in {*lr_search.initial_values, *((lr_search.lower_expansion_value, lr_search.upper_expansion_value) if lr_search.maximum_expansion_rounds else ())}:
        raise ValueError(f'Learning rate {learning_rate} is outside the frozen search grid')
    if epoch_checkpoints:
        if num_train_epochs is None or num_train_epochs <= 0:
            raise ValueError('epoch checkpoint mode requires positive num_train_epochs')
    elif max_steps <= 0:
        raise ValueError('max_steps must be positive')
    if save_steps is not None and save_steps <= 0:
        raise ValueError('save_steps must be positive when provided')
    if save_total_limit is not None and save_total_limit <= 0:
        raise ValueError('save_total_limit must be positive when provided')
    training = base_config.training
    batch = training.batch
    optimizer = training.optimizer
    scheduler = training.scheduler
    training_kwargs = dict(output_dir=str(Path(output_dir).expanduser().resolve()), do_train=True, do_eval=False, eval_strategy='no', per_device_train_batch_size=batch.per_device_train_batch_size, per_device_eval_batch_size=batch.per_device_eval_batch_size, gradient_accumulation_steps=batch.gradient_accumulation_steps, learning_rate=learning_rate, weight_decay=optimizer.weight_decay, adam_beta1=optimizer.beta1, adam_beta2=optimizer.beta2, adam_epsilon=optimizer.epsilon, max_grad_norm=optimizer.max_grad_norm, optim=optimizer.name, lr_scheduler_type=scheduler.name, max_steps=-1 if epoch_checkpoints else max_steps, num_train_epochs=num_train_epochs if epoch_checkpoints else 3.0, logging_strategy='no', save_strategy='epoch' if epoch_checkpoints else 'steps', save_steps=save_steps or 1, save_total_limit=save_total_limit if save_total_limit is not None else int(num_train_epochs) if epoch_checkpoints else max_steps, save_only_model=False, bf16=True, fp16=False, gradient_checkpointing=training.gradient_checkpointing, remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=True, seed=training.seed, data_seed=training.data_seed, report_to='none', push_to_hub=False, disable_tqdm=True, use_cpu=False)
    if 'warmup_ratio' in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs['warmup_ratio'] = scheduler.warmup_ratio
    else:
        training_kwargs['warmup_steps'] = math.ceil(max_steps * scheduler.warmup_ratio) if max_steps > 0 else 0
    return TrainingArguments(**training_kwargs)

def create_lora_trainer(*, base_config: BaseLoraConfig, model: Any, processor: Any, train_examples: Sequence[AdapterExample], output_dir: str | Path, learning_rate: float, max_steps: int, compatibility: Mapping[str, Any], num_train_epochs: float | None=None, epoch_checkpoints: bool=False, budget_steps: int | None=None, extra_callbacks: Sequence[Any]=(), save_steps: int | None=None, save_total_limit: int | None=None):
    try:
        from transformers import Trainer
    except ImportError as exc:
        raise ImportError('Transformers Trainer is unavailable. Run the runtime environment setup job.') from exc
    output_path = Path(output_dir).expanduser().resolve()
    backend = get_model_backend(base_config)
    backend.validate_config(base_config)
    args = create_training_arguments(base_config=base_config, output_dir=output_path, learning_rate=learning_rate, max_steps=max_steps, num_train_epochs=num_train_epochs, epoch_checkpoints=epoch_checkpoints, save_steps=save_steps, save_total_limit=save_total_limit)
    callbacks = [CompleteCheckpointCallback(compatibility)]
    if budget_steps is not None:
        callbacks.append(TrainingBudgetCallback(budget_steps))
    callbacks.extend(extra_callbacks)
    return Trainer(model=model, args=args, train_dataset=AdapterExampleDataset(train_examples), data_collator=backend.create_collator(processor, base_config.training.max_length), processing_class=processor, callbacks=callbacks)
