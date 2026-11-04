"""Schema."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SplitStrategy(str, Enum):
    REUSE_NATIVE_VAL = "reuse_native_train_val_test"
    REUSE_NATIVE_VALIDATION = "reuse_native_train_validation_test"
    SINGLE_SOURCE = "single_source_80_10_10"
    CITY_STRATIFIED = "city_stratified_single_source_80_10_10"
    TRAIN_TEST = "resplit_native_train_90_10_keep_native_test"
    POOL_NATIVE_EQUAL_THIRDS = "pool_native_train_validation_test_seed0_equal_thirds"
    RARE_PLANES_REAL_SCENE_DISJOINT = (
        "rareplanes_real_scene_disjoint"
    )


class MetricName(str, Enum):
    TOTAL_COUNT_MAE = "total_count_mae"
    COUNT_MAE = "count_mae"
    ACCURACY = "accuracy"
    MEAN_IOU = "mean_iou"
    WATER_IOU = "water_iou"
    FIVE_FIELD_MACRO_MAE = "five_field_macro_mae"
    R2 = "r2"


class MetricAggregation(str, Enum):
    EQUAL_WEIGHT = "equal_weight_arithmetic_mean"
    # xView2: mean of building-counting r^2 (count_before) and change-detection
    # macro r^2 over the four damage categories.
    EQUAL_WEIGHT_BENCHMARK_TASKS = "equal_weight_benchmark_tasks"


class ParseFailurePolicy(str, Enum):
    ZERO_PREDICTION = "zero_prediction"
    INCORRECT = "incorrect"
    REFUSAL_TIEBREAK = "refusal_tiebreak"
    ZERO_PER_FIELD = "zero_per_field"


class AnswerEncoding(str, Enum):
    JSON = "json"
    PLAIN_TEXT = "plain_text"


class LifecycleStatus(str, Enum):
    NOT_STARTED = "not_started"
    PROVISIONAL_EXISTING = "provisional_existing"
    BLOCKED = "blocked"
    READY = "ready"
    COMPLETED = "completed"


class RegistryReference(StrictModel):
    path: Path


class ScopeConfig(StrictModel):
    experiment: str = Field(min_length=1)
    few_shot_k: int = Field(ge=0)
    complete_canonical_test_split: bool
    test_based_tuning: bool
    allowed_tasks: tuple[str, ...] | None = None


class BaseModelConfig(StrictModel):
    name: str = Field(min_length=1)
    path: Path
    architecture: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    identity_files: tuple[str, ...]

    @model_validator(mode="after")
    def validate_identity_files(self) -> "BaseModelConfig":
        expected = {
            "config.json",
            "model.safetensors.index.json",
            "preprocessor_config.json",
        }
        if set(self.identity_files) != expected or len(self.identity_files) != len(expected):
            raise ValueError(f"identity_files must contain exactly {sorted(expected)}")
        return self


class PathTemplates(StrictModel):
    repository_root: Path
    raw_dataset_root: Path
    prepared_data_template: str = Field(min_length=1)
    temporary_trial_template: str = Field(min_length=1)
    final_adapter_template: str = Field(min_length=1)
    permanent_result_root: Path
    validation_result_template: str = Field(min_length=1)
    test_result_template: str = Field(min_length=1)
    manifest_template: str = Field(min_length=1)


class HardwareConfig(StrictModel):
    gpu_type: str = Field(min_length=1)
    gpus_per_trial: int = Field(gt=0)
    model_parallel_size: int = Field(default=1, gt=0)
    required_memory_headroom_fraction: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_parallel_size(self) -> "HardwareConfig":
        if self.gpus_per_trial % self.model_parallel_size:
            raise ValueError("gpus_per_trial must be divisible by model_parallel_size")
        return self


class QuantizationConfig(StrictModel):
    bits: int = Field(gt=0)
    method: str = Field(min_length=1)
    quant_type: str = Field(min_length=1)
    double_quantization: bool
    compute_dtype: str = Field(min_length=1)


class LoraConfig(StrictModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: str = Field(min_length=1)
    freeze_vision_encoder: bool


class BatchConfig(StrictModel):
    per_device_train_batch_size: int = Field(gt=0)
    per_device_eval_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    effective_global_batch_size: int = Field(gt=0)


class OptimizerConfig(StrictModel):
    name: str = Field(min_length=1)
    weight_decay: float = Field(ge=0.0)
    beta1: float = Field(gt=0.0, lt=1.0)
    beta2: float = Field(gt=0.0, lt=1.0)
    epsilon: float = Field(gt=0.0)
    max_grad_norm: float = Field(gt=0.0)


class SchedulerConfig(StrictModel):
    name: str = Field(min_length=1)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)


class TrainingConfig(StrictModel):
    method: str = Field(min_length=1)
    loss_scope: str = Field(min_length=1)
    mask_non_assistant_tokens: bool
    truncate_gold_answer: bool
    overflow_action: str = Field(min_length=1)
    max_length: int = Field(gt=0)
    attention_implementation: str = Field(min_length=1)
    quantization: QuantizationConfig
    lora: LoraConfig
    batch: BatchConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    gradient_checkpointing: bool
    data_augmentation: str = Field(min_length=1)
    seed: int = Field(ge=0)
    data_seed: int = Field(ge=0)
    additional_training_seeds: tuple[int, ...] = ()


class LearningRateSearch(StrictModel):
    initial_values: tuple[float, ...] = Field(min_length=1)
    lower_boundary_value: float = Field(gt=0.0)
    lower_expansion_value: float = Field(gt=0.0)
    upper_boundary_value: float = Field(gt=0.0)
    upper_expansion_value: float = Field(gt=0.0)
    # One expansion candidate is added when the successive-halving winner is a
    # boundary value of the initial grid (Supplementary Material S.3-C).
    maximum_expansion_rounds: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "LearningRateSearch":
        if len(set(self.initial_values)) != len(self.initial_values):
            raise ValueError("initial learning rates must be unique")
        if tuple(sorted(self.initial_values)) != self.initial_values:
            raise ValueError("initial learning rates must be sorted")
        if self.lower_boundary_value != self.initial_values[0]:
            raise ValueError("lower_boundary_value must equal the first initial value")
        if self.upper_boundary_value != self.initial_values[-1]:
            raise ValueError("upper_boundary_value must equal the last initial value")
        if not self.lower_expansion_value < self.lower_boundary_value:
            raise ValueError("lower expansion must be below the initial grid")
        if not self.upper_expansion_value > self.upper_boundary_value:
            raise ValueError("upper expansion must be above the initial grid")
        return self


class DurationSearch(StrictModel):
    validation_interval_epochs: int = Field(gt=0)
    initial_max_epochs: int = Field(gt=0)
    extended_max_epochs: int = Field(gt=0)
    extend_when_best_checkpoint_at_epoch: int = Field(gt=0)
    extend_only_same_learning_rate_trial: bool
    extend_beyond_epoch_10: bool

    @model_validator(mode="after")
    def validate_epochs(self) -> "DurationSearch":
        if self.extended_max_epochs < self.initial_max_epochs:
            raise ValueError("extended_max_epochs must not be below initial_max_epochs")
        if self.extend_when_best_checkpoint_at_epoch != self.initial_max_epochs:
            raise ValueError("extension boundary must equal initial_max_epochs")
        return self


class EarlyStoppingConfig(StrictModel):
    patience_validation_events: int = Field(ge=0)
    min_delta: float = Field(ge=0.0)
    improvement_must_be_strict: bool


class SuccessiveHalvingConfig(StrictModel):

    enabled: bool
    budget_unit: str = Field(min_length=1)
    rung_steps: tuple[int, ...] = Field(min_length=3)
    survivors_after_rung: tuple[int, ...] = Field(min_length=3)
    screening_sampling: str = Field(min_length=1)
    screening_rungs_use_fixed_validation_subset: bool
    # Screening rungs are scored on the first N validation samples in a
    # deterministic ordering (at most 256 in the paper).
    screening_validation_max_examples: int = Field(gt=0)
    restart_final_training_from_base: bool
    final_training_sampling: str = Field(min_length=1)
    final_training_max_examples: int | None = Field(default=None, gt=0)
    final_training_max_optimizer_steps: int = Field(gt=0)
    maximum_optimizer_step_equivalents_per_task: int = Field(gt=0)
    final_training_uses_complete_validation: bool
    final_training_max_epochs: int = Field(gt=0)
    final_training_scheduler_horizon_epochs: int = Field(gt=0)
    final_training_patience_validation_events: int = Field(ge=0)
    target_wall_clock_compute_hours: float = Field(gt=0.0)
    queue_time_excluded_from_target: bool

    @model_validator(mode="after")
    def validate_successive_halving(self) -> "SuccessiveHalvingConfig":
        if self.budget_unit != "optimizer_steps":
            raise ValueError("successive-halving budget_unit must be optimizer_steps")
        if tuple(sorted(set(self.rung_steps))) != self.rung_steps:
            raise ValueError("successive-halving rung_steps must be unique and increasing")
        if len(self.survivors_after_rung) != len(self.rung_steps):
            raise ValueError("survivors_after_rung must match rung_steps length")
        if any(value <= 0 for value in self.survivors_after_rung):
            raise ValueError("successive-halving survivor counts must be positive")
        if any(
            right > left
            for left, right in zip(
                self.survivors_after_rung, self.survivors_after_rung[1:]
            )
        ):
            raise ValueError("successive-halving survivor counts must not increase")
        if any(
            left > 1 and right >= left
            for left, right in zip(
                self.survivors_after_rung, self.survivors_after_rung[1:]
            )
        ):
            raise ValueError("each active successive-halving rung must eliminate a candidate")
        if self.survivors_after_rung[-1] != 1:
            raise ValueError("the final successive-halving rung must select one LR")
        if self.screening_sampling != "deterministic_order":
            raise ValueError("screening_sampling must be deterministic_order")
        if not self.screening_rungs_use_fixed_validation_subset:
            raise ValueError("every screening rung must use one fixed validation subset")
        if self.screening_validation_max_examples > 256:
            raise ValueError("the screening validation subset holds at most 256 samples")
        if not self.restart_final_training_from_base:
            raise ValueError("the full-data final run must restart from the frozen Base")
        if self.final_training_sampling != "deterministic_stratified":
            raise ValueError(
                "final_training_sampling must be deterministic_stratified"
            )
        if not self.final_training_uses_complete_validation:
            raise ValueError("final training must use complete canonical validation")
        if self.rung_steps[-1] > 128:
            raise ValueError("screening may use at most 128 optimizer steps")
        if self.final_training_max_optimizer_steps > 1024:
            raise ValueError("final training may use at most 1024 optimizer steps")
        if self.final_training_max_epochs != 1:
            raise ValueError("compute-bounded final training is exactly one epoch")
        if self.final_training_patience_validation_events != 0:
            raise ValueError("one-epoch final training does not use early-stopping patience")
        if (
            self.final_training_scheduler_horizon_epochs
            != self.final_training_max_epochs
        ):
            raise ValueError("final scheduler horizon must equal final max epochs")
        if not self.queue_time_excluded_from_target:
            raise ValueError("the compute target must explicitly exclude runtime queue time")
        return self


class SearchConfig(StrictModel):
    searchable_parameters: tuple[str, ...]
    forbidden_search_parameters: tuple[str, ...]
    learning_rate: LearningRateSearch
    duration: DurationSearch
    early_stopping: EarlyStoppingConfig
    successive_halving: SuccessiveHalvingConfig | None = None
    global_tie_break_order: tuple[str, ...]
    tune_each_task_independently: bool
    aggregate_rank_across_tasks: bool

    @model_validator(mode="after")
    def validate_search_policy(self) -> "SearchConfig":
        policy = self.successive_halving
        if policy is None or not policy.enabled:
            return self
        if policy.survivors_after_rung[0] > len(self.learning_rate.initial_values):
            raise ValueError("first rung cannot retain more LR trials than were started")
        if len(self.learning_rate.initial_values) != 3:
            raise ValueError("successive halving requires exactly three initial LRs")
        if self.learning_rate.maximum_expansion_rounds > 1:
            raise ValueError("successive halving permits at most one boundary expansion")
        if tuple(self.searchable_parameters) != ("learning_rate",):
            raise ValueError("validation-based selection is restricted to the learning rate")
        increments = (
            policy.rung_steps[0],
            *(
                right - left
                for left, right in zip(policy.rung_steps, policy.rung_steps[1:])
            ),
        )
        active_trials = (
            len(self.learning_rate.initial_values),
            *policy.survivors_after_rung[:-1],
        )
        screening_steps = sum(
            count * increment
            for count, increment in zip(active_trials, increments)
        )
        boundary_steps = (
            policy.rung_steps[-1]
            if self.learning_rate.maximum_expansion_rounds
            else 0
        )
        maximum_steps = (
            screening_steps
            + boundary_steps
            + policy.final_training_max_optimizer_steps
        )
        if maximum_steps != policy.maximum_optimizer_step_equivalents_per_task:
            raise ValueError("successive-halving total optimizer-step budget is inconsistent")
        return self


class ResolutionCandidate(StrictModel):
    id: str = Field(min_length=1)
    max_pixels_per_image: int = Field(gt=0)
    max_visual_tokens_per_image: int | None = Field(default=None, gt=0)


class ResolutionConfig(StrictModel):
    selection_basis: str = Field(min_length=1)
    statistics_splits: tuple[str, ...]
    inspect_test_split: bool
    preserve_aspect_ratio: bool
    crop: bool
    tile: bool
    stretch: bool
    offline_resize: bool
    augmentation: bool
    statistics: tuple[str, ...]
    stress_examples: tuple[str, ...]
    required_checks: tuple[str, ...]
    candidates_high_to_low: tuple[ResolutionCandidate, ...] = Field(min_length=1)
    choose_highest_stable_candidate: bool
    required_memory_headroom_fraction: float = Field(ge=0.0, lt=1.0)
    validation_performance_must_not_be_used: bool
    test_properties_must_not_be_used: bool
    selected_value_fixed_across_task_trials_and_test: bool

    @model_validator(mode="after")
    def validate_candidates(self) -> "ResolutionConfig":
        ids = [candidate.id for candidate in self.candidates_high_to_low]
        pixels = [candidate.max_pixels_per_image for candidate in self.candidates_high_to_low]
        if len(set(ids)) != len(ids):
            raise ValueError("resolution candidate ids must be unique")
        if pixels != sorted(pixels, reverse=True):
            raise ValueError("resolution candidates must be ordered from highest to lowest")
        if self.inspect_test_split:
            raise ValueError("resolution inspection must not use test")
        return self


class TestPolicy(StrictModel):
    eligible_only_after_validation_selection_and_adapter_promotion: bool
    eligible_only_after_evaluator_conformance: bool
    one_final_result_per_task: bool
    complete_canonical_split_required: bool
    subsampling_allowed: bool
    interrupted_run_may_resume_same_jsonl: bool
    interrupted_run_may_resample: bool
    new_result_after_partial_failure: bool
    completion_marker: str = Field(min_length=1)
    completion_marker_atomic_write: bool
    completion_marker_required_fields: tuple[str, ...] = Field(min_length=1)
    parameter_changes_after_completion_forbidden: bool
    evaluation_scope: str = "complete_canonical_split"

    @model_validator(mode="after")
    def validate_test_only_once(self) -> "TestPolicy":
        if not self.eligible_only_after_validation_selection_and_adapter_promotion:
            raise ValueError("test requires validation selection and adapter promotion")
        if not self.eligible_only_after_evaluator_conformance:
            raise ValueError("test requires evaluator conformance")
        if not self.one_final_result_per_task:
            raise ValueError("exactly one final test result per task is required")
        if self.evaluation_scope == "complete_canonical_split":
            if not self.complete_canonical_split_required or self.subsampling_allowed:
                raise ValueError("complete test scope cannot permit subsampling")
        else:
            raise ValueError(f"unsupported test evaluation_scope={self.evaluation_scope!r}")
        if not self.interrupted_run_may_resume_same_jsonl:
            raise ValueError("an interrupted test must resume the same JSONL")
        if self.interrupted_run_may_resample or self.new_result_after_partial_failure:
            raise ValueError("an interrupted test cannot resample or create a second result")
        if not self.completion_marker_atomic_write:
            raise ValueError("the test completion marker must be written atomically")
        if len(set(self.completion_marker_required_fields)) != len(
            self.completion_marker_required_fields
        ):
            raise ValueError("completion marker required fields must be unique")
        if not self.parameter_changes_after_completion_forbidden:
            raise ValueError("parameter changes after test completion are forbidden")
        return self


class BaseLoraConfig(StrictModel):
    registry: RegistryReference
    scope: ScopeConfig
    base_model: BaseModelConfig
    paths: PathTemplates
    hardware: HardwareConfig
    training: TrainingConfig
    search: SearchConfig
    resolution: ResolutionConfig
    test_policy: TestPolicy

    @model_validator(mode="after")
    def validate_batch_and_scope(self) -> "BaseLoraConfig":
        expected_batch = (
            self.training.batch.per_device_train_batch_size
            * self.training.batch.gradient_accumulation_steps
            * self.hardware.gpus_per_trial
            // self.hardware.model_parallel_size
        )
        if self.training.batch.effective_global_batch_size != expected_batch:
            raise ValueError(
                "effective_global_batch_size must equal train micro-batch * "
                "gradient accumulation * data-parallel replica count"
            )
        if self.scope.few_shot_k != 0:
            raise ValueError("Task-LoRA evaluation is frozen to zero-shot")
        complete_scope = self.test_policy.evaluation_scope == "complete_canonical_split"
        if self.scope.complete_canonical_test_split != complete_scope:
            raise ValueError("scope and test_policy evaluation scope disagree")
        if self.scope.test_based_tuning:
            raise ValueError("test-based tuning is forbidden")
        policy = self.search.successive_halving
        if policy is not None and policy.enabled:
            if policy.final_training_max_examples is None:
                object.__setattr__(policy, 'final_training_max_examples',
                                   policy.final_training_max_optimizer_steps
                                   * self.training.batch.effective_global_batch_size)
            maximum_steps = (
                policy.final_training_max_examples
                + self.training.batch.effective_global_batch_size
                - 1
            ) // self.training.batch.effective_global_batch_size
            if maximum_steps != policy.final_training_max_optimizer_steps:
                raise ValueError(
                    "final training example and optimizer-step caps disagree with batch size"
                )
        return self


class DatasetIdentity(StrictModel):
    id: str = Field(min_length=1)
    config: str | None = None
    configs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_config_choice(self) -> "DatasetIdentity":
        if bool(self.config) == bool(self.configs):
            raise ValueError("exactly one of dataset.config or dataset.configs is required")
        if len(set(self.configs)) != len(self.configs):
            raise ValueError("dataset.configs must be unique")
        return self


class CanonicalSplitConfig(StrictModel):
    source_split: str = Field(min_length=1)


class DatasetSplitConfig(StrictModel):
    strategy: SplitStrategy
    seed: int = Field(ge=0)
    train: CanonicalSplitConfig
    validation: CanonicalSplitConfig
    test: CanonicalSplitConfig

    @model_validator(mode="after")
    def reject_test_as_validation(self) -> "DatasetSplitConfig":
        validation_source = self.validation.source_split.strip().casefold()
        if validation_source in {"test", "evaluation"}:
            raise ValueError("validation.source_split cannot be test or evaluation")

        sources = (
            self.train.source_split,
            self.validation.source_split,
            self.test.source_split,
        )
        if self.strategy in {
            SplitStrategy.REUSE_NATIVE_VAL,
            SplitStrategy.REUSE_NATIVE_VALIDATION,
        }:
            if len(set(sources)) != 3:
                raise ValueError("native reuse strategies require three distinct source splits")
        elif self.strategy in {
            SplitStrategy.SINGLE_SOURCE,
            SplitStrategy.CITY_STRATIFIED,
        }:
            if len(set(sources)) != 1:
                raise ValueError("single-source strategies require one shared raw split")
        elif self.strategy is SplitStrategy.TRAIN_TEST:
            if self.train.source_split != self.validation.source_split:
                raise ValueError("train/test strategy must split train into train and validation")
            if self.test.source_split == self.validation.source_split:
                raise ValueError("train/test strategy must retain a distinct native test split")
        elif self.strategy is SplitStrategy.POOL_NATIVE_EQUAL_THIRDS:
            if len(set(sources)) != 1:
                raise ValueError("pooled equal-thirds strategy requires one shared source pool")
        return self


class PromptConfig(StrictModel):
    authoritative_script: Path
    system_symbol: str = Field(min_length=1)
    user_symbol: str = Field(min_length=1)
    dynamic: bool
    dynamic_builder_symbols: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_dynamic_builder_symbols(self) -> "PromptConfig":
        if len(set(self.dynamic_builder_symbols)) != len(self.dynamic_builder_symbols):
            raise ValueError("dynamic_builder_symbols must be unique")
        return self


class DatasetInputConfig(StrictModel):
    image_count: int = Field(gt=0)
    image_order: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_image_order(self) -> "DatasetInputConfig":
        if len(self.image_order) != self.image_count:
            raise ValueError("image_order length must equal image_count")
        if len(set(self.image_order)) != len(self.image_order):
            raise ValueError("image_order values must be unique")
        return self


class DatasetGenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=128, gt=0)
    do_sample: bool = False

    @model_validator(mode="after")
    def validate_greedy_generation(self) -> "DatasetGenerationConfig":
        if self.max_new_tokens not in {128, 1024, 2048}:
            raise ValueError("generation supports max_new_tokens=128, 1024, or 2048")
        if self.do_sample:
            raise ValueError("formal validation and test generation must be greedy")
        return self


class AssistantAnswerConfig(StrictModel):
    encoding: AnswerEncoding
    answer_schema: str = Field(alias="schema", min_length=1)
    required_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_required_keys(self) -> "AssistantAnswerConfig":
        if len(set(self.required_keys)) != len(self.required_keys):
            raise ValueError("assistant required_keys must be unique")
        if self.encoding is AnswerEncoding.JSON and not self.required_keys:
            raise ValueError("JSON assistant answers require at least one required key")
        if self.encoding is AnswerEncoding.PLAIN_TEXT and self.required_keys:
            raise ValueError("plain-text assistant answers cannot declare JSON keys")
        return self


class ValidationMetricConfig(StrictModel):
    metric: MetricName
    direction: Direction
    fields: tuple[str, ...] = ()
    aggregation: MetricAggregation | None = None
    parse_failure_policy: ParseFailurePolicy

    @model_validator(mode="after")
    def validate_metric_contract(self) -> "ValidationMetricConfig":
        minimize = {
            MetricName.TOTAL_COUNT_MAE,
            MetricName.COUNT_MAE,
            MetricName.FIVE_FIELD_MACRO_MAE,
        }
        expected_direction = Direction.MINIMIZE if self.metric in minimize else Direction.MAXIMIZE
        if self.direction is not expected_direction:
            raise ValueError(f"{self.metric.value} must use direction={expected_direction.value}")

        if self.metric is MetricName.FIVE_FIELD_MACRO_MAE:
            if not self.fields or len(set(self.fields)) != len(self.fields):
                raise ValueError("five-field Macro-MAE requires non-empty unique fields")
            if self.aggregation is not MetricAggregation.EQUAL_WEIGHT:
                raise ValueError("five-field Macro-MAE requires equal-weight aggregation")
        elif self.metric is MetricName.R2 and (self.fields or self.aggregation is not None):
            if not self.fields or len(set(self.fields)) != len(self.fields):
                raise ValueError("multi-field r2 requires non-empty unique fields")
            if self.aggregation is not MetricAggregation.EQUAL_WEIGHT_BENCHMARK_TASKS:
                raise ValueError("multi-field r2 requires equal weighting across benchmark tasks")
        elif self.fields or self.aggregation is not None:
            raise ValueError("fields and aggregation are only valid for multi-field metrics")
        return self


class DatasetPaths(StrictModel):
    raw_dataset_root: Path
    prepared_dir: Path
    final_adapter_dir: Path
    validation_result_dir: Path
    test_result_dir: Path
    manifest_dir: Path


class DatasetStatus(StrictModel):
    data_preparation: LifecycleStatus
    resolution_selection: LifecycleStatus
    training: LifecycleStatus
    validation: LifecycleStatus
    test: LifecycleStatus


class DatasetConfig(StrictModel):
    task: str = Field(min_length=1)
    adapter_name: str = Field(min_length=1)
    dataset: DatasetIdentity
    splits: DatasetSplitConfig
    prompt: PromptConfig
    input: DatasetInputConfig
    generation: DatasetGenerationConfig = DatasetGenerationConfig()
    assistant_answer: AssistantAnswerConfig
    validation: ValidationMetricConfig
    paths: DatasetPaths
    status: DatasetStatus

    @model_validator(mode="after")
    def validate_task_paths(self) -> "DatasetConfig":
        if self.adapter_name != self.task:
            raise ValueError("adapter_name must equal the canonical task name")
        if self.paths.final_adapter_dir.name != self.adapter_name:
            raise ValueError("final_adapter_dir must end with adapter_name")
        return self


class ResolvedExperimentConfig(StrictModel):
    base: BaseLoraConfig
    dataset: DatasetConfig
    registry_task: dict[str, Any]
