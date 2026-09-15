from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LORA_CONFIG_PATH = Path("training/configs/lora_config.yaml")
TRAINING_ARGS_PATH = Path("training/configs/training_args.yaml")


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoraConfigContract(FrozenModel):
    config_version: str = Field(min_length=1)
    method: Literal["lora", "qlora"]
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    target_modules: tuple[str, ...] = Field(min_length=1)
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    quantization: Literal["none", "bitsandbytes_4bit_nf4"]
    compute_dtype: Literal["float16", "bfloat16", "float32"]
    use_double_quant: bool = False
    seed: int

    @model_validator(mode="after")
    def validate_method(self) -> LoraConfigContract:
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target_modules must be unique")
        if self.method == "qlora" and self.quantization != "bitsandbytes_4bit_nf4":
            raise ValueError("QLoRA requires bitsandbytes_4bit_nf4 quantization")
        if self.method == "lora" and self.quantization != "none":
            raise ValueError("plain LoRA must not enable 4-bit quantization")
        if self.quantization == "none" and self.use_double_quant:
            raise ValueError("double quantization requires a quantized runtime")
        return self


class TrainingArgumentsContract(FrozenModel):
    config_version: str = Field(min_length=1)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    max_seq_length: int = Field(gt=0, le=32768)
    per_device_train_batch_size: int = Field(gt=0)
    per_device_eval_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    optimizer: str = Field(min_length=1)
    scheduler: str = Field(min_length=1)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    num_train_epochs: float = Field(gt=0.0)
    max_steps: int | None = Field(default=None, gt=0)
    logging_steps: int = Field(gt=0)
    save_strategy: Literal["steps", "epoch"]
    eval_strategy: Literal["steps", "epoch"]
    save_steps: int | None = Field(default=None, gt=0)
    eval_steps: int | None = Field(default=None, gt=0)
    save_total_limit: int = Field(gt=0)
    gradient_checkpointing: bool
    max_grad_norm: float = Field(gt=0.0)
    checkpoint_selection_metric: Literal["eval_loss"] = "eval_loss"
    checkpoint_selection_mode: Literal["min"] = "min"
    early_stopping_patience: int | None = Field(default=None, gt=0)
    include_reasoning_target: bool = True
    seed: int

    @model_validator(mode="after")
    def validate_schedules(self) -> TrainingArgumentsContract:
        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("save_steps is required for step-based checkpointing")
        if self.eval_strategy == "steps" and self.eval_steps is None:
            raise ValueError("eval_steps is required for step-based validation")
        if self.early_stopping_patience is not None:
            raise ValueError(
                "Phase 09 protocol does not authorize early stopping; keep it disabled"
            )
        return self


class TrainingConfigBundle(FrozenModel):
    lora: LoraConfigContract
    training: TrainingArgumentsContract

    @model_validator(mode="after")
    def validate_shared_identity(self) -> TrainingConfigBundle:
        if self.lora.seed != self.training.seed:
            raise ValueError("LoRA and training seeds must match")
        if self.lora.base_model_id != self.training.tokenizer_id:
            raise ValueError("tokenizer ID must match the frozen base model ID")
        if self.lora.base_model_revision != self.training.tokenizer_revision:
            raise ValueError("tokenizer revision must match the frozen base model revision")
        return self

    def canonical_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _read_json_yaml(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must contain one JSON/YAML object: {path}")
    return payload


def load_training_config(
    root: Path,
    *,
    lora_path: Path = LORA_CONFIG_PATH,
    training_args_path: Path = TRAINING_ARGS_PATH,
) -> TrainingConfigBundle:
    bundle = TrainingConfigBundle(
        lora=LoraConfigContract.model_validate(_read_json_yaml(root / lora_path)),
        training=TrainingArgumentsContract.model_validate(
            _read_json_yaml(root / training_args_path)
        ),
    )
    model_payload = _read_json_yaml(root / "configs/model.yaml")
    expected_id = str(model_payload["base_model_id"])
    expected_revision = str(model_payload["base_model_revision"])
    if bundle.lora.base_model_id != expected_id:
        raise ValueError("training config changed the frozen base model ID")
    if bundle.lora.base_model_revision != expected_revision:
        raise ValueError("training config changed the frozen base model revision")
    return bundle
