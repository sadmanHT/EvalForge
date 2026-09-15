from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.training.config import TrainingConfigBundle


class TrainingRuntimeError(RuntimeError):
    pass


class TrainingOOMError(TrainingRuntimeError):
    pass


@dataclass(frozen=True)
class TrainingRunResult:
    output_dir: str
    adapter_dir: str
    adapter_sha256: str
    selected_checkpoint: str | None
    train_metrics: dict[str, float]
    log_history: tuple[dict[str, object], ...]
    checkpoint_paths: tuple[str, ...]


def _oom_like(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise TrainingRuntimeError(f"adapter directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = item.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _read_prepared_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TrainingRuntimeError(f"invalid prepared row at {path}:{line_number}")
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            message = f"prepared row must contain two messages: {path}:{line_number}"
            raise TrainingRuntimeError(message)
        rows.append(payload)
    if not rows:
        raise TrainingRuntimeError(f"prepared split is empty: {path}")
    return tuple(rows)


class PeftTrainingRuntime:
    """Real Transformers/PEFT runtime imported only in a connected training environment."""

    def __init__(self, bundle: TrainingConfigBundle) -> None:
        self.bundle = bundle

    @staticmethod
    def _load_stack() -> tuple[Any, Any, Any]:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise TrainingRuntimeError(
                "real Phase 09 training requires torch, transformers, accelerate, peft, "
                "and bitsandbytes in the GPU runtime"
            ) from exc
        return torch, transformers, peft

    def train(
        self,
        *,
        train_path: Path,
        validation_path: Path,
        output_dir: Path,
        resume_from_checkpoint: Path | None = None,
        report_to_wandb: bool = False,
    ) -> TrainingRunResult:
        torch, transformers, peft = self._load_stack()
        train_rows = _read_prepared_rows(train_path)
        validation_rows = _read_prepared_rows(validation_path)
        lora = self.bundle.lora
        args = self.bundle.training

        if hasattr(transformers, "set_seed"):
            transformers.set_seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        dtype = getattr(torch, lora.compute_dtype, None)
        if dtype is None:
            raise TrainingRuntimeError(f"unsupported compute dtype: {lora.compute_dtype}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            lora.base_model_id,
            revision=lora.base_model_revision,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, object] = {
            "revision": lora.base_model_revision,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if lora.method == "qlora":
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=lora.use_double_quant,
            )
        else:
            model_kwargs["torch_dtype"] = dtype

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                lora.base_model_id,
                **model_kwargs,
            )
            if lora.method == "qlora":
                model = peft.prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=args.gradient_checkpointing,
                )
            peft_config = peft.LoraConfig(
                r=lora.rank,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                bias=lora.bias,
                task_type=lora.task_type,
                target_modules=list(lora.target_modules),
            )
            model = peft.get_peft_model(model, peft_config)
        except RuntimeError as exc:
            if _oom_like(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise TrainingOOMError(str(exc)) from exc
            raise TrainingRuntimeError(f"PEFT model initialization failed: {exc}") from exc
        except Exception as exc:
            raise TrainingRuntimeError(f"PEFT model initialization failed: {exc}") from exc

        class InstructionDataset:
            def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
                self.rows = rows

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, list[int]]:
                row = self.rows[index]
                messages = row["messages"]
                user_message = messages[0]
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prompt_text = tokenizer.apply_chat_template(
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_seq_length,
                )
                prompt_ids = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_seq_length,
                )["input_ids"]
                input_ids = list(encoded["input_ids"])
                attention_mask = list(encoded["attention_mask"])
                labels = list(input_ids)
                prompt_length = min(len(prompt_ids), len(labels))
                labels[:prompt_length] = [-100] * prompt_length
                if not any(label != -100 for label in labels):
                    raise TrainingRuntimeError(
                        "assistant target was fully truncated for incident "
                        f"{row.get('incident_id')}"
                    )
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        class DataCollator:
            def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
                max_length = max(len(item["input_ids"]) for item in features)
                padded_ids: list[list[int]] = []
                padded_masks: list[list[int]] = []
                padded_labels: list[list[int]] = []
                for item in features:
                    padding = max_length - len(item["input_ids"])
                    padded_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
                    padded_masks.append(item["attention_mask"] + [0] * padding)
                    padded_labels.append(item["labels"] + [-100] * padding)
                return {
                    "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
                    "labels": torch.tensor(padded_labels, dtype=torch.long),
                }

        class FiniteLossCallback(transformers.TrainerCallback):
            def on_log(
                self,
                _args: Any,
                _state: Any,
                _control: Any,
                logs: dict[str, object] | None = None,
                **_kwargs: object,
            ) -> None:
                for key in ("loss", "eval_loss", "grad_norm"):
                    if logs is None or key not in logs:
                        continue
                    value = logs[key]
                    if isinstance(value, int | float) and not math.isfinite(float(value)):
                        raise TrainingRuntimeError(f"non-finite training signal: {key}={value}")

        training_kwargs: dict[str, object] = {
            "output_dir": str(output_dir),
            "seed": args.seed,
            "data_seed": args.seed,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optim": args.optimizer,
            "lr_scheduler_type": args.scheduler,
            "warmup_ratio": args.warmup_ratio,
            "num_train_epochs": args.num_train_epochs,
            "logging_steps": args.logging_steps,
            "save_strategy": args.save_strategy,
            "eval_strategy": args.eval_strategy,
            "save_total_limit": args.save_total_limit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_grad_norm": args.max_grad_norm,
            "load_best_model_at_end": True,
            "metric_for_best_model": args.checkpoint_selection_metric,
            "greater_is_better": args.checkpoint_selection_mode == "max",
            "report_to": ["wandb"] if report_to_wandb else [],
            "remove_unused_columns": False,
            "fp16": lora.compute_dtype == "float16",
            "bf16": lora.compute_dtype == "bfloat16",
        }
        if args.max_steps is not None:
            training_kwargs["max_steps"] = args.max_steps
        if args.save_steps is not None:
            training_kwargs["save_steps"] = args.save_steps
        if args.eval_steps is not None:
            training_kwargs["eval_steps"] = args.eval_steps

        try:
            hf_args = transformers.TrainingArguments(**training_kwargs)
            trainer = transformers.Trainer(
                model=model,
                args=hf_args,
                train_dataset=InstructionDataset(train_rows),
                eval_dataset=InstructionDataset(validation_rows),
                data_collator=DataCollator(),
                callbacks=[FiniteLossCallback()],
            )
            train_output = trainer.train(
                resume_from_checkpoint=(
                    str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
                )
            )
            adapter_dir = output_dir / "adapter"
            trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
            tokenizer.save_pretrained(adapter_dir)
        except RuntimeError as exc:
            if _oom_like(exc):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise TrainingOOMError(str(exc)) from exc
            raise TrainingRuntimeError(f"training failed: {exc}") from exc
        except TrainingRuntimeError:
            raise
        except Exception as exc:
            raise TrainingRuntimeError(f"training failed: {exc}") from exc

        raw_metrics = getattr(train_output, "metrics", {})
        train_metrics = {
            str(key): float(value)
            for key, value in raw_metrics.items()
            if isinstance(value, int | float) and math.isfinite(float(value))
        }
        log_history = tuple(
            {str(key): value for key, value in entry.items()}
            for entry in getattr(trainer.state, "log_history", [])
            if isinstance(entry, dict)
        )
        selected = getattr(trainer.state, "best_model_checkpoint", None)
        checkpoints = tuple(
            str(path) for path in sorted(output_dir.glob("checkpoint-*")) if path.is_dir()
        )
        return TrainingRunResult(
            output_dir=str(output_dir),
            adapter_dir=str(adapter_dir),
            adapter_sha256=_tree_sha256(adapter_dir),
            selected_checkpoint=str(selected) if selected else None,
            train_metrics=train_metrics,
            log_history=log_history,
            checkpoint_paths=checkpoints,
        )
