from __future__ import annotations

import importlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evaluation.calibration import normalize_label_sequence_log_likelihoods
from app.evaluation.contracts import (
    ConfidenceSource,
    ParseStatus,
    Prediction,
    RankedLabel,
)
from app.inference.prompts import (
    BASELINE_PROMPT_VERSION,
    get_prompt_template,
    parse_baseline_output,
)


class QuantizationMode(StrEnum):
    NONE = "none"
    BITSANDBYTES_4BIT_NF4 = "bitsandbytes_4bit_nf4"


class InferenceError(RuntimeError):
    pass


class InferenceOOMError(InferenceError):
    pass


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    seed: int
    device: str = "auto"
    dtype: str = "auto"
    quantization: QuantizationMode = QuantizationMode.NONE


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=128, gt=0, le=1024)
    do_sample: bool = False
    temperature: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_deterministic_primary_generation(self) -> GenerationConfig:
        if self.do_sample:
            raise ValueError("Phase 06 primary baseline generation must be deterministic")
        if self.temperature != 0.0:
            raise ValueError("deterministic Phase 06 generation requires temperature=0")
        return self


class IncidentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


@dataclass(frozen=True)
class BackendOutput:
    raw_text: str
    label_log_likelihoods: Mapping[str, float]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None = None
    runtime_metadata: Mapping[str, object] | None = None


class BaseModelBackend(Protocol):
    @property
    def runtime_config(self) -> RuntimeConfig: ...

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: Sequence[str],
        generation_config: GenerationConfig,
    ) -> BackendOutput: ...


def _oom_like(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


class TransformersBackend:
    """Lazy Transformers runtime for the frozen real-model baseline.

    Torch/Transformers are intentionally imported only when load() is called so the
    service/CI dependency lock stays CPU-light. The connected GPU environment must
    install the versions recorded with the experiment evidence.
    """

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._runtime_config = runtime_config
        self._torch: Any | None = None
        self._transformers: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise InferenceError(
                "real baseline inference requires torch and transformers in the GPU runtime"
            ) from exc

        self._seed(torch)
        load_kwargs: dict[str, Any] = {
            "revision": self.runtime_config.revision,
            "device_map": self.runtime_config.device,
            "low_cpu_mem_usage": True,
        }
        if self.runtime_config.quantization is QuantizationMode.BITSANDBYTES_4BIT_NF4:
            try:
                load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                raise InferenceError(
                    "bitsandbytes_4bit_nf4 requested but quantization support is unavailable"
                ) from exc
        elif self.runtime_config.dtype != "auto":
            dtype = getattr(torch, self.runtime_config.dtype, None)
            if dtype is None:
                raise InferenceError(f"unsupported torch dtype: {self.runtime_config.dtype}")
            load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = "auto"

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.runtime_config.model_id,
                revision=self.runtime_config.revision,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.runtime_config.model_id,
                **load_kwargs,
            )
        except RuntimeError as exc:
            if _oom_like(exc):
                self._clear_cuda_cache(torch)
                raise InferenceOOMError(str(exc)) from exc
            raise InferenceError(f"model loading failed: {exc}") from exc
        except Exception as exc:
            raise InferenceError(f"model loading failed: {exc}") from exc

        self._torch = torch
        self._transformers = transformers
        self._tokenizer = tokenizer
        self._model = model

    def _seed(self, torch: Any) -> None:
        random.seed(self.runtime_config.seed)
        torch.manual_seed(self.runtime_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.runtime_config.seed)

    @staticmethod
    def _clear_cuda_cache(torch: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_loaded(self) -> tuple[Any, Any, Any]:
        self.load()
        if self._torch is None or self._tokenizer is None or self._model is None:
            raise AssertionError("model backend failed to initialize")
        return self._torch, self._tokenizer, self._model

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: Sequence[str],
        generation_config: GenerationConfig,
    ) -> BackendOutput:
        labels = tuple(allowed_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("allowed_labels must be a non-empty unique sequence")
        torch, tokenizer, model = self._require_loaded()
        self._seed(torch)
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        started = perf_counter()
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_tokens = int(inputs["input_ids"].shape[1])
            new_tokens = generated[0][prompt_tokens:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            scores = self._score_labels(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                prompt_input_ids=inputs["input_ids"],
                allowed_labels=labels,
            )
        except RuntimeError as exc:
            if _oom_like(exc):
                self._clear_cuda_cache(torch)
                raise InferenceOOMError(str(exc)) from exc
            raise InferenceError(f"model inference failed: {exc}") from exc
        except Exception as exc:
            raise InferenceError(f"model inference failed: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000.0
        return BackendOutput(
            raw_text=text,
            label_log_likelihoods=scores,
            input_tokens=prompt_tokens,
            output_tokens=int(new_tokens.shape[0]),
            latency_ms=latency_ms,
            runtime_metadata={
                "model_id": self.runtime_config.model_id,
                "model_revision": self.runtime_config.revision,
                "device": str(model.device),
                "dtype": self.runtime_config.dtype,
                "quantization": self.runtime_config.quantization.value,
                "seed": self.runtime_config.seed,
            },
        )

    @staticmethod
    def _score_labels(
        *,
        torch: Any,
        tokenizer: Any,
        model: Any,
        prompt_input_ids: Any,
        allowed_labels: Sequence[str],
    ) -> dict[str, float]:
        prefix_ids = tokenizer(
            '{"root_cause_code":"',
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(model.device)
        base_ids = prompt_input_ids.to(model.device)
        base_length = int(base_ids.shape[1] + prefix_ids.shape[1])
        prefix_context = torch.cat([base_ids, prefix_ids], dim=1)
        scores: dict[str, float] = {}
        for label in allowed_labels:
            label_ids = tokenizer(label, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].to(model.device)
            if int(label_ids.shape[1]) == 0:
                raise InferenceError(f"label tokenized to zero tokens: {label}")
            combined = torch.cat([prefix_context, label_ids], dim=1)
            with torch.inference_mode():
                logits = model(input_ids=combined).logits
            prediction_logits = logits[:, base_length - 1 : combined.shape[1] - 1, :]
            log_probabilities = torch.log_softmax(prediction_logits.float(), dim=-1)
            gathered = log_probabilities.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)
            score = float(gathered.sum().item())
            if not math.isfinite(score):
                raise InferenceError(f"non-finite label log-likelihood for {label}")
            scores[label] = score
        return scores


class ZeroShotBaselineAdapter:
    def __init__(
        self,
        *,
        backend: BaseModelBackend,
        allowed_labels: Sequence[str],
        prompt_version: str = BASELINE_PROMPT_VERSION,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        labels = tuple(allowed_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("allowed_labels must be a non-empty unique sequence")
        self.backend = backend
        self.allowed_labels = labels
        self.prompt_template = get_prompt_template(prompt_version)
        self.generation_config = generation_config or GenerationConfig()

    def predict(self, incident: IncidentInput) -> Prediction:
        prompt = self.prompt_template.render(
            title=incident.title,
            description=incident.description,
            allowed_labels=self.allowed_labels,
        )
        output = self.backend.generate(
            prompt=prompt,
            allowed_labels=self.allowed_labels,
            generation_config=self.generation_config,
        )
        if set(output.label_log_likelihoods) != set(self.allowed_labels):
            raise InferenceError("backend label scores do not exactly cover the frozen label set")
        probabilities = normalize_label_sequence_log_likelihoods(
            output.label_log_likelihoods,
            allowed_labels=self.allowed_labels,
        )
        metadata: dict[str, object] = {
            "pipeline_type": "ZERO_SHOT",
            "prompt_version": self.prompt_template.version,
            "output_schema_version": self.prompt_template.output_schema_version,
            "generation_config": self.generation_config.model_dump(mode="json"),
            "label_log_likelihoods": dict(output.label_log_likelihoods),
            "runtime": dict(output.runtime_metadata or {}),
        }
        parsed = parse_baseline_output(
            incident_id=incident.incident_id,
            raw_output=output.raw_text,
            allowed_labels=self.allowed_labels,
            latency_ms=output.latency_ms,
            pipeline_metadata=metadata,
        )
        payload = parsed.model_dump(mode="python", exclude_none=False)
        payload.update(
            {
                "input_tokens": output.input_tokens,
                "output_tokens": output.output_tokens,
                "total_tokens": output.input_tokens + output.output_tokens,
                "cost_usd": output.cost_usd,
                "label_probabilities": probabilities,
            }
        )
        if parsed.parse_status is ParseStatus.OK and parsed.predicted_root_cause_code is not None:
            predicted = parsed.predicted_root_cause_code
            payload["confidence_probability"] = probabilities[predicted]
            payload["confidence_source"] = ConfidenceSource.NORMALIZED_LABEL_SEQUENCE_LOG_LIKELIHOOD
            ranked_labels = sorted(
                self.allowed_labels,
                key=lambda label: (float(output.label_log_likelihoods[label]), label),
                reverse=True,
            )
            if ranked_labels[0] == predicted:
                payload["ranked_labels"] = tuple(
                    RankedLabel(
                        label=label,
                        score=float(output.label_log_likelihoods[label]),
                        probability=probabilities[label],
                    )
                    for label in ranked_labels
                )
            else:
                metadata = dict(payload["pipeline_metadata"])
                metadata["ranked_labels_omitted"] = "generated_label_disagrees_with_score_argmax"
                payload["pipeline_metadata"] = metadata
        return Prediction.model_validate(payload)

    def predict_batch(self, incidents: Sequence[IncidentInput]) -> tuple[Prediction, ...]:
        seen: set[str] = set()
        predictions: list[Prediction] = []
        for incident in incidents:
            if incident.incident_id in seen:
                raise ValueError(f"duplicate incident_id in batch: {incident.incident_id}")
            seen.add(incident.incident_id)
            predictions.append(self.predict(incident))
        return tuple(predictions)
