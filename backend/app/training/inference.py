from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.inference.base_model import (
    BackendOutput,
    BaseModelBackend,
    GenerationConfig,
    IncidentInput,
    RuntimeConfig,
    TransformersBackend,
    build_prediction_from_backend_output,
)
from app.inference.prompts import BASELINE_PROMPT_VERSION, get_prompt_template
from app.training.smoke import SmokeAdapter


class PeftTransformersBackend(TransformersBackend):
    """Load a PEFT adapter on the frozen base model through the canonical inference backend."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        *,
        adapter_id: str,
        adapter_revision: str,
        use_adapter_revision_for_loading: bool = True,
    ) -> None:
        super().__init__(runtime_config)
        if not adapter_id.strip() or not adapter_revision.strip():
            raise ValueError("adapter ID and revision are required")
        self.adapter_id = adapter_id.strip()
        self.adapter_revision = adapter_revision.strip()
        self.use_adapter_revision_for_loading = bool(use_adapter_revision_for_loading)
        self._adapter_loaded = False

    def load(self) -> None:
        if self._adapter_loaded:
            return
        super().load()
        try:
            peft: Any = importlib.import_module("peft")
        except ImportError as exc:
            raise RuntimeError("fine-tuned inference requires the peft package") from exc
        if self._model is None:
            raise AssertionError("base model failed to initialize before adapter load")
        try:
            load_kwargs: dict[str, object] = {"is_trainable": False}
            if self.use_adapter_revision_for_loading:
                load_kwargs["revision"] = self.adapter_revision
            self._model = peft.PeftModel.from_pretrained(
                self._model,
                self.adapter_id,
                **load_kwargs,
            )
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(f"failed to load PEFT adapter: {exc}") from exc
        self._adapter_loaded = True

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: Sequence[str],
        generation_config: GenerationConfig,
    ) -> BackendOutput:
        output = super().generate(
            prompt=prompt,
            allowed_labels=allowed_labels,
            generation_config=generation_config,
        )
        metadata = dict(output.runtime_metadata or {})
        metadata.update(
            {
                "adapter_id": self.adapter_id,
                "adapter_revision": self.adapter_revision,
            }
        )
        return BackendOutput(
            raw_text=output.raw_text,
            label_log_likelihoods=output.label_log_likelihoods,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=output.latency_ms,
            cost_usd=output.cost_usd,
            runtime_metadata=metadata,
        )


class SmokeAdapterBackend:
    """CPU contract backend that reloads a saved smoke adapter via the canonical interface."""

    def __init__(self, adapter_path: Path) -> None:
        self.adapter_path = adapter_path.resolve()
        payload = self.adapter_path.read_bytes()
        self.adapter_sha256 = hashlib.sha256(payload).hexdigest()
        self.adapter = SmokeAdapter.load(self.adapter_path)
        self._runtime_config = RuntimeConfig(
            model_id="evalforge/phase9-smoke-adapter",
            revision=self.adapter_sha256,
            seed=0,
            device="cpu",
            dtype="float64",
        )

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: Sequence[str],
        generation_config: GenerationConfig,
    ) -> BackendOutput:
        del generation_config
        labels = tuple(allowed_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("allowed_labels must be a non-empty unique sequence")
        digest = hashlib.sha256(prompt.encode()).digest()
        feature = (int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)) * 2.0 - 1.0
        preferred = self.adapter.predict_code(feature)
        predicted = preferred if preferred in labels else labels[0]
        score = self.adapter.forward(feature)
        label_scores = {label: (-0.1 if label == predicted else -4.0) for label in labels}
        raw_text = json.dumps(
            {
                "root_cause_code": predicted,
                "reasoning": "phase9 CPU adapter fresh-process contract",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return BackendOutput(
            raw_text=raw_text,
            label_log_likelihoods=label_scores,
            input_tokens=len(prompt.split()),
            output_tokens=len(raw_text.split()),
            latency_ms=0.0,
            cost_usd=0.0,
            runtime_metadata={
                "adapter_id": str(self.adapter_path),
                "adapter_revision": self.adapter_sha256,
                "adapter_step": self.adapter.step,
                "adapter_forward_score": score,
                "contract_backend": "phase9-cpu-smoke-v1",
            },
        )


class FineTunedAdapterPipeline:
    """Canonical fine-tuned prediction adapter shared by smoke and real PEFT backends."""

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

    def predict(self, incident: IncidentInput) -> Any:
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
        return build_prediction_from_backend_output(
            incident_id=incident.incident_id,
            output=output,
            allowed_labels=self.allowed_labels,
            pipeline_metadata={
                "pipeline_type": "FINETUNED",
                "prompt_version": self.prompt_template.version,
                "output_schema_version": self.prompt_template.output_schema_version,
                "generation_config": self.generation_config.model_dump(mode="json"),
            },
        )
