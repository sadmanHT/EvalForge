from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from app.inference.base_model import (
    BackendOutput,
    GenerationConfig,
    RuntimeConfig,
    TransformersBackend,
)


class PeftTransformersBackend(TransformersBackend):
    """Load a PEFT adapter on the frozen base model through the canonical inference backend."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        *,
        adapter_id: str,
        adapter_revision: str,
    ) -> None:
        super().__init__(runtime_config)
        if not adapter_id.strip() or not adapter_revision.strip():
            raise ValueError("adapter ID and revision are required")
        self.adapter_id = adapter_id.strip()
        self.adapter_revision = adapter_revision.strip()
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
            self._model = peft.PeftModel.from_pretrained(
                self._model,
                self.adapter_id,
                revision=self.adapter_revision,
                is_trainable=False,
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
