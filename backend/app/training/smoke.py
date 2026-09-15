from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmokeAdapter:
    """Tiny deterministic CPU contract adapter; not a substitute for PEFT training."""

    weight: float
    bias: float
    step: int

    def forward(self, feature: float) -> float:
        return self.weight * feature + self.bias

    def predict_code(
        self,
        feature: float,
        *,
        negative_label: str = "disk_exhaustion",
        positive_label: str = "memory_leak",
    ) -> str:
        return positive_label if self.forward(feature) >= 0.5 else negative_label

    def save(self, path: Path) -> str:
        payload = {
            "format": "evalforge-phase9-smoke-adapter-v1",
            "weight": self.weight,
            "bias": self.bias,
            "step": self.step,
        }
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
        return hashlib.sha256((body + "\n").encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> SmokeAdapter:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "evalforge-phase9-smoke-adapter-v1":
            raise ValueError("unknown smoke adapter format")
        adapter = cls(
            weight=float(payload["weight"]),
            bias=float(payload["bias"]),
            step=int(payload["step"]),
        )
        if not all(math.isfinite(value) for value in (adapter.weight, adapter.bias)):
            raise ValueError("smoke adapter contains non-finite parameters")
        return adapter


@dataclass(frozen=True)
class SmokeRunResult:
    adapter_path: str
    adapter_sha256: str
    checkpoint_path: str
    steps_completed: int
    losses: tuple[float, ...]


def _checkpoint_payload(adapter: SmokeAdapter, losses: tuple[float, ...]) -> dict[str, object]:
    return {
        "format": "evalforge-phase9-smoke-checkpoint-v1",
        "adapter": {
            "weight": adapter.weight,
            "bias": adapter.bias,
            "step": adapter.step,
        },
        "losses": list(losses),
    }


def _save_checkpoint(path: Path, adapter: SmokeAdapter, losses: tuple[float, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _checkpoint_payload(adapter, losses),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(path: Path) -> tuple[SmokeAdapter, tuple[float, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "evalforge-phase9-smoke-checkpoint-v1":
        raise ValueError("unknown smoke checkpoint format")
    raw_adapter = payload["adapter"]
    adapter = SmokeAdapter(
        weight=float(raw_adapter["weight"]),
        bias=float(raw_adapter["bias"]),
        step=int(raw_adapter["step"]),
    )
    losses = tuple(float(value) for value in payload["losses"])
    if not all(math.isfinite(value) for value in (*losses, adapter.weight, adapter.bias)):
        raise ValueError("smoke checkpoint contains non-finite values")
    return adapter, losses


def run_smoke_training(
    output_dir: Path,
    *,
    total_steps: int = 6,
    learning_rate: float = 0.05,
    resume_from_checkpoint: Path | None = None,
) -> SmokeRunResult:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    if resume_from_checkpoint is None:
        adapter = SmokeAdapter(weight=0.0, bias=0.0, step=0)
        losses: tuple[float, ...] = ()
    else:
        adapter, losses = _load_checkpoint(resume_from_checkpoint)
        if adapter.step >= total_steps:
            raise ValueError("resume checkpoint already reached requested total_steps")

    fixtures = ((-1.0, 0.0), (1.0, 1.0), (-0.5, 0.0), (0.5, 1.0))
    mutable_losses = list(losses)
    for step in range(adapter.step, total_steps):
        feature, target = fixtures[step % len(fixtures)]
        prediction = adapter.forward(feature)
        error = prediction - target
        loss = error * error
        grad_weight = 2.0 * error * feature
        grad_bias = 2.0 * error
        if not all(math.isfinite(value) for value in (prediction, loss, grad_weight, grad_bias)):
            raise RuntimeError("non-finite smoke-training gradient or loss")
        adapter = SmokeAdapter(
            weight=adapter.weight - learning_rate * grad_weight,
            bias=adapter.bias - learning_rate * grad_bias,
            step=step + 1,
        )
        mutable_losses.append(loss)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"checkpoint-{adapter.step}.json"
    final_losses = tuple(mutable_losses)
    _save_checkpoint(checkpoint, adapter, final_losses)
    adapter_path = output_dir / "adapter.json"
    adapter_sha = adapter.save(adapter_path)
    reloaded = SmokeAdapter.load(adapter_path)
    if reloaded != adapter:
        raise RuntimeError("smoke adapter reload changed model parameters")
    return SmokeRunResult(
        adapter_path=str(adapter_path),
        adapter_sha256=adapter_sha,
        checkpoint_path=str(checkpoint),
        steps_completed=adapter.step,
        losses=final_losses,
    )
