from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from app.db import build_engine, session_scope
from app.inference.base_model import BaseModelBackend, TransformersBackend, ZeroShotBaselineAdapter
from app.inference.protocol import BaselineProtocol, load_baseline_protocol, load_taxonomy
from app.inference.runner import BaselineRunSummary, run_baseline_experiment
from app.inference.tracking import ExperimentTracker, build_tracker_from_env
from app.models import Run
from app.services.dataset_import import import_phase3_dataset

BackendFactory = Callable[[BaselineProtocol], BaseModelBackend]


def _default_backend_factory(protocol: BaselineProtocol) -> BaseModelBackend:
    return TransformersBackend(protocol.runtime_config)


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Phase 06 baseline job requires non-empty {key}")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when provided")
    return value.strip()


class Phase6BaselineJobHandler:
    """Worker/CLI orchestration around the reusable Phase 06 baseline runner."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        engine: Engine | None = None,
        backend_factory: BackendFactory | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.root = (root or Path(os.environ.get("EVALFORGE_ROOT", "."))).resolve()
        self.engine = engine or build_engine()
        self.backend_factory = backend_factory or _default_backend_factory
        self.tracker = tracker or build_tracker_from_env()

    def __call__(self, payload: dict[str, Any]) -> dict[str, object]:
        protocol = load_baseline_protocol(self.root)
        labels, _categories = load_taxonomy(self.root)
        backend = self.backend_factory(protocol)
        adapter = ZeroShotBaselineAdapter(
            backend=backend,
            allowed_labels=labels,
            prompt_version=protocol.prompt_version,
            generation_config=protocol.generation_config,
        )
        split = _required_text(payload, "split")
        git_commit = _required_text(payload, "git_commit")
        hardware_runtime_descriptor = _required_text(payload, "hardware_runtime_descriptor")
        cost_rate_snapshot_version = _required_text(payload, "cost_rate_snapshot_version")
        max_attempts = int(payload.get("max_attempts", 2))
        upfront_cost_usd = float(payload.get("upfront_cost_usd", 0.0))

        with session_scope(self.engine) as session:
            import_phase3_dataset(session, self.root, protocol.dataset_version)
            summary = run_baseline_experiment(
                session,
                root=self.root,
                protocol=protocol,
                adapter=adapter,
                split=split,
                git_commit=git_commit,
                hardware_runtime_descriptor=hardware_runtime_descriptor,
                cost_rate_snapshot_version=cost_rate_snapshot_version,
                experiment_id=_optional_text(payload, "experiment_id"),
                run_id=_optional_text(payload, "run_id"),
                max_attempts=max_attempts,
                upfront_cost_usd=upfront_cost_usd,
            )

        summary_payload = _summary_payload(summary)
        tracking = self.tracker.log_baseline(
            run_id=summary.run_id,
            protocol_payload=protocol.scientific_payload(),
            summary=summary_payload,
        )
        with session_scope(self.engine) as session:
            run = session.get(Run, summary.run_id)
            if run is None:
                raise RuntimeError("completed Phase 06 run disappeared before tracking update")
            run.runtime_metadata = {
                **run.runtime_metadata,
                "experiment_tracking": tracking.as_dict(),
            }

        return {**summary_payload, "tracking": tracking.as_dict()}


def _summary_payload(summary: BaselineRunSummary) -> dict[str, object]:
    payload = asdict(summary)
    return {str(key): value for key, value in payload.items()}
