#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.db import build_engine, session_scope
from app.inference.base_model import TransformersBackend
from app.inference.costing import HourlyRateCostBackend
from app.inference.gpu_host import (
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)
from app.inference.rag_evidence import (
    export_phase8_rag_run_evidence,
    validate_phase8_rag_run_evidence,
)
from app.inference.rag_orchestration import Phase8RAGJobHandler
from app.inference.rag_protocol import RAGProtocol, RAGProtocolState, load_rag_protocol

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run every registered Phase 08 RAG variant on validation while sharing one "
            "frozen base-model load across the controlled ablation suite."
        )
    )
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--gpu-host-evidence", required=True, type=Path)
    parser.add_argument("--cost-rate-snapshot-version", required=True)
    parser.add_argument("--gpu-hour-usd", required=True, type=float)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("evidence/phase-08/validation"),
    )
    parser.add_argument("--id-suffix", default="kaggle-v1")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--upfront-cost-usd", type=float, default=0.0)
    args = parser.parse_args()

    protocol, suite = load_rag_protocol(ROOT)
    if protocol.state is not RAGProtocolState.VALIDATION or protocol.locked_test_authorized:
        raise SystemExit("PHASE08_VALIDATION_ABLATIONS=FAIL protocol must be validation/unlocked")
    if not os.environ.get("WANDB_PROJECT", "").strip():
        raise SystemExit("PHASE08_VALIDATION_ABLATIONS=FAIL WANDB_PROJECT is required")

    host_contract = load_gpu_host_contract(ROOT)
    verify_reference_smoke(ROOT, host_contract)
    host = validate_gpu_host_evidence(
        _load_json(_resolve(args.gpu_host_evidence)),
        contract=host_contract,
    )
    descriptor = hardware_runtime_descriptor(host)
    output_dir = _resolve(args.evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / f"{variant.variant_id}.json"
        for variant in suite.variants
        if (output_dir / f"{variant.variant_id}.json").exists()
    ]
    if existing:
        raise SystemExit(
            "PHASE08_VALIDATION_ABLATIONS=FAIL refusing to overwrite existing evidence: "
            + ", ".join(str(path) for path in existing)
        )

    shared_backend = HourlyRateCostBackend(
        TransformersBackend(protocol.runtime_config),
        gpu_hour_usd=args.gpu_hour_usd,
    )

    def backend_factory(requested: RAGProtocol) -> HourlyRateCostBackend:
        if requested.runtime_config != protocol.runtime_config:
            raise ValueError("Phase 08 shared backend protocol drift detected")
        return shared_backend

    engine = build_engine()
    handler = Phase8RAGJobHandler(
        root=ROOT,
        engine=engine,
        backend_factory=backend_factory,
    )
    summaries: list[dict[str, object]] = []
    try:
        for variant in suite.variants:
            identifier = f"phase8-{variant.variant_id}-validation-{args.id_suffix}"
            result = handler(
                {
                    "split": "validation",
                    "variant_id": variant.variant_id,
                    "git_commit": args.git_commit,
                    "hardware_runtime_descriptor": descriptor,
                    "cost_rate_snapshot_version": args.cost_rate_snapshot_version,
                    "gpu_hour_usd": args.gpu_hour_usd,
                    "experiment_id": identifier,
                    "run_id": identifier,
                    "max_attempts": args.max_attempts,
                    "upfront_cost_usd": args.upfront_cost_usd,
                }
            )
            with session_scope(engine) as session:
                evidence = export_phase8_rag_run_evidence(
                    session,
                    root=ROOT,
                    run_id=str(result["run_id"]),
                    protocol=protocol,
                    variant=variant,
                )
            validate_phase8_rag_run_evidence(
                evidence,
                root=ROOT,
                protocol=protocol,
                expected_split="validation",
                expected_variant=variant,
            )
            output = output_dir / f"{variant.variant_id}.json"
            output.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summaries.append(
                {
                    "variant_id": variant.variant_id,
                    "run_id": result["run_id"],
                    "evidence_sha256": evidence["evidence_sha256"],
                    "primary_exact_accuracy": evidence["metrics"]["primary.exact_accuracy"],
                    "retrieval_context_recall": evidence["metrics"]["retrieval.context_recall"],
                    "evidence_path": str(output),
                }
            )
    finally:
        engine.dispose()

    print(
        json.dumps(
            {
                "status": "PASS",
                "variant_count": len(summaries),
                "gpu_environment_fingerprint_sha256": host.environment_fingerprint_sha256,
                "runs": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
