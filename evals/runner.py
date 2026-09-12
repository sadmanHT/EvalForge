#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.db import build_engine, session_scope
from app.inference.evidence import export_phase6_run_evidence, validate_run_evidence
from app.inference.gpu_host import (
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)
from app.inference.orchestration import Phase6BaselineJobHandler
from app.inference.protocol import load_baseline_protocol

ROOT = Path(__file__).resolve().parents[1]
FREEZE_RECORD_PATH = Path("evidence/phase-06/protocol-freeze.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _assert_locked_test_host_matches_freeze(environment_fingerprint: str) -> None:
    freeze_path = ROOT / FREEZE_RECORD_PATH
    if not freeze_path.is_file():
        raise ValueError("locked test requires committed Phase 06 protocol-freeze evidence")
    freeze = _load_json(freeze_path)
    expected = freeze.get("gpu_environment_fingerprint_sha256")
    if expected != environment_fingerprint:
        raise ValueError("locked test GPU/software fingerprint differs from validation freeze")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 06 zero-shot baseline through the canonical worker handler."
    )
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--gpu-host-evidence", required=True, type=Path)
    parser.add_argument("--cost-rate-snapshot-version", required=True)
    parser.add_argument("--gpu-hour-usd", required=True, type=float)
    parser.add_argument("--experiment-id")
    parser.add_argument("--run-id")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--upfront-cost-usd", type=float, default=0.0)
    parser.add_argument("--evidence-output", type=Path)
    args = parser.parse_args()

    protocol = load_baseline_protocol(ROOT)
    host_contract = load_gpu_host_contract(ROOT)
    verify_reference_smoke(ROOT, host_contract)
    host_evidence_path = _resolve(args.gpu_host_evidence)
    host_evidence = validate_gpu_host_evidence(
        _load_json(host_evidence_path),
        contract=host_contract,
    )
    if args.split == "test":
        _assert_locked_test_host_matches_freeze(
            host_evidence.environment_fingerprint_sha256
        )

    payload = {
        "split": args.split,
        "git_commit": args.git_commit,
        "hardware_runtime_descriptor": hardware_runtime_descriptor(host_evidence),
        "cost_rate_snapshot_version": args.cost_rate_snapshot_version,
        "gpu_hour_usd": args.gpu_hour_usd,
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "max_attempts": args.max_attempts,
        "upfront_cost_usd": args.upfront_cost_usd,
    }
    result = Phase6BaselineJobHandler(root=ROOT)(payload)

    if args.evidence_output is not None:
        engine = build_engine()
        with session_scope(engine) as session:
            evidence = export_phase6_run_evidence(
                session,
                root=ROOT,
                run_id=str(result["run_id"]),
                protocol=protocol,
            )
        validate_run_evidence(
            evidence,
            root=ROOT,
            protocol=protocol,
            expected_split=args.split,
        )
        output = _resolve(args.evidence_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {
            **result,
            "gpu_environment_fingerprint_sha256": (
                host_evidence.environment_fingerprint_sha256
            ),
            "gpu_host_evidence_sha256": host_evidence.evidence_sha256,
            "evidence_output": str(output),
            "evidence_sha256": evidence["evidence_sha256"],
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
