#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.gpu_host import (  # noqa: E402
    capture_gpu_host_evidence,
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture or verify the pinned Phase 06 real-GPU host evidence."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()

    contract = load_gpu_host_contract(ROOT)
    verify_reference_smoke(ROOT, contract)
    if args.verify is not None:
        path = args.verify if args.verify.is_absolute() else ROOT / args.verify
        evidence = validate_gpu_host_evidence(_load_json(path), contract=contract)
        print("PHASE06_GPU_HOST_PREFLIGHT=PASS")
        print(f"ENVIRONMENT_FINGERPRINT={evidence.environment_fingerprint_sha256}")
        print(f"HARDWARE_RUNTIME_DESCRIPTOR={hardware_runtime_descriptor(evidence)}")
        return 0

    evidence = capture_gpu_host_evidence(root=ROOT, contract=contract)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE06_GPU_HOST_PREFLIGHT=PASS")
    print(f"OUTPUT={output.relative_to(ROOT)}")
    print(f"ENVIRONMENT_FINGERPRINT={evidence.environment_fingerprint_sha256}")
    print(f"HARDWARE_RUNTIME_DESCRIPTOR={hardware_runtime_descriptor(evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
