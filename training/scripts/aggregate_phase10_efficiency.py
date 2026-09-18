#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.efficiency_study import aggregate_efficiency_conditions  # noqa: E402
from app.inference.finetuned_protocol import load_phase10_protocol  # noqa: E402
from app.training.evidence import sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"condition evidence must contain an object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the complete predeclared Phase 10 data-efficiency matrix."
    )
    parser.add_argument("--conditions-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    protocol = load_phase10_protocol(ROOT)
    paths = sorted(args.conditions_root.glob("*/condition-evidence.json"))
    payloads = [_load(path) for path in paths]
    aggregate = aggregate_efficiency_conditions(payloads, protocol=protocol)
    aggregate["conditions"] = [
        {
            "condition_id": payload["condition_id"],
            "fraction": payload["fraction"],
            "seed": payload["seed"],
            "evidence_sha256": sha256_file(path),
            "subset_manifest_sha256": payload["subset_manifest_sha256"],
            "adapter_sha256": payload["adapter_sha256"],
            "validation_result_hash": payload["validation_evaluation"]["result_hash"],
            "tracking": payload["tracking"],
        }
        for path, payload in zip(paths, payloads, strict=True)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PHASE10_DATA_EFFICIENCY_AGGREGATE=PASS")
    print(f"CONDITION_COUNT={aggregate['condition_count']}")
    print(f"OUTPUT={args.output}")
    print(f"OUTPUT_SHA256={sha256_file(args.output)}")
    print("TEST_SPLIT_USED=false")
    print("PRIMARY_ADAPTER_SELECTION_USE=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
