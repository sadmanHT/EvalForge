#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.retrieval.phase7_gate import (  # noqa: E402
    Phase7Stage,
    evaluate_phase7_repository_state,
)


def _fail(message: str) -> None:
    raise SystemExit(f"PHASE07_EXIT_GATE=FAIL {message}")


def main() -> int:
    try:
        status = evaluate_phase7_repository_state(ROOT)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))

    if status.stage is not Phase7Stage.COMPLETE:
        _fail(f"stage={status.stage.value} blocker={status.blocker}")
    if status.reviewer is None:
        _fail("completed repository state is missing manual reviewer")

    print("PHASE07_EXIT_GATE=PASS")
    print(f"KB_VERSION={status.kb_version}")
    print(f"KB_MANIFEST_CHECKSUM={status.manifest_checksum}")
    print(f"DOCUMENT_COUNT={status.document_count}")
    print(f"CHUNK_COUNT={status.chunk_count}")
    print(f"LEAKAGE_QUERIES={status.leakage_query_count}")
    print(f"MANUAL_REVIEWER={status.reviewer}")
    print(f"CI_EVIDENCE_RUN_ID={status.ci_run_id}")
    print(f"CI_EVIDENCE_HEAD_SHA={status.ci_head_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
