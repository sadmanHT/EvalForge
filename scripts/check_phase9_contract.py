#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.config import load_training_config  # noqa: E402
from app.training.formatter import prepare_training_dataset  # noqa: E402
from app.training.phase9_gate import Phase9Stage, evaluate_phase9_repository_state  # noqa: E402
from app.training.smoke import SmokeAdapter, run_smoke_training  # noqa: E402

DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


def main() -> int:
    bundle = load_training_config(ROOT)
    if bundle.lora.method != "qlora":
        raise RuntimeError("Phase 09 contract requires QLoRA")
    if bundle.lora.quantization != "bitsandbytes_4bit_nf4":
        raise RuntimeError("Phase 09 contract requires NF4 4-bit quantization")
    if not bundle.training.gradient_checkpointing:
        raise RuntimeError("Phase 09 contract requires gradient checkpointing")

    with tempfile.TemporaryDirectory(prefix="evalforge-phase9-contract-") as temp_dir:
        work = Path(temp_dir)
        prepared = prepare_training_dataset(
            root=ROOT,
            output_dir=work / "prepared",
            dataset_version=DATASET_VERSION,
            include_reasoning=bundle.training.include_reasoning_target,
        )
        if set(prepared.train_family_ids) & set(prepared.validation_family_ids):
            raise RuntimeError("Phase 09 contract detected train/validation family leakage")

        partial = run_smoke_training(work / "partial", total_steps=3)
        resumed = run_smoke_training(
            work / "resumed",
            total_steps=6,
            resume_from_checkpoint=Path(partial.checkpoint_path),
        )
        reloaded = SmokeAdapter.load(Path(resumed.adapter_path))
        if reloaded.step != 6 or resumed.steps_completed != 6:
            raise RuntimeError("Phase 09 CPU smoke failed save/reload/resume contract")

    status = evaluate_phase9_repository_state(ROOT)
    if status.stage is Phase9Stage.TRAINING_EVIDENCE_READY and status.adapter_sha256 is None:
        raise RuntimeError(f"Phase 09 repository contains invalid training evidence: {status.blocker}")

    print("PHASE09_CONTRACT=PASS")
    print(f"TRAINING_CONFIG_HASH={bundle.config_hash()}")
    print(f"TRAIN_RECORD_COUNT={prepared.train_record_count}")
    print(f"VALIDATION_RECORD_COUNT={prepared.validation_record_count}")
    print(f"LINEAGE_SHA256={prepared.lineage_sha256}")
    print(f"CPU_SMOKE_ADAPTER_SHA256={resumed.adapter_sha256}")
    print(f"PHASE09_STAGE={status.stage.value}")
    if status.blocker is not None:
        print(f"PHASE09_BLOCKER={status.blocker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
