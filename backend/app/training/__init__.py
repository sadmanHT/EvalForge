"""Canonical Phase 09 fine-tuning library.

The real PEFT/Transformers stack is imported lazily so ordinary service and CI
paths stay lightweight.  The deterministic smoke trainer exercises the
training/checkpoint/reload contracts without pretending to be Mistral training.
"""

from app.training.config import (
    LoraConfigContract,
    TrainingArgumentsContract,
    TrainingConfigBundle,
    load_training_config,
)
from app.training.formatter import (
    FORMATTER_VERSION,
    PreparedDatasetManifest,
    TrainingExample,
    prepare_training_dataset,
)
from app.training.smoke import SmokeAdapter, SmokeRunResult, run_smoke_training

__all__ = [
    "FORMATTER_VERSION",
    "LoraConfigContract",
    "PreparedDatasetManifest",
    "SmokeAdapter",
    "SmokeRunResult",
    "TrainingArgumentsContract",
    "TrainingConfigBundle",
    "TrainingExample",
    "load_training_config",
    "prepare_training_dataset",
    "run_smoke_training",
]
