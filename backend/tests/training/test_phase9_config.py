from pathlib import Path

import pytest
from pydantic import ValidationError

from app.training.config import LoraConfigContract, load_training_config

ROOT = Path(__file__).resolve().parents[3]


def test_phase9_config_is_pinned_to_frozen_base_model() -> None:
    bundle = load_training_config(ROOT)
    assert bundle.lora.method == "qlora"
    assert bundle.lora.quantization == "bitsandbytes_4bit_nf4"
    assert bundle.lora.base_model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert bundle.lora.base_model_revision == "e8737b84b4470b28db3a0be719b362b1bd39a14d"
    assert bundle.training.early_stopping_patience is None
    assert len(bundle.config_hash()) == 64


def test_qlora_rejects_unquantized_config() -> None:
    with pytest.raises(ValidationError):
        LoraConfigContract(
            config_version="test",
            method="qlora",
            base_model_id="x",
            base_model_revision="r",
            target_modules=("q_proj",),
            rank=8,
            alpha=16,
            dropout=0.0,
            quantization="none",
            compute_dtype="float32",
            seed=1,
        )
