from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.inference.protocol import (
    BaselineProtocol,
    ProtocolState,
    build_experiment_config,
    load_baseline_protocol,
)

ROOT = Path(__file__).resolve().parents[3]


def test_committed_protocol_is_validation_only_and_deterministic() -> None:
    protocol = load_baseline_protocol(ROOT)
    assert protocol.state is ProtocolState.VALIDATION
    assert protocol.locked_test_authorized is False
    assert protocol.assert_split_allowed("validation") == "validation"
    with pytest.raises(ValueError, match="locked test"):
        protocol.assert_split_allowed("test")
    first = protocol.scientific_config_hash()
    second = BaselineProtocol.model_validate(
        json.loads((ROOT / "configs/phase6-baseline.json").read_text(encoding="utf-8"))
    ).scientific_config_hash()
    assert first == second
    assert len(first) == 64


def test_locked_test_requires_frozen_authorized_protocol() -> None:
    protocol = load_baseline_protocol(ROOT)
    payload = protocol.model_dump(mode="json")
    payload.update({"state": "frozen", "locked_test_authorized": True})
    frozen = BaselineProtocol.model_validate(payload)
    assert frozen.assert_split_allowed("test") == "test"

    invalid = protocol.model_dump(mode="json")
    invalid["locked_test_authorized"] = True
    with pytest.raises(ValueError, match="requires a frozen protocol"):
        BaselineProtocol.model_validate(invalid)


def test_experiment_config_captures_protocol_and_runtime_identity() -> None:
    protocol = load_baseline_protocol(ROOT)
    config = build_experiment_config(
        root=ROOT,
        protocol=protocol,
        git_commit="phase6-test-commit",
        hardware_runtime_descriptor="phase6-test-runtime",
        cost_rate_snapshot_version="phase6-test-rates-v1",
    )
    assert config.pipeline_type.value == "ZERO_SHOT"
    assert config.base_model_id == protocol.base_model_id
    assert config.base_model_revision == protocol.base_model_revision
    assert config.prompt_version == protocol.prompt_version
    assert config.output_schema_version == protocol.output_schema_version
    assert config.generation_config == protocol.generation_config.model_dump(mode="json")
    assert config.git_commit == "phase6-test-commit"
    assert len(config.dependency_lock_checksum) == 64
