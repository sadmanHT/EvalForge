from __future__ import annotations

import json
from pathlib import Path

from app.inference.finetuned_protocol import load_phase10_protocol

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    protocol = load_phase10_protocol(ROOT)
    print("PHASE10_CONTRACT=PASS")
    print(f"PHASE10_PROTOCOL_VERSION={protocol.protocol_version}")
    print(f"PHASE10_STATE={protocol.state.value}")
    print(f"PHASE10_SCIENTIFIC_CONFIG_HASH={protocol.scientific_config_hash()}")
    print(f"BASE_MODEL_ID={protocol.base_model_id}")
    print(f"BASE_MODEL_REVISION={protocol.base_model_revision}")
    print(f"CANDIDATE_ADAPTER_SHA256={protocol.candidate_adapter_sha256}")
    print(
        "DATA_EFFICIENCY_MATRIX="
        + json.dumps(
            {
                "fractions": list(protocol.data_efficiency.fractions),
                "seeds": list(protocol.data_efficiency.seeds),
                "subset_policy": protocol.data_efficiency.subset_policy,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    print(f"LOCKED_TEST_AUTHORIZED={str(protocol.locked_test_authorized).lower()}")


if __name__ == "__main__":
    main()
