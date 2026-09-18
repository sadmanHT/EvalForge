from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from app.inference.finetuned_comparison import (
    validate_paired_baseline_finetuned_comparison,
)
from app.inference.finetuned_evidence import validate_portable_finetuned_evidence
from app.inference.finetuned_protocol import load_phase10_protocol
from app.inference.protocol import load_baseline_protocol
from app.training.evidence import sha256_file, tree_sha256

PHASE10_MODEL_CARD_VERSION = "phase10-huggingface-model-card-v1"
PHASE10_HUB_RELEASE_EVIDENCE_VERSION = "phase10-huggingface-release-v1"
PHASE10_HUB_SMOKE_EVIDENCE_VERSION = "phase10-huggingface-smoke-v1"
LOCKED_TEST_SOURCE_COMMIT = "c59910e00f5e4fd0a722d2796da416c977753ddd"


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _metric(payload: Mapping[str, object], name: str) -> float:
    metrics = payload.get("metric_values")
    if not isinstance(metrics, Mapping):
        raise ValueError("Phase 10 evidence metric_values are malformed")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Phase 10 evidence is missing numeric metric: {name}")
    return float(value)


def adapter_file_manifest(adapter_dir: Path) -> tuple[dict[str, object], ...]:
    if not adapter_dir.is_dir():
        raise ValueError("adapter directory does not exist")
    files: list[dict[str, object]] = []
    for path in sorted(item for item in adapter_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(adapter_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("adapter directory is empty")
    return tuple(files)


def build_phase10_model_card(*, root: Path, repo_id: str) -> str:
    repo_id = repo_id.strip()
    if "/" not in repo_id or repo_id.startswith("/") or repo_id.endswith("/"):
        raise ValueError("Hugging Face repo_id must use the owner/name form")

    protocol = load_phase10_protocol(root)
    locked_path = root / "evidence/phase-10/locked-test/phase10-finetuned-test.json"
    comparison_path = root / "evidence/phase-10/baseline-finetuned-comparison.json"
    baseline_path = root / "evidence/phase-06/test-run.json"
    locked = _load_object(locked_path)
    comparison = _load_object(comparison_path)
    baseline = _load_object(baseline_path)

    validate_portable_finetuned_evidence(
        locked,
        root=root,
        protocol=protocol,
        expected_split="test",
        expected_run_id="phase10-finetuned-test-v1",
        expected_source_commit=LOCKED_TEST_SOURCE_COMMIT,
    )
    validate_paired_baseline_finetuned_comparison(
        comparison,
        root=root,
        baseline_protocol=load_baseline_protocol(root),
        finetuned_protocol=protocol,
        baseline_evidence=baseline,
        finetuned_evidence=locked,
        finetuned_evidence_file_sha256=sha256_file(locked_path),
        expected_finetuned_source_commit=LOCKED_TEST_SOURCE_COMMIT,
    )

    model_contract = _load_object(root / "configs/model.yaml")
    license_name = str(model_contract.get("license", "apache-2.0"))
    validation = _load_object(root / "evidence/phase-10/validation-run.json")
    paired = comparison["paired_accuracy"]
    if not isinstance(paired, Mapping):
        raise ValueError("paired Phase 10 comparison is malformed")

    test_accuracy = _metric(locked, "primary.exact_accuracy")
    test_parse_failure = _metric(locked, "quality.parse_failure_rate")
    validation_accuracy = _metric(validation, "primary.exact_accuracy")
    validation_parse_failure = _metric(validation, "quality.parse_failure_rate")
    baseline_section = comparison["baseline"]
    finetuned_section = comparison["finetuned"]
    if not isinstance(baseline_section, Mapping) or not isinstance(
        finetuned_section, Mapping
    ):
        raise ValueError("paired Phase 10 comparison sections are malformed")

    return f"""---
license: {license_name}
base_model: {protocol.base_model_id}
library_name: peft
pipeline_tag: text-generation
tags:
- peft
- lora
- qlora
- incident-diagnosis
- evalforge
---

# EvalForge Mistral-7B Incident-Diagnosis QLoRA Adapter

Model-card contract: `{PHASE10_MODEL_CARD_VERSION}`

This repository publishes the **PEFT adapter only** for EvalForge's frozen FINETUNED arm.
It is intended for reproducible research on the EvalForge production-incident diagnosis
benchmark, not for autonomous production incident response.

## Frozen identity

- Hugging Face repository: `{repo_id}`
- Base model: `{protocol.base_model_id}`
- Base revision: `{protocol.base_model_revision}`
- Adapter SHA-256: `{protocol.candidate_adapter_sha256}`
- Source training run: `{protocol.source_training_run_id}`
- Source training evidence SHA-256: `{protocol.source_training_evidence_sha256}`
- Source training config SHA-256: `{protocol.source_training_config_hash}`
- Dataset: `{protocol.dataset_version}`
- Prompt: `{protocol.prompt_version}`
- Output schema: `{protocol.output_schema_version}`
- Evaluator: `{protocol.evaluator_version}`
- Phase 10 scientific config SHA-256: `{protocol.scientific_config_hash()}`

The adapter checkpoint was selected in Phase 09 using **validation eval_loss** before the
Phase 10 locked test. The locked-test outcome was not used to retrain, switch checkpoints,
change the prompt, change decoding, alter confidence scoring, or select a data-efficiency
condition.

## Benchmark results

The benchmark contains six validation incidents and six locked-test incidents.

| Split / arm | Exact accuracy | Parse failure rate |
| --- | ---: | ---: |
| Fine-tuned validation | {validation_accuracy:.6f} | {validation_parse_failure:.6f} |
| Fine-tuned locked test | {test_accuracy:.6f} | {test_parse_failure:.6f} |
| Zero-shot locked test | {float(baseline_section["exact_accuracy"]):.6f} | {float(baseline_section["parse_failure_rate"]):.6f} |

On the six paired locked-test incidents, fine-tuned minus zero-shot exact accuracy was
`{float(paired["finetuned_minus_baseline"]):.6f}`. The 95% paired bootstrap interval was
`[{float(paired["bootstrap_ci_lower"]):.6f}, {float(paired["bootstrap_ci_upper"]):.6f}]`
with {int(paired["bootstrap_resamples"])} resamples, and exact two-sided McNemar
`p={float(paired["mcnemar_p_value"]):.6f}`.

The single fine-tuned locked-test miss was an `INVALID_JSON` parse failure after the
generation reached the frozen 128-token output limit. That observation is preserved without
post-test retuning.

## Training and data boundaries

Training uses only train incident families. Validation controls checkpoint/hyperparameter
choices. Test families are locked and are never used as training examples. Synthetic
descendants, when present, are permitted only for train families and retain parent-family
lineage.

The Phase 10 data-efficiency matrix is secondary descriptive analysis only. It does not
replace or select this primary adapter.

## Loading

Load the exact base revision and then the adapter under the `adapter/` directory with
Transformers + PEFT. EvalForge's scientific runtime uses bitsandbytes 4-bit NF4,
double quantization, float16 compute, deterministic decoding, and the frozen prompt/output
contract above.

## Limitations

- The locked test has only six incidents; estimates have very high uncertainty.
- Several benchmark narratives are diagnosis-explicit, so these results do not establish
  robustness to ambiguous real-world incidents.
- The primary fine-tuned locked-test accuracy is 5/6, while the preserved zero-shot baseline
  is 6/6 on the same six IDs. This small benchmark does not support a broad superiority claim.
- One fine-tuned test response failed strict JSON parsing at the frozen output-token limit.
- The adapter is a research artifact, not an operational safety system or substitute for
  production observability and human incident analysis.

## Reproducibility

The EvalForge repository preserves raw validation/test predictions, the exact adapter hash,
paired statistics, frozen source/training identities, and the release evidence that records
the immutable Hugging Face commit used for clean-download verification.
"""


def validate_release_revision(revision: str) -> str:
    normalized = revision.strip().lower()
    if len(normalized) != 40 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("Hugging Face release revision must be an immutable 40-hex commit")
    return normalized


def build_release_evidence(
    *,
    root: Path,
    repo_id: str,
    revision: str,
    commit_url: str,
    adapter_dir: Path,
    source_git_commit: str,
    model_card_sha256: str,
) -> dict[str, object]:
    protocol = load_phase10_protocol(root)
    revision = validate_release_revision(revision)
    observed = tree_sha256(adapter_dir)
    if observed != protocol.candidate_adapter_sha256:
        raise ValueError("release adapter content does not match the frozen Phase 10 adapter")
    return {
        "evidence_version": PHASE10_HUB_RELEASE_EVIDENCE_VERSION,
        "status": "completed",
        "repo_id": repo_id,
        "revision": revision,
        "commit_url": commit_url,
        "public": True,
        "source_git_commit": source_git_commit,
        "phase10_scientific_config_hash": protocol.scientific_config_hash(),
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "adapter_sha256": observed,
        "adapter_files": list(adapter_file_manifest(adapter_dir)),
        "model_card_sha256": model_card_sha256,
        "locked_test_evidence_sha256": sha256_file(
            root / "evidence/phase-10/locked-test/phase10-finetuned-test.json"
        ),
        "primary_adapter_selection_use": False,
        "post_test_retuning": False,
    }
