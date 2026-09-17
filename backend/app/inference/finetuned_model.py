from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.inference.base_model import GenerationConfig, RuntimeConfig
from app.training.evidence import load_training_evidence_identity, tree_sha256
from app.training.inference import FineTunedAdapterPipeline, PeftTransformersBackend


class AdapterRevisionMismatch(ValueError):
    pass


@dataclass(frozen=True)
class FineTunedAdapterIdentity:
    adapter_id: str
    adapter_revision: str
    adapter_sha256: str
    base_model_id: str
    base_model_revision: str
    source_training_run_id: str
    training_config_hash: str


def load_phase9_candidate_identity(root: Path) -> FineTunedAdapterIdentity:
    identity = load_training_evidence_identity(root / "evidence/phase-09/training-run.json")
    return FineTunedAdapterIdentity(
        adapter_id=identity.wandb_artifact_reference,
        adapter_revision=identity.adapter_sha256,
        adapter_sha256=identity.adapter_sha256,
        base_model_id=identity.base_model_id,
        base_model_revision=identity.base_model_revision,
        source_training_run_id=identity.run_id,
        training_config_hash=identity.training_config_hash,
    )


def verify_adapter_locator(
    locator: str,
    *,
    identity: FineTunedAdapterIdentity,
    adapter_revision: str,
) -> None:
    if adapter_revision != identity.adapter_revision:
        raise AdapterRevisionMismatch(
            "adapter revision mismatch: requested revision does not match the "
            "Phase 09 content-addressed adapter revision"
        )
    path = Path(locator)
    if path.exists():
        if not path.is_dir():
            raise ValueError("PEFT adapter locator must be a directory when local")
        observed = tree_sha256(path)
        if observed != identity.adapter_sha256:
            raise AdapterRevisionMismatch(
                "local adapter content checksum does not match the frozen adapter"
            )


def build_finetuned_pipeline(
    *,
    root: Path,
    adapter_locator: str,
    adapter_revision: str,
    runtime_config: RuntimeConfig,
    allowed_labels: Sequence[str],
    prompt_version: str,
    generation_config: GenerationConfig,
) -> FineTunedAdapterPipeline:
    identity = load_phase9_candidate_identity(root)
    if runtime_config.model_id != identity.base_model_id:
        raise ValueError("fine-tuned runtime changed the frozen base model ID")
    if runtime_config.revision != identity.base_model_revision:
        raise ValueError("fine-tuned runtime changed the frozen base model revision")
    verify_adapter_locator(
        adapter_locator,
        identity=identity,
        adapter_revision=adapter_revision,
    )
    is_local = Path(adapter_locator).exists()
    backend = PeftTransformersBackend(
        runtime_config,
        adapter_id=adapter_locator,
        adapter_revision=adapter_revision,
        use_adapter_revision_for_loading=not is_local,
    )
    return FineTunedAdapterPipeline(
        backend=backend,
        allowed_labels=allowed_labels,
        prompt_version=prompt_version,
        generation_config=generation_config,
    )
