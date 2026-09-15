# Phase 09 handoff

## Current state

Repository-side Phase 09 implementation is ready for validation, but the phase remains intentionally incomplete until real connected GPU/W&B evidence exists. In the absence of `evidence/phase-09/training-run.json`, `scripts/check_phase9_exit.py` must report `PHASE09_STAGE=pre_external_training` and return a failing hard-exit status. CI treats that exact pre-external state as an expected boundary rather than fabricating completion evidence.

## Implemented system

The Phase 09 training system includes deterministic train/validation instruction formatting, held-out family and synthetic-lineage checks, frozen QLoRA/training configuration, a lazy Transformers/PEFT runtime, prompt-token loss masking, finite-signal checks, checkpointing and resume support, W&B tracking, adapter inference/reload support, adapter export, a CPU-only contract trainer, Redis worker orchestration, and Postgres registration of Job, AdapterVersion, Artifact, and candidate FINETUNED experiment records.

The production worker registers `phase9_training` with the real connected PEFT executor. That path fails closed without W&B configuration and relies on the runtime's CUDA requirement; the deterministic CPU executor is injected only by smoke/integration tests. Its artifact is explicitly `scientific_adapter=false`, so it proves orchestration and persistence without substituting for training the frozen 7B model.

## Frozen identity

The frozen base model is `mistralai/Mistral-7B-Instruct-v0.3` at revision `e8737b84b4470b28db3a0be719b362b1bd39a14d`. The canonical configuration is versioned in `training/configs/lora_config.yaml` and `training/configs/training_args.yaml`. `app.training.config.load_training_config()` cross-checks that identity against `configs/model.yaml` and produces the deterministic training-config hash used by the evidence gate.

Training data comes only from `train`; `validation` may select the checkpoint; the locked `test` split is never formatted into Phase 09 training files and may not choose hyperparameters or checkpoints.

## Evidence contract still required

The connected GPU run must preserve `evidence/phase-09/training-run.json` with `phase9-training-run-v1`, the exact training-config hash, base-model identity, dataset/prepared-data hashes, lineage hash, exact git commit, hardware descriptor, CUDA environment metadata, selected validation checkpoint, adapter checksum, and configured W&B run/artifact references.

After that run, the hard exit also requires:

- `evidence/phase-09/adapter-reload.json` with `phase9-adapter-reload-v1` and status `pass`;
- `evidence/phase-09/resume.json` with `phase9-resume-v1` and status `pass`;
- `evidence/phase-09/adapter-export.json` with `phase9-adapter-export-evidence-v1` and status `pass`.

Each must reference the same adapter checksum and training-config hash as the connected training run. The gate rejects CUDA/W&B omissions and mismatched identities.

## Verification commands

Before GPU work, use `make phase9-contract` and `make test-phase9`. For the connected procedure follow `docs/phase-09-runbook.md`. After preserving real evidence, run `make phase9-exit`, then the cumulative `make verify-all` and clean-environment smoke.

## Scope boundary

Do not run the locked held-out test as part of Phase 09 checkpoint selection, do not publish a final model card, and do not make the final baseline-vs-fine-tuned claim here. Those belong to later evaluation/release work. Phase 09 ends when the training system and its real adapter evidence are reproducible and the hard exit passes.
