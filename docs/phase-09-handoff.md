# Phase 09 handoff

## Current state

Repository-side Phase 09 implementation is ready for the real connected run, but the phase remains intentionally incomplete until CUDA/W&B evidence exists. In the absence of `evidence/phase-09/training-run.json`, `scripts/check_phase9_exit.py` must report `PHASE09_STAGE=pre_external_training` and return a failing hard-exit status. CI treats that exact pre-external state as an expected boundary rather than fabricating completion evidence.

## Implemented system

The Phase 09 training system includes deterministic train/validation instruction formatting, held-out family and synthetic-lineage checks, frozen QLoRA/training configuration, a lazy Transformers/PEFT runtime, prompt-token loss masking, finite-signal checks, checkpointing and resume support, W&B tracking, canonical fine-tuned adapter inference, adapter export, a CPU-only contract trainer, Redis worker orchestration, and Postgres registration of Job, AdapterVersion, Artifact, and candidate FINETUNED experiment records.

The production worker registers `phase9_training` with the real connected PEFT executor. That path fails closed without W&B configuration and relies on the runtime's CUDA requirement; the deterministic CPU executor is injected only by smoke/integration tests. Its artifact is explicitly `scientific_adapter=false`, so it proves orchestration and persistence without substituting for training the frozen 7B model.

A fresh-process CPU adapter reload test now goes through the same `FineTunedAdapterPipeline` prediction abstraction used by PEFT inference, so save/reload/schema behavior is not tested through a separate fake output contract.

## Frozen identity

The frozen base model is `mistralai/Mistral-7B-Instruct-v0.3` at revision `e8737b84b4470b28db3a0be719b362b1bd39a14d`. The canonical configuration is versioned in `training/configs/lora_config.yaml` and `training/configs/training_args.yaml`. `app.training.config.load_training_config()` cross-checks that identity against `configs/model.yaml` and produces the deterministic training-config hash used by the evidence gate.

Training data comes only from `train`; `validation` may select the checkpoint; the locked `test` split is never formatted into Phase 09 training files and may not choose hyperparameters or checkpoints.

## Connected completion path

`training/scripts/complete_phase9.py` is the preferred external execution entrypoint. From a clean tracked worktree it:

1. prepares deterministic train/validation files only;
2. runs the frozen QLoRA configuration on CUDA and writes `training-run.json` with W&B references;
3. reloads that exact adapter and performs validation-only inference through the `FINETUNED` pipeline;
4. selects a retained source checkpoint and performs a second tracked CUDA run via the resume path;
5. exports the adapter candidate while proving the copied adapter checksum is unchanged;
6. copies generated evidence into `evidence/phase-09/` only after those steps succeed;
7. invokes the Phase 09 hard-exit checker and removes newly copied evidence if the gate rejects it.

The script requires `WANDB_PROJECT`, verifies `--git-commit` equals the checked-out HEAD, refuses a dirty tracked worktree, refuses to overwrite existing canonical evidence, and requires at least two retained checkpoints for a meaningful resume proof.

## Evidence contract still required

The connected GPU run must preserve `evidence/phase-09/training-run.json` with `phase9-training-run-v1`, the exact training-config hash, base-model identity, dataset/prepared-data hashes, lineage hash, exact git commit, hardware descriptor, CUDA environment metadata, selected validation checkpoint, retained checkpoints, adapter checksum, and configured W&B run/artifact references.

The hard exit also requires generated, linked records:

- `adapter-reload.json` with `phase9-adapter-reload-v1` and status `pass`;
- `resume-training-run.json`, itself a valid CUDA/W&B `phase9-training-run-v1` record with a non-null `resume_from_checkpoint`;
- `resume.json` with `phase9-resume-v1` and status `pass`;
- `adapter-export.json` with `phase9-adapter-export-evidence-v1` and status `pass`.

The reload/resume/export PASS records carry the SHA-256 of the exact primary `training-run.json`. The resume PASS record also carries the SHA-256 of `resume-training-run.json`. The gate rejects stale records, mismatched source or resumed W&B identities, changed dataset/model identities, a non-FINETUNED reload path, test-split reload evidence, adapter-copy checksum changes, or a Phase 09 Hub-release request.

## Verification commands

Before GPU work, use `make phase9-contract` and `make test-phase9`. For the connected procedure follow `docs/phase-09-runbook.md`. After preserving real evidence, run:

```bash
make phase9-exit
make verify-all
make fresh-smoke
```

The evidence-bearing commit must then pass cumulative CI and the clean Compose job on that exact commit before Phase 09 may be marked complete.

## Scope boundary

Do not run the locked held-out test as part of Phase 09 checkpoint selection, do not publish a final model card, and do not make the final baseline-vs-fine-tuned claim here. Those belong to later evaluation/release work. Phase 09 ends when the training system and its real adapter evidence are reproducible and the hard exit passes.
