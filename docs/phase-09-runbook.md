# Phase 09 runbook

Phase 09 builds and verifies the LoRA/QLoRA training system for the frozen EvalForge incident-diagnosis study. It does not run the locked held-out test, publish a final model card, or make the final fine-tuned-vs-baseline claim.

## Frozen scientific identity

The base model is `mistralai/Mistral-7B-Instruct-v0.3` at revision `e8737b84b4470b28db3a0be719b362b1bd39a14d`. The canonical training inputs are `training/configs/lora_config.yaml` and `training/configs/training_args.yaml`. The current protocol is QLoRA with 4-bit NF4, double quantization, float16 compute, gradient checkpointing, validation-loss checkpoint selection, and no early stopping.

Do not silently substitute a smaller model, another revision, CPU full-model training, or another quantization method. If the frozen protocol cannot run on the connected GPU, preserve the failure and change the protocol explicitly in a new revision.

## Repository preflight

Run the repository-side contract before allocating a GPU:

```bash
make phase9-contract
make test-phase9
```

`phase9-contract` proves deterministic train/validation preparation, family-disjoint lineage, CPU save/reload/resume mechanics, and the frozen QLoRA configuration. The CPU smoke adapter is deliberately marked non-scientific and cannot satisfy the Phase 09 hard exit.

## Prepare deterministic training data

```bash
python training/scripts/prepare_dataset.py \
  --dataset-version evalforge-incident-diagnosis-v0.1.0 \
  --output-dir /tmp/evalforge-phase9-prepared
```

The output contains `train.jsonl`, `validation.jsonl`, and `manifest.json`. There is no `test.jsonl`. Re-running from the same repository state must reproduce the same split checksums and lineage hash.

## Connected GPU environment

Use a clean Linux CUDA environment and install the pinned Phase 09 GPU stack:

```bash
python -m pip install -r training/requirements.gpu.txt
```

Authenticate the external services through environment variables rather than committed credentials. Completion evidence requires W&B, so set `WANDB_PROJECT` and authenticate W&B before training. Record the exact branch commit and a concrete hardware/runtime descriptor.

## Run QLoRA training

```bash
export WANDB_PROJECT=evalforge
python training/scripts/train.py \
  --prepared-dir /tmp/evalforge-phase9-prepared \
  --output-dir /tmp/evalforge-phase9-train \
  --run-id phase9-qlora-validation-v1 \
  --git-commit <exact-branch-head> \
  --hardware-runtime-descriptor '<GPU/CUDA/runtime description>' \
  --evidence-path /tmp/phase9-training-run.json
```

Do not use `--allow-untracked` for scientific evidence. The connected run must produce finite training signals, a selected validation checkpoint, a saved PEFT adapter, an adapter checksum, GPU/package metadata, and W&B run/artifact references.

## Resume proof

Interrupt or stop after a checkpoint, then resume with the same scientific configuration:

```bash
python training/scripts/resume.py \
  --prepared-dir /tmp/evalforge-phase9-prepared \
  --output-dir /tmp/evalforge-phase9-resume \
  --run-id phase9-qlora-resume-v1 \
  --git-commit <exact-branch-head> \
  --hardware-runtime-descriptor '<GPU/CUDA/runtime description>' \
  --resume-from-checkpoint /tmp/evalforge-phase9-train/checkpoint-<N> \
  --evidence-path /tmp/phase9-resume-training-run.json
```

Preserve a `phase9-resume-v1` pass record in `evidence/phase-09/resume.json` only after the resumed run is actually verified against the same adapter/config identity.

## Reload and validation-only inference proof

Load the saved adapter over the exact frozen base model and run the validation-only schema smoke:

```bash
python training/scripts/evaluate_checkpoint.py \
  --adapter /tmp/evalforge-phase9-train/adapter \
  --adapter-revision local
```

Do not use the locked test split for this proof. Preserve `evidence/phase-09/adapter-reload.json` only after the reload/inference path succeeds and its adapter checksum matches the connected training evidence.

## Export adapter candidate

Create a checksum-bearing adapter export tied to the exact base model, dataset manifest, and training-config hash:

```bash
python training/scripts/export_adapter.py \
  --adapter-dir /tmp/evalforge-phase9-train/adapter \
  --destination /tmp/evalforge-phase9-adapter-export \
  --training-config-hash <hash-from-phase9-contract> \
  --dataset-manifest-checksum <dataset-manifest-sha256> \
  --base-model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --base-model-revision e8737b84b4470b28db3a0be719b362b1bd39a14d
```

The optional Hub upload capability is not Phase 09 release authorization. Public release and final model-card work remain deferred.

## Completion gate

Commit or otherwise preserve the real evidence under `evidence/phase-09/` using the schemas enforced by `app.training.phase9_gate`. Then run:

```bash
make phase9-exit
make verify-all
make fresh-smoke
```

Phase 09 is complete only when the hard exit passes with real CUDA/W&B training evidence, matching adapter reload/resume/export evidence, cumulative Phase 1–9 verification, and a clean-environment smoke. A decreasing training loss or a passing CPU smoke alone is insufficient.
