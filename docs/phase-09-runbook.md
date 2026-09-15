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

The connected completion procedure must start from a clean tracked worktree. Record the exact checked-out commit with:

```bash
git rev-parse HEAD
```

## Connected GPU environment

Use a clean Linux CUDA environment and install the pinned Phase 09 GPU stack:

```bash
python -m pip install -r training/requirements.gpu.txt
```

Authenticate external services through environment variables rather than committed credentials. Completion evidence requires W&B, so set `WANDB_PROJECT` and authenticate W&B before training. The training scripts record Python, platform, PyTorch/CUDA, GPU name/VRAM, Transformers, Accelerate, bitsandbytes, PEFT, and W&B versions.

## Recommended: one-command connected completion

The canonical connected path is `training/scripts/complete_phase9.py`. It prepares leakage-safe data, runs the real QLoRA job, reloads the saved PEFT adapter through the canonical `FINETUNED` inference abstraction on a validation example, performs a second tracked CUDA run resumed from a checkpoint recorded by the source run, exports the adapter candidate, writes linked evidence, and invokes the Phase 09 hard-exit checker. It never formats or evaluates the locked test split.

```bash
export WANDB_PROJECT=evalforge
python training/scripts/complete_phase9.py \
  --work-dir /tmp/evalforge-phase9-connected \
  --run-id phase9-qlora-validation-v1 \
  --git-commit "$(git rev-parse HEAD)" \
  --hardware-runtime-descriptor '<GPU/CUDA/runtime description>'
```

The work directory must be empty or absent. The script refuses to overwrite existing canonical Phase 09 evidence and requires at least two retained checkpoints so the resume proof demonstrates actual recovery rather than a no-op restart.

On success it preserves these files under `evidence/phase-09/`:

- `training-run.json` — primary connected CUDA/W&B training record;
- `adapter-reload.json` — validation-only schema-compatible `FINETUNED` inference proof;
- `resume-training-run.json` — second connected CUDA/W&B run started from a recorded source checkpoint;
- `resume.json` — cryptographic link between the source run and the resumed run;
- `adapter-export.json` — checksum-preserving export proof without a Phase 10 Hub release.

Supporting evidence stores the SHA-256 of the exact `training-run.json`. The hard-exit gate rejects stale supporting files after the source training record changes. `resume.json` also stores the SHA-256 of `resume-training-run.json`, and the gate revalidates that resumed run as CUDA + W&B evidence.

## Manual component commands

The one-command path above is preferred. These component commands remain available for diagnosis or controlled reruns.

Prepare deterministic train/validation data:

```bash
python training/scripts/prepare_dataset.py \
  --dataset-version evalforge-incident-diagnosis-v0.1.0 \
  --output-dir /tmp/evalforge-phase9-prepared
```

Run connected training:

```bash
python training/scripts/train.py \
  --prepared-dir /tmp/evalforge-phase9-prepared \
  --output-dir /tmp/evalforge-phase9-train \
  --run-id phase9-qlora-validation-v1 \
  --git-commit "$(git rev-parse HEAD)" \
  --hardware-runtime-descriptor '<GPU/CUDA/runtime description>' \
  --evidence-path /tmp/phase9-training-run.json
```

Do not use `--allow-untracked` for scientific evidence. The connected run must produce finite training signals, a selected validation checkpoint, retained checkpoints, a saved PEFT adapter, an adapter checksum, CUDA/package metadata, and W&B run/artifact references.

Reload the saved adapter and run validation-only inference while generating linked evidence:

```bash
python training/scripts/evaluate_checkpoint.py \
  --adapter /tmp/evalforge-phase9-train/adapter \
  --adapter-revision local \
  --dataset-version evalforge-incident-diagnosis-v0.1.0 \
  --training-evidence /tmp/phase9-training-run.json \
  --evidence-path /tmp/phase9-adapter-reload.json
```

Resume from a checkpoint listed by the source training evidence. The command runs `train.py` again and emits both the resumed training record and the linked resume proof:

```bash
python training/scripts/resume.py \
  --prepared-dir /tmp/evalforge-phase9-prepared \
  --output-dir /tmp/evalforge-phase9-resume \
  --run-id phase9-qlora-resume-v1 \
  --git-commit "$(git rev-parse HEAD)" \
  --hardware-runtime-descriptor '<GPU/CUDA/runtime description>' \
  --resume-from-checkpoint /tmp/evalforge-phase9-train/checkpoint-<N> \
  --source-training-evidence /tmp/phase9-training-run.json \
  --resumed-training-evidence-path /tmp/phase9-resume-training-run.json \
  --evidence-path /tmp/phase9-resume.json
```

Export the adapter candidate and generate linked export evidence:

```bash
python training/scripts/export_adapter.py \
  --adapter-dir /tmp/evalforge-phase9-train/adapter \
  --destination /tmp/evalforge-phase9-adapter-export \
  --training-config-hash <hash-from-training-run> \
  --dataset-manifest-checksum <checksum-from-training-run> \
  --base-model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --base-model-revision e8737b84b4470b28db3a0be719b362b1bd39a14d \
  --training-evidence /tmp/phase9-training-run.json \
  --evidence-path /tmp/phase9-adapter-export.json
```

The export path verifies that the copied adapter checksum is identical to the trained adapter before adding the export manifest. Do not pass `--push-hub-repo` for the Phase 09 completion evidence; public Hugging Face release is Phase 10.

## Completion gate

After real evidence is preserved under `evidence/phase-09/`, run:

```bash
make phase9-exit
make verify-all
make fresh-smoke
```

`make phase9-exit` must report `PHASE09_STAGE=complete` and `PHASE09_EXIT=PASS`. Cumulative CI and the clean-environment Compose job must then pass on the exact evidence-bearing commit. Phase 09 is not complete merely because training loss decreased, the CPU smoke passed, or a manually authored JSON file resembles the evidence schema.
