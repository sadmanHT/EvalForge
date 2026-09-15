# Phase 09 fine-tuning guide

Phase 09 builds the reproducible LoRA/QLoRA training system. It does **not** run the
locked held-out test, publish the final Hugging Face model card, or make the
fine-tuned-vs-baseline claim; those belong to later phases.

## Scientific boundary

Training input is limited to incident families assigned to `train`. Validation may
select checkpoints and training settings. The locked `test` split is never formatted
into Phase 09 training files and may not choose hyperparameters or checkpoints.
Synthetic descendants are accepted only when both child and parent remain in the
same training family.

The inference prompt/output family remains the canonical
`zero-shot-baseline-v1` / `root-cause-prediction-v1` contract. Fine-tuning changes
adapter state, not the frozen base-model identity. The frozen base is
`mistralai/Mistral-7B-Instruct-v0.3` at
`e8737b84b4470b28db3a0be719b362b1bd39a14d`.

## Canonical configuration

`training/configs/lora_config.yaml` and `training/configs/training_args.yaml` are
versioned scientific inputs. `app.training.config.load_training_config()` validates
them against `configs/model.yaml` and computes a deterministic SHA-256 config hash.

The default is QLoRA using 4-bit NF4 with double quantization and float16 compute.
That choice follows the already-proven single-T4 load path and is a memory strategy,
not a model-family substitution. Never silently fall back to another base model.
If the connected GPU cannot train the frozen revision, preserve the failure and
change the training protocol explicitly.

Early stopping is disabled because the current protocol does not authorize it.
Validation loss may identify the best saved checkpoint, but the locked test may not
participate in checkpoint selection.

## Prepare deterministic data

```bash
python training/scripts/prepare_dataset.py \
  --dataset-version evalforge-incident-diagnosis-v0.1.0 \
  --output-dir /tmp/evalforge-phase9-prepared
```

The command writes `train.jsonl`, `validation.jsonl`, and `manifest.json`. Repeating
it from the same repository state must produce byte-identical split files and the
same checksums. There is deliberately no `test.jsonl`.

## CPU contract smoke

CI uses a tiny deterministic analytic adapter to prove forward/loss/backward/update,
checkpoint save, resume, adapter save/reload, and schema-compatible prediction
contracts without pretending that a 7B model was trained:

```bash
python training/scripts/smoke_train.py --output-dir /tmp/phase9-smoke --steps 6
```

The smoke adapter is **not** scientific model evidence. Phase 09 cannot exit on the
CPU smoke alone.

## Connected GPU training

Use a fresh Linux CUDA environment. Keep the Phase 01-compatible package family
recorded in `training/requirements.gpu.txt`, authenticate W&B through environment
secrets, prepare the dataset, then run:

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

Do not use `--allow-untracked` for completion evidence. A real completion candidate
must preserve a W&B run/artifact reference, environment/package/GPU metadata,
dataset and config checksums, selected validation checkpoint, and adapter checksum.

Resume uses the same command contract via `training/scripts/resume.py` with
`--resume-from-checkpoint`. To prove reload, run
`training/scripts/evaluate_checkpoint.py` against the saved adapter; it deliberately
uses one **validation** incident only.

## Adapter export boundary

`training/scripts/export_adapter.py` creates a checksum-bearing local adapter
manifest tied to the exact base model, dataset manifest, and training config.
It includes an explicit optional Hub push capability, but public release/model-card
publication is deferred to Phase 10.

## Failure handling

On OOM, divergence, non-finite loss/gradients, unusable outputs, or reload failure,
preserve the failing evidence. Inspect formatting, tokenizer/padding, target modules,
quantization/compute dtype, learning rate, and checkpoint integrity. Rerun the
smallest smoke first, then a short train/validation trial, then the intended run.
Do not call Phase 09 complete merely because training loss decreased.

## Phase 09 completion evidence

The hard exit requires all of the following: smoke train/save/reload/resume/inference;
no held-out family in training lineage; a connected W&B run with reproducibility
metadata; a versioned adapter linked to the exact base/dataset/config; cumulative
Phase 1–9 verification; and a clean-environment smoke with no blocker/critical
regressions.
