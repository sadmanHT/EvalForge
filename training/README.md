# Training boundary

Phase 09 activates this directory as the training orchestration surface while
reusable implementation lives in `backend/app/training/`.

Canonical Phase 09 assets:

- `configs/lora_config.yaml` — frozen LoRA/QLoRA adapter configuration;
- `configs/training_args.yaml` — tokenizer/trainer/checkpoint configuration;
- `requirements.gpu.txt` — connected GPU package pins;
- `scripts/prepare_dataset.py` — deterministic train/validation preparation;
- `scripts/train.py` / `scripts/resume.py` — real PEFT training and recovery;
- `scripts/evaluate_checkpoint.py` — validation-only adapter reload smoke;
- `scripts/export_adapter.py` — checksum-bearing export and optional future Hub push;
- `scripts/smoke_train.py` — CPU-only contract trainer for CI.

Top-level scripts must call canonical library code rather than duplicating dataset,
inference, evaluation, or persistence contracts. Locked-test evaluation and public
adapter release are Phase 10 responsibilities.
