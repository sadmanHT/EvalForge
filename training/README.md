# Training boundary

Phase 02 establishes this directory as the training orchestration boundary. LoRA/QLoRA training, checkpointing, W&B tracking, and adapter release are intentionally deferred to the later training phases. Future scripts must call reusable implementation from canonical libraries instead of duplicating backend/evaluation contracts.
