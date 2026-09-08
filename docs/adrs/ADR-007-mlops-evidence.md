# ADR-007 — W&B and Hugging Face Evidence Responsibilities

**Status:** Accepted

## Decision
Use Weights & Biases for training/evaluation run tracking, hyperparameters, curves, and ML artifacts. Use Hugging Face Hub for the public released adapter/model card. PostgreSQL remains the canonical in-product experiment store.

## Consequences
External systems supplement reproducibility evidence; they do not replace database experiment identity/version records.
