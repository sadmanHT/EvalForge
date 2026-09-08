# ADR-007 — W&B and Hugging Face Evidence

**Status:** Accepted

Weights & Biases records training/evaluation run metadata, curves, and artifacts; PostgreSQL remains the system of record. Hugging Face Hub publishes the final adapter/model card at a frozen revision. External services must never be the sole copy of experiment identity or metrics, and CI must use local/fake contract tests where network access is unavailable.
