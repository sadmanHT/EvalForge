# EvalForge

A domain-specific LLM evaluation, fine-tuning, and deployment platform for production incident diagnosis.

## Phase status

**Phase 01 — Research Contract, Benchmark Rules & Architecture Decisions: in progress**

Phase 01 establishes the contracts that every later experiment must obey: one frozen base-model revision, one locked benchmark protocol, deterministic root-cause labels, reproducibility metadata, research-validity rules, and hard quality gates.

## Primary study

EvalForge compares four strategies under one common evaluator:

1. zero-shot base model
2. RAG-only
3. LoRA/QLoRA fine-tuned only
4. fine-tuned + RAG

The primary outcome is exact root-cause-code accuracy on a leakage-safe locked held-out set. Supporting analysis covers hierarchical/top-3 accuracy, calibration, retrieval quality, latency, cost, uncertainty, and failure modes.

## Phase 01 verification

```bash
python -m unittest discover -s tests -v
python scripts/validate_phase1.py
```

The full 7B weight-load/generation smoke is intentionally a separate connected-hardware gate:

```bash
python scripts/model_smoke.py
```

Do not mark Phase 01 complete until the model smoke succeeds on a machine capable of loading the frozen model revision.

See `docs/research-protocol.md`, `docs/architecture.md`, `docs/quality-gates.md`, and `docs/adrs/`.
