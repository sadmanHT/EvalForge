# Phase 01 Handoff Status

## Current status

**IN PROGRESS — repository-static Phase 01 contracts/tests are implemented; the connected full 7B weight-load/generation smoke remains required before Phase 01 may be marked complete.**

## Canonical artifacts

- `docs/research-protocol.md`
- `docs/architecture.md`
- `docs/quality-gates.md`
- `docs/adrs/ADR-001..009`
- `datasets/schema.md`
- `configs/model.yaml`
- `configs/study.yaml`
- `configs/label-taxonomy.yaml`
- `contracts/phase1.py`
- `tests/test_phase1_contracts.py`
- `scripts/validate_phase1.py`
- `scripts/model_smoke.py`

## Verification commands

```bash
python -m unittest discover -s tests -v
python scripts/validate_phase1.py
python scripts/model_smoke.py
```

The first two commands are dependency-free and form the repository/static gate. The third is the hard connected-model gate and requires `torch`, `transformers`, and sufficient RAM/VRAM.

## Frozen external provenance

EvalForge records a read-only upstream OpsSentinel benchmark label snapshot from commit:

`40467b27085130c110098cac5077fb62dee4e5aa`

No EvalForge Phase 01 change writes to OpsSentinel. Future upstream changes must be imported/versioned explicitly rather than mutating this taxonomy in place.

## Known limitation

The current agent execution environment has not yet proven a full 7B weight load/generation. Until `scripts/model_smoke.py` succeeds against the exact frozen revision, the Phase 01 hard exit gate is not satisfied.
