# Phase 01 Evidence Directory

Do not fabricate evidence files.

Expected before Phase 01 completion:

- `test-report.txt` — actual Phase 01 test output
- `static-validation.txt` — actual `scripts/validate_phase1.py` output
- `model-smoke.json` — actual GPU smoke stdout from `scripts/model_smoke.py`
- research-protocol commit SHA recorded in the handoff after repository synchronization

`model-smoke.json` must not exist unless the exact frozen model/revision was actually loaded and the script exited successfully.
