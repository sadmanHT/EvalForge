# Phase 01 Handoff

## Current status

**NOT COMPLETE** until the exact frozen model/revision passes the real GPU load + structured-output smoke and the resulting evidence is preserved.

## Implemented and proven repository-static outcomes

- scoped research question and pre-specified hypotheses;
- primary/secondary metrics, calibration/confidence, paired statistics, cost accounting, allowed claims, invalidation rules;
- versioned canonical root-cause taxonomy and change policy;
- base-model selection rule, candidate record, exact frozen model ID/revision;
- architecture contract and ADR set;
- project-wide Definition of Done, defect severities, mandatory repair/retest loop;
- Phase 01 config/label/experiment validators and tests;
- strict real-model smoke script that validates canonical structured output.

## Canonical versions

- Study: `evalforge-incident-diagnosis-primary-v1`
- Protocol: `1.0.0`
- Taxonomy: `1.0.0`
- Incident schema: `incident-schema-v1`
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Base revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`

## Phase 01 verification commands

Focused config/schema unit suite:

```bash
python -m unittest tests.test_phase1_contracts -v
```

Full current Phase 01 test suite:

```bash
python -m unittest discover -s tests -p 'test_phase1_*.py' -v
```

Static documentation/config gate:

```bash
python scripts/validate_phase1.py
```

Hard real-model gate on intended GPU environment:

```bash
python scripts/model_smoke.py
# or, when the chosen GPU environment requires quantized loading:
python scripts/model_smoke.py --load-in-4bit
```

Preserve successful model smoke stdout exactly at:

```text
evidence/phase-01/model-smoke.json
```

Then run the machine-checkable hard exit gate:

```bash
python scripts/check_phase1_exit.py
```

It must print `PHASE01_EXIT_GATE=PASS` before Phase 01 can be marked complete.

## Clean-environment rule

Phase 02 will establish Docker/Makefile/CI infrastructure. For Phase 01, “clean environment” means a clean checkout with no developer-local generated state, running only declared Python/runtime dependencies and the commands above. Once Phase 02 exists, `make verify-all` supersedes this temporary Phase 01 command surface.

## Open hard gate

A successful `scripts/model_smoke.py` run is mandatory. The phase file explicitly forbids advancing if the selected model cannot load because of access/license/output/hardware incompatibility; in that case reopen ADR-005, choose another compatible 7B–8B instruction model, freeze a new revision, and rerun all Phase 01 tests.

## Evidence required before completion

- `evidence/phase-01/model-smoke.json`
- Phase 01 test report
- ADR set
- exact research-protocol repository commit hash once synchronized to GitHub
- exact verification commands and runtime descriptor

## Remaining limitations

No blocker/critical defect may remain in a completed handoff. At present the missing real-model smoke evidence is a **blocker for Phase 01 completion**, not an intentionally deferred feature.
