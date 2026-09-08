# EvalForge Quality Gates and Definition of Done

## Project-wide Definition of Done

A phase is complete only when:

1. required implementation/artifacts exist;
2. focused tests for the new behavior pass;
3. the entire current-phase suite passes;
4. all applicable earlier-phase regressions/integration checks pass;
5. the clean-environment smoke path passes;
6. no blocker or critical defect remains;
7. evidence is preserved so the next agent can reproduce the claim.

A feature passing once in isolation is not sufficient.

## Defect severity model

| Severity | Definition | Phase-exit rule |
| --- | --- | --- |
| **Blocker** | Invalidates research, causes data loss/corruption, security boundary failure, or prevents the required end-to-end workflow from being meaningfully exercised. | Must be fixed before exit. |
| **Critical** | Breaks an integrated workflow or produces materially wrong experiment/system behavior. | Must be fixed before exit. |
| **Major** | Important feature is wrong, degraded, unreliable, or missing but does not invalidate the entire phase contract. | May remain only if explicitly allowed by the phase brief; otherwise fix. |
| **Minor** | Cosmetic/non-blocking issue with no effect on research validity or required workflow correctness. | May be carried with documentation. |

## Mandatory repair-and-retest loop

1. Run the smallest relevant test after each coherent vertical slice.
2. Preserve failure evidence.
3. Diagnose the true failing layer: data, config, schema, migration, adapter, worker, API, frontend, environment, or external integration.
4. Fix the root cause. Do not disable tests, weaken valid assertions, swallow errors, hard-code outputs, or mock away behavior the phase must prove.
5. Rerun the narrow failing test.
6. Rerun the entire current-phase suite.
7. Rerun cumulative applicable regressions/integration tests.
8. Repeat the clean-environment smoke path.
9. Mark complete only when all required gates are green.

## Cumulative test categories

- static quality: formatting/linting/type checks when the relevant code/tooling exists;
- unit tests: pure logic and validators;
- property/invariant tests: split disjointness, metric bounds, deterministic hashes, label consistency, pipeline invariants;
- integration tests: API/database/worker/vector/migration/external-client boundaries as introduced;
- end-to-end tests: browser/API/queue/worker/database/SSE/UI path as introduced;
- research regression: frozen tiny smoke benchmark once the evaluation harness exists;
- fresh-environment verification: rebuild from clean state once Phase 02 establishes containers/tooling.

## Coverage floor once executable modules expand

- aim for at least 80% backend line coverage overall;
- aim for at least 90% in critical evaluation/dataset/reproducibility modules;
- coverage is evidence density, not a substitute for meaningful tests.

## Canonical cumulative commands

Phase 02 establishes the canonical Makefile surface:

```text
make format
make lint
make typecheck
make test
make test-integration
make test-e2e
make test-regression
make db-migrate
make smoke
make eval-smoke
make verify-all
```

Until Phase 02 creates that tooling, Phase 01 uses the explicit commands documented in `docs/phase-01-handoff.md`. The absence of a Phase-02 Makefile is not treated as Phase-01 completion evidence.

## Phase 01-specific exit rule

Phase 01 is not complete until the exact frozen 7B–8B model revision is actually loaded in the intended GPU environment and produces a valid structured smoke prediction using a canonical label, in addition to all static/config/schema tests passing.
