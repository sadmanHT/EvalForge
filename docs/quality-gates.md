# EvalForge Quality Gates v1

## Definition of done

- **Implemented**: code/config/docs exist.
- **Working**: intended behavior is proven by appropriate tests.
- **Phase complete**: current-phase tests, all applicable prior-phase regressions, cumulative integration checks, and a clean-environment smoke path pass with no unresolved blocker or critical defect.

A single successful run is not sufficient.

## Mandatory repair-and-retest loop

1. Implement the smallest coherent vertical slice.
2. Run the narrowest relevant test immediately.
3. Preserve failure evidence.
4. Identify the actual responsible layer.
5. Fix the root cause.
6. Never disable a valid test, loosen a valid assertion, swallow an exception, hard-code a result, or mock away behavior the phase is intended to prove.
7. Rerun the narrow failing test.
8. Rerun the current phase suite.
9. Rerun all applicable cumulative regressions.
10. Repeat clean-environment smoke verification.
11. Only then mark the phase gate complete.

## Defect severity

| Severity | Meaning | Phase exit |
|---|---|---|
| BLOCKER | invalid research, data loss/corruption, serious security flaw, impossible core workflow | forbidden |
| CRITICAL | integrated workflow broken or materially incorrect | forbidden |
| MAJOR | important behavior degraded/wrong but core evidence remains valid | must be documented and scheduled |
| MINOR | cosmetic/non-blocking issue | may be documented |

## Global quality categories

- static quality: formatter/lint/type checks once tooling exists
- unit tests: pure contract/metric/schema logic
- property/invariant tests: split disjointness, hashes, metric bounds, idempotency
- integration: Postgres/pgvector, worker/Redis, APIs, migrations, external adapters via fakes
- end-to-end: browser/API/queue/worker/database/SSE through a deterministic smoke benchmark
- research regression: frozen tiny fixture with expected labels/metrics
- fresh environment: rebuild from empty state; hidden developer state cannot be required

## Phase 01 gate

Required:
- research protocol internally consistent
- exact base-model ID/revision frozen
- label taxonomy validates and has versioning rules
- unknown/duplicate labels rejected
- primary-comparison model mismatch rejected
- incomplete reproducibility identity cannot become RUNNING
- documentation/config validation passes
- full frozen-model weight load and one structured generation succeeds on suitable hardware

If the connected full-model smoke cannot be executed in the current environment, Phase 01 remains **in progress**, not complete.
