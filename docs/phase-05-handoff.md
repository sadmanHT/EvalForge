# Phase 05 handoff — Evaluation Harness, Metric Correctness & Statistical Comparison

## Status

Phase 05 implementation is functionally complete and passed its implementation hard-exit criteria on verified branch head `a4bd5b8489e1d190c4d0dafbbf8d7021d2889c8e` in GitHub Actions run `34617761405`.

This handoff is part of the documentation-closure commit. Phase 05 must not be declared complete until the final cleaned Phase 05 head also passes normal CI plus fresh Compose, is merged into the Phase 05 target branch, and the exact target merge SHA passes the same post-merge validation. Phase 06 has not started.

## What Phase 05 adds

Phase 05 defines one common evaluation contract for all four primary study arms without running the real zero-shot, RAG, fine-tuned, or combined benchmark.

The implementation adds:

- a frozen, pipeline-independent canonical `Prediction` contract;
- scored parse-failure outcomes for malformed, missing-label, unknown-label, and partial model outputs;
- deterministic primary exact root-cause accuracy;
- supporting hierarchical and top-3 accuracy;
- normalized multiclass Brier score, fixed-width ECE, and reliability-diagram data;
- deterministic confidence from normalized label-sequence log-likelihoods;
- validation-only temperature scaling, with test-set fitting explicitly rejected;
- latency p50/p95 and marginal/amortized cost metrics;
- deterministic retrieval context precision/recall when reference chunk IDs exist;
- a supporting-only versioned RAGAS adapter namespace;
- paired bootstrap confidence intervals and exact McNemar/binomial testing;
- a versioned deterministic/manual failure taxonomy;
- one evaluator registry shared by all pipeline variants;
- persistence/reload/rescore through the existing Phase 04 PostgreSQL schema;
- independent hand-worked golden fixtures and a Phase 05 hard-exit checker;
- a six-incident real-PostgreSQL smoke evaluation used by cumulative CI and clean Compose;
- focused correctness, validation-edge, and persistence integration tests.

## Canonical prediction contract

Every model pipeline must emit the same `Prediction` structure before scoring. It records:

- incident identity;
- predicted root-cause code;
- ranked labels;
- optional full label-probability distribution;
- confidence probability plus a named confidence source;
- reasoning/evidence citations and retrieved chunk IDs;
- token counts, latency, and per-prediction cost;
- raw model output;
- parse status and parse error;
- pipeline-specific metadata that does not alter scoring semantics.

Malformed output is not an evaluator exception. It is persisted as a prediction with a non-`OK` parse status, counts as incorrect on the primary metric, and remains available for failure analysis.

## Primary and supporting correctness

The primary correctness metric is exact root-cause-code accuracy. A prediction is correct only if its parse status is `OK` and its predicted root-cause code exactly matches the frozen reference label.

Supporting metrics are kept separate:

- hierarchical accuracy compares frozen taxonomy categories;
- top-3 accuracy checks whether the reference code appears among the first three ranked labels;
- supporting judge/RAGAS metrics live under a versioned `supporting.ragas.*` namespace and cannot replace the deterministic primary metric.

## Confidence and calibration

Phase 05 does not treat an arbitrary model-reported number as confidence. Supported confidence is derived reproducibly from one sequence log-likelihood per allowed label, normalized over exactly the frozen label set.

Temperature scaling is supported only when fit on `validation`. The implementation rejects `test` as a calibration-fitting source. This invariant is enforced by unit tests and the Phase 05 hard-exit checker.

The normalized multiclass Brier score is:

`mean_i [0.5 * sum_k (p_ik - y_ik)^2]`

The `0.5` factor gives the valid probability-vector score a documented `[0,1]` bound.

ECE uses deterministic fixed-width confidence bins on `[0,1]`; confidence `1.0` belongs to the final bin. Reliability-bin count, empirical accuracy, mean confidence, and absolute gap are retained in evaluator output and persisted ECE metadata.

## Latency, cost, retrieval, and paired statistics

Latency p50 and p95 use the deterministic R-7 linear quantile definition.

Cost reporting distinguishes:

- marginal cost/query: model/evaluation cost accumulated per prediction;
- amortized cost/query: `(upfront_cost + marginal_total) / N`, allowing later fine-tuning phases to include training cost explicitly.

Retrieval context precision/recall are macro-averaged only over incidents with reference-relevant chunk IDs; missing references are not silently scored as zero relevance.

Paired comparison requires identical held-out incident IDs and supports:

- deterministic paired-bootstrap confidence intervals for accuracy difference;
- two-sided exact McNemar/binomial testing on discordant pairs.

Point-estimate differences alone are not labeled statistically meaningful.

## Failure taxonomy

Failure taxonomy version: `phase5-failure-taxonomy-v1`.

Deterministic rules can assign:

- `HALLUCINATED_EVIDENCE`;
- `CORRECT_CATEGORY_WRONG_CAUSE`;
- `OVERCONFIDENT_WRONG`;
- `UNDERCONFIDENT_CORRECT`;
- `RETRIEVAL_MISS`;
- `OTHER_UNCLASSIFIED`.

`ANCHORING` and `INSUFFICIENT_CONTEXT` deliberately require manual or secondary annotation with explanatory notes rather than weak automatic heuristics.

## Persistence and reload identity

Phase 05 reuses the Phase 04 relational schema instead of creating a second results store.

- canonical payloads are written to `predictions.output_json`;
- scalar confidence and latency remain available in typed prediction columns;
- evaluator metrics are written to `metrics`;
- failure labels are written to `failure_annotations`;
- deterministic IDs prevent silent identity drift;
- every persisted metric carries the evaluation result hash;
- stored canonical predictions can be reloaded and rescored through the same harness.

The persistence integration test and smoke gate require the live evaluation, persisted snapshot, and reload/rescore path to reproduce the same result identity and metric map.

No Phase 05 Alembic migration was needed because the Phase 04 schema already supplied the required canonical persistence surfaces.

## Golden fixture

The independent fixture is `backend/tests/fixtures/phase5/golden_metrics.json`.

Its hand calculations are documented in `evidence/phase-05/golden-fixture-calculations.md`. Expected values are not generated from the implementation under test.

The golden result hash in the verified implementation run is:

`ffd7a9297b0e035e00d3620605ffc0eaa839afc6a4223e1f2d0960ad1c3ec6b4`

## Validation evidence

Verified implementation checkpoint:

- SHA: `a4bd5b8489e1d190c4d0dafbbf8d7021d2889c8e`
- Actions run: `34617761405`
- `verify-all`: PASS
- `clean-compose`: PASS

Key results from `verify-all`:

- Ruff lint: PASS
- Ruff format: `74 files already formatted`
- frontend formatting/lint/typecheck: PASS
- mypy: no issues in `45` source files
- backend non-integration suite: `92 passed`, `14 deselected`
- Phase 03 focused suite: `36 passed`, `90.96%` `app.data` coverage
- experiment-config suite: `16 passed`, `97.53%` coverage
- Phase 05 focused + PostgreSQL persistence suite: `35 passed`, `94.00%` `app.evaluation` coverage
- full real-PostgreSQL integration suite: `14 passed`
- frontend tests: `2 passed`
- repository unittest discovery: `51` tests, PASS
- Phase 01 exit: PASS
- Phase 02 foundation: PASS
- Phase 03 contract/reproducibility/hard exit: PASS
- Phase 04 persistence seed/hard exit: PASS
- Phase 05 smoke/hard exit: PASS
- frontend production build: PASS
- secret scan: PASS

Phase 05 smoke output:

- `RESULT_HASH=93bcade544e1b57709cb6c08260e588c88abe66f7d4b99be7be33fdb1c483a3f`
- `PREDICTIONS=6`
- `primary.exact_accuracy=1`
- `supporting.hierarchical_accuracy=1`
- `supporting.top3_accuracy=1`
- `quality.parse_failure_rate=0`
- `calibration.ece=0.3`
- `calibration.normalized_multiclass_brier=0.054`
- `latency.p50_ms=22.5`
- `latency.p95_ms=24.75`
- `cost.marginal_mean_usd=0.001`
- `cost.amortized_mean_usd=0.001`

Phase 05 hard-exit output:

- `PHASE05_EXIT_GATE=PASS`
- `EVALUATOR_VERSION=phase5-evaluator-v1`
- `FAILURE_TAXONOMY_VERSION=phase5-failure-taxonomy-v1`
- `GOLDEN_FIXTURE=phase5-golden-metrics-v1`
- `GOLDEN_RESULT_HASH=ffd7a9297b0e035e00d3620605ffc0eaa839afc6a4223e1f2d0960ad1c3ec6b4`
- `PERSISTED_SMOKE_RESULT_HASH=93bcade544e1b57709cb6c08260e588c88abe66f7d4b99be7be33fdb1c483a3f`
- `PERSISTED_PREDICTIONS=6`
- `PERSISTED_METRICS=10`
- `CALIBRATION_TEST_FIT_REJECTED=true`

The service logs from integration tests can contain deliberate FK/immutability errors generated by Phase 04 negative tests. Those errors are expected evidence that invalid writes are rejected; the integration suite itself passed.

## Fresh Compose proof

Run `34617761405` destroyed prior Compose state, rebuilt without cache, started the stack against fresh volumes, executed the smoke/exit path, captured logs, and tore the stack down cleanly.

The `clean-compose` job passed all steps, including Phase 05 clean-environment evidence upload.

## Evidence locations

- `docs/evaluation-methodology.md`
- `docs/phase-05-handoff.md`
- `evidence/phase-05/README.md`
- `evidence/phase-05/golden-fixture-calculations.md`
- `backend/tests/fixtures/phase5/golden_metrics.json`
- `backend/tests/evaluation/`
- `backend/tests/integration/test_phase5_evaluation_persistence.py`
- `scripts/run_phase5_smoke_evaluation.py`
- `scripts/check_phase5_exit.py`

GitHub Actions run `34617761405` preserved the Phase 05 evaluation evidence and clean-environment evidence artifacts.

## Known scope boundaries

Phase 05 is an evaluator/methodology phase, not a benchmark-results phase. It does not claim:

- real zero-shot benchmark performance;
- real RAG benchmark performance;
- fine-tuned model performance;
- combined RAG + fine-tuning performance;
- production model-serving performance;
- production retrieval quality beyond the deterministic evaluator contract.

Those later phases must emit the same canonical prediction contract and use this common evaluator rather than implementing pipeline-specific scoring.

No OpsSentinel source-project changes were made in Phase 05. Phase 06 has not started.
