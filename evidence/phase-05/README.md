# Phase 05 evidence — Evaluation Harness & Metric Correctness

This directory preserves committed, reviewable evidence for the Phase 05 common evaluator. Phase 05 defines how later experiment arms are scored; it does not claim real zero-shot, RAG, fine-tuned, or combined benchmark results.

## Frozen evaluator identities

- evaluator: `phase5-evaluator-v1`
- failure taxonomy: `phase5-failure-taxonomy-v1`
- golden fixture: `phase5-golden-metrics-v1`
- golden result hash: `ffd7a9297b0e035e00d3620605ffc0eaa839afc6a4223e1f2d0960ad1c3ec6b4`

## Independent golden calculations

`golden-fixture-calculations.md` derives the expected metric values used by `backend/tests/fixtures/phase5/golden_metrics.json`. The implementation is tested against these externally stated values rather than generating its own expected answers.

The fixture covers exact/hierarchical/top-k correctness, normalized multiclass Brier score, fixed-bin ECE/reliability, latency quantiles, and marginal/amortized cost.

## Verified implementation checkpoint

GitHub Actions run `34617761405` on SHA `a4bd5b8489e1d190c4d0dafbbf8d7021d2889c8e` passed both `verify-all` and `clean-compose`.

Key Phase 05 results:

- focused evaluator + PostgreSQL persistence tests: `35 passed`
- `app.evaluation` coverage: `94.00%` (required >= `90%`)
- full PostgreSQL integration suite: `14 passed`
- mypy: no issues in `45` source files
- smoke predictions persisted: `6`
- persisted metrics: `10`
- smoke/live/reload-rescore result hash: `93bcade544e1b57709cb6c08260e588c88abe66f7d4b99be7be33fdb1c483a3f`
- calibration fitting from `test`: rejected
- Phase 05 hard exit: PASS
- fresh no-cache Compose build/smoke: PASS
- clean teardown: PASS

## Canonical smoke metrics

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

The smoke corpus is deterministic validation infrastructure, not a scientific estimate of model quality.

## Runtime evidence

CI uploads preserve:

- Phase 05 hard-exit output;
- this committed evidence directory;
- `docs/evaluation-methodology.md`;
- the golden fixture;
- clean-environment Compose logs/evidence.

The implementation checkpoint artifact `phase05-evaluation-evidence` from run `34617761405` had ZIP SHA-256 `fc39e926c06de72d03739023c8b9424505ba6227daa530a11eab3fd6f08bb9ed`.

## Scope notes

The evaluator deliberately keeps primary deterministic root-cause correctness separate from supporting hierarchical, retrieval, and judge-style metrics. Malformed outputs remain persisted/scored outcomes. Temperature scaling is validation-only. Paired comparisons require the same held-out incident IDs.

No OpsSentinel source-project changes are part of Phase 05. Phase 06 has not started.
