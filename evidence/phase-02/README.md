# Phase 02 Evidence

Status: **COMPLETE — hard gates preserved.**

Phase 02 completion is grounded in GitHub Actions run #7 on the lock-enforced implementation head `70ce01d825c9112890bb9cfcd2164a3f3bfdd1a4`:

- Run: https://github.com/sadmanHT/EvalForge/actions/runs/34251390646
- `verify-all`: PASS
- `clean-compose`: PASS
- backend frozen lock: `backend/requirements.full.lock`
- frontend complete lock: `frontend/package-lock.json`
- backend lock SHA-256: `917fb1f22c38b666d0baadcfaa8a1d9b22d424b0991321e7c622c0da45ae8a95`
- frontend lock SHA-256: `eb539321a5d51c55dd9d672d0a5bc1b2992bf83cf1eac191d9a14c1c6d6aaf75`

## Preserved repository evidence

- `final-gate-summary.txt`: exact gate markers, test counts, run/head identifiers, lock hashes, and artifact digests.
- `fresh-start-smoke.log`: worker state smoke plus HTTP and dependency checks from the clean Compose run.
- `service-health.txt`: point-in-time Compose service health snapshot. The frontend was still transitioning in `docker compose ps`, while the independent smoke client had already returned HTTP 200.
- Existing `ci-run-0*-failure.txt` files preserve repaired intermediate CI defects instead of hiding them.

## GitHub Actions artifacts

Run #7 uploaded:

- `phase02-dependency-locks`, artifact ID `10066157289`, SHA-256 `7998520af78da900f103d41bf38ede891400b962518c25354112f8b8aa83412b`.
- `phase02-fresh-start`, artifact ID `10066229661`, SHA-256 `ff51abf5c1b3790cfeb4f1dd174ce583541346560d88fc8641b91b9b9a4b213c`.

The fresh-start artifact includes full Compose logs in addition to the two snapshots copied into this directory.

## Cumulative regression evidence from Run #7

- Phase 01 unittest contracts: 51 PASS.
- Phase 01 static validation: PASS.
- Phase 01 hard exit: PASS with the frozen Mistral revision and prior real T4 smoke evidence.
- Backend non-integration tests: 6 PASS.
- Backend integration tests: 3 PASS.
- Frontend tests: 2 PASS.
- Phase 02 foundation checker: PASS.
- Alembic upgrade: PASS.
- Frontend production build: PASS.
- Tracked-secret scan: PASS.

Known dependency deprecation/future-flag warnings were non-blocking and no test or gate was weakened to obtain a pass. Phase 01 evidence remains immutable under `evidence/phase-01/`.
