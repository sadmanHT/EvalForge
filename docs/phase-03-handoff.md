# Phase 03 Handoff — Dataset Engineering, Versioning & Leakage-Safe Benchmark

Status: **IN PROGRESS — HARD EXIT BLOCKED ON INDEPENDENT FAMILY DEPTH / ORIGINAL-SOURCE PRESERVATION.**

Phase 03 implementation logic is present and regression-tested. The hard exit gate is intentionally not marked complete. Independent public incident evidence now has at least one defensible candidate mapping for all six frozen labels, but no candidate is research-admitted and the corpus still lacks the preserved originals and per-label independent family depth required for a credible locked split.

## Implemented

- `incident-schema-v1` runtime `IncidentRecord` and `IncidentFamily` validation.
- Frozen taxonomy `1.0.0` validation without changing Phase 01 labels.
- Deterministic family-stratified split library with seed `20260908`.
- Stable dataset-content and manifest hashing.
- JSONL round-trip loaders.
- Train-only synthetic augmentation lineage contract and `training/dataset_prep.py` hook.
- Leakage audit for family overlap, exact/near duplicates, label/category validity, parent/child lineage, held-out references, research-source eligibility, class balance, and difficulty coverage.
- Read-only OpsSentinel source snapshot provenance pinned to commit `fae661fc1634aad6a3855a1dec8dcddb16a890dd`.
- UCI/ServiceNow dataset 498 deterministic source materialization with source/license/DOI metadata, reviewed transport hashes, canonical inner-CSV identity, and structural inspection.
- Explicit separation between real-world auxiliary evidence and primary labeled research-holdout eligibility.
- Deterministic source-candidate inspection and reproducible candidate manifest.
- Non-research deterministic CI smoke dataset.
- Data card with actual source counts and limitations.

## Source evidence

### OpsSentinel

- Source scenarios: **50 generated scenarios**.
- Upstream source split counts: dev 30 / validation 10 / hidden_test 10.
- Inferred generation families: 29.
- Upstream generation families crossing source splits: **3**.
- Raw semantic snapshot SHA-256: `b4828ac9793d64c7850c7bbd6c262d4e7a89db0fd09346a9ed72a48f82dc8241`.

### UCI source materialization and transport-equivalence verification

GitHub Actions diagnostic run `34262191383` verified the current official UCI dataset 498 download on 2026-09-08. The official ZIP SHA-256 is `6294e29a311647306bfdfc85783f7df66517c197b9cd49aa5ee36ba9c525d1d6`; its sole `incident_event_log.csv` member is 46,212,397 bytes with SHA-256 `fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef`, exactly matching the member in the user-provided Kaggle/mirror archive. EvalForge therefore treats the CSV payload hash as the stable scientific source identity and validates ZIP envelopes separately.

### Public incident coverage review

- Pinned discovery repository: `icco/postmortems` at `42bac673432f564d317dbfc30b5d400e1812c684`.
- Conservatively reviewed independent candidates: **23**.
- Supported candidate mappings: **18**, covering all frozen labels (`broken_payment_configuration`, `database_connection_leak`, `disk_exhaustion`, `memory_leak`, `n_plus_one_query`, `no_fault`).
- Candidate-level frozen-label gaps: **none**.
- Research-admitted public-incident records: **0**.
- Cloudflare's parser incident remains rejected as a `memory_leak` false friend because it is a buffer over-read/data-disclosure bug, not runtime resource exhaustion.
- incident.io's accidental N+1 join remains rejected as the incident's primary root cause because the persistent failure is GKE Dataplane V2 `anetd` CPU saturation/packet loss.
- Medoc remains a strong `n_plus_one_query` candidate pinned to exact external Git evidence, but not an EvalForge-preserved original-source archive.
- Dispatcharr #1416 is accepted as a strong `database_connection_leak` candidate because the report directly traces the outage to PostgreSQL connections not returned on stream teardown, shows a pool pinned at 8/8 after streams end, provides deterministic reproduction and a negative control, and notes restart recovery. Its closure was an automated issue-template action rather than a technical rejection.
- Google Cloud incident `E18Caoo5X1m6dTa1PVr1` is accepted as a strong `broken_payment_configuration` candidate because the official root cause is a payment-configuration update that stopped affected credit-card processing until rollback.
- Google Cloud / Mandiant incident `fLYHLzSGXGkLkAjc8MJG` is accepted as a `no_fault` negative-control candidate because the final investigation reports no service degradation and no supported CrowdStrike alerts missed. The mapping is limited to that declared supported-service scope.
- The three new direct-source entries are preserved as normalized EvalForge semantic snapshots with Git-blob and SHA-256 checks, but `original_source_snapshot_preserved=false` remains correct because they are not byte-for-byte original-source archives.


Family-depth milestone: **18 supported candidate families = 3 per frozen label**. This meets the deterministic splitter's minimum depth for one train, one validation, and one test family per label, but does not admit any research row. Original primary-source preservation, canonicalization, leakage audit, and final split construction remain required.

The candidate review is preserved at `datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0-candidate/public-postmortem-candidate-coverage.json`.

### UCI / ServiceNow

- Real operational event rows: **141,712**.
- Unique incidents: **24,918**.
- Canonical CSV payload SHA-256: `fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef`.
- Accepted transport ZIP SHA-256 values: user-provided mirror `3ea92768cb2cfada908dd601b057d7066127fe75dc8fe19e4abaa1c6766a6c13`; official UCI verified 2026-09-08 `6294e29a311647306bfdfc85783f7df66517c197b9cd49aa5ee36ba9c525d1d6`.
- Inner CSV SHA-256: `fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef`.
- DOI: `10.24432/C57S4H`.
- License: CC BY 4.0.
- Primary labeled research-holdout eligibility: **false** because textual RCA semantics are unavailable/anonymized.

Canonical eligible research train/validation/test counts remain **0 / 0 / 0**. The sources are preserved for their valid roles without fabricating labels.

## Focused verification

Local focused Phase 03 suite:

`python -m pytest backend/tests/data/test_phase3_data.py -q`

Result: **18 passed**.

Critical dataset module coverage:

`python -m pytest backend/tests/data/test_phase3_data.py --cov=app.data --cov-report=term-missing --cov-fail-under=90 -q`

Result: **91.36% overall `app.data` coverage**.

Reproducibility and contract checks:

- `PHASE03_REPRODUCIBILITY=PASS`
- `PHASE03_CONTRACT=PASS`
- CI smoke records: 7
- CI smoke families: 6

## Remote CI materialization history

- Full Phase 03 implementation commit: `dfd5a59475fadd49ef51d06a23edfe1c8995f400`.
- Exact-head GitHub Actions run `34270444416` passed source materialization, locked backend/frontend installs, and dependency-lock verification, then failed at the cumulative gate on eight Ruff findings before tests or clean Compose could run.
- A guarded one-off helper workflow applied only the Ruff-reported repairs, required Ruff plus the focused **18-test** Phase 03 suite and an exact changed-file whitelist to pass, then pushed lint-only commit `1c92ed06bb26f9f4b2b77ed4d03ba04b64eb35a4` directly on top of the implementation commit.
- Exact-head run `34276634516` then passed source materialization, both locked installs, dependency-lock verification, and Ruff lint, but `ruff format --check` identified six formatting-only files and stopped the cumulative gate before later tests or clean Compose.
- Guarded formatter run `34276839484` required the complete Phase 03 Python Ruff lint and format scopes, the focused **18-test** suite, and an exact six-file whitelist to pass before pushing formatting-only commit `48321ebf57a6969754c36ae88fe78a71dad1a87a`.
- Exact-head run `34277009480` then passed source materialization, locked installs, dependency-lock verification, Python lint/format, frontend Prettier, and frontend ESLint. It stopped at `mypy app` because three `Counter` variables in `backend/app/data/opssentinel.py` lacked explicit generic annotations.
- Guarded type-repair run `34277344415` applied only those three annotations, then required complete Phase 03 Python Ruff lint/format, `mypy app`, the focused **18-test** suite, and an exact one-file whitelist to pass before pushing type-only commit `d98b5c8aaf50fcf01172020bbdb359c5f43e4532`.
- The repair commits were produced by the GitHub Actions token, so GitHub did not recursively trigger the normal CI workflow from those pushes. This handoff update records the verified repair chain and provides a normal user-authored push so the safe read-only CI workflow can validate the resulting exact head end-to-end.
- Exact-head push CI Run `34277519206` (Run #22) completed SUCCESS: cumulative `make verify-all`, strict known-blocker recording, evidence upload, and fresh no-cache Compose build/smoke/teardown all passed on `0ddc04187b4a490ca9a08ce07de141f4b3f90fa4`.
- Draft PR #3 then triggered PR-context Run `34278144995` (Run #24); both `verify-all` and `clean-compose` completed SUCCESS on the same validated head.
- Medoc evidence validation head `f74a2c85a1b6242e2d72e54df04497b9b4391d0d` passed push Run `34280566223` (Run #28) and PR-context Run `34280571091` (Run #29); `verify-all` and fresh `clean-compose` were SUCCESS in both contexts.

No hard gate was weakened during this repair sequence.

## Hard exit status

`python scripts/check_phase3_exit.py`

Expected current result:

```text
PHASE03_EXIT_GATE=BLOCKED
BLOCKER=original_source_preservation_and_research_admission_required
SOURCE_CANDIDATES=50
AUXILIARY_REAL_INCIDENTS=24918
INDEPENDENT_POSTMORTEM_CANDIDATES=23
SUPPORTED_MAPPING_CANDIDATES=18
SUPPORTED_TAXONOMY_LABELS=6
MISSING_TAXONOMY_LABELS=
BLOCKER_DETAIL=original_primary_source_preservation_and_canonical_research_admission_required
RESEARCH_TRAIN=0
RESEARCH_VALIDATION=0
RESEARCH_TEST=0
```

This is a scientific blocker, not a disabled test. The Phase 03 brief prohibits solving a weak/unlabeled source corpus with synthetic test evidence or invented mappings.

## Required next input to finish Phase 03

Additional independently sourced incident families, plus preserved original-source evidence for accepted candidates, until taxonomy `1.0.0` has enough clean family coverage for a credible train/validation/test split. Once available, the existing import/split/audit code must create and lock the real manifest, run the full audit, rerun all Phase 1–3 regression/integration tests and `make verify-all`, then rerun the fresh Compose smoke before this status may become COMPLETE.

## Intentionally deferred

Persistence tables, evaluator metrics, model inference pipelines, RAG indexing, fine-tuning, benchmark execution, and dashboard result views remain later-phase work.

No OpsSentinel files were modified. Phase 04 has not started.
