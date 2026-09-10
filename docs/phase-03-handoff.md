# Phase 03 Handoff — Dataset Engineering, Versioning & Leakage-Safe Benchmark

Status: **FINAL VALIDATION IN PROGRESS — RESEARCH HARD EXIT PASSES ON THE GENERATED LOCKED CORPUS; CLEANED-BRANCH NORMAL CI AND TARGET-MERGE VALIDATION STILL REQUIRED.**

Phase 03 now has a canonical leakage-safe research benchmark built from 18 independently sourced, checksum-preserved public production incidents. The frozen taxonomy remains unchanged at six labels. Each label has exactly three independent families, and deterministic family-level splitting with seed `20260908` produces one family per label in each research split.

## Locked research corpus

- Dataset version: `evalforge-incident-diagnosis-v0.1.0`
- Admission policy: `phase3-research-admission-v1`
- Canonical records: **18**
- Independent families: **18**
- Research train / validation / test: **6 / 6 / 6**
- Independent families per frozen label: **3**
- Families per label in each split: **1**
- Synthetic rows in locked base dataset: **0**
- Research leakage-audit errors: **0**
- Split seed: `20260908`
- Research content checksum: `da9292d0e550cb141dbf1048e1a7c02f2fb4f6bd74b8642e781f06dd93acdbc5`
- Research manifest identity checksum: `4a6767f0859eb93a2a9e7e8a8ea3caecb56b8a666c9ba9da97fee8e054c10c7d`

The base research dataset deliberately contains no synthetic rows. Any later augmentation must happen only after the locked family split and only from training-family parents.

## Provenance and admission

The public-incident review ledger contains **24 reviewed candidates**, including **18 supported mappings** and the retained rejected/ambiguous controls. The 18 supported candidates all have byte-preserved original primary-source evidence with SHA-256 validation and are explicitly admitted by the research admission plan. Preservation depth is exactly three supported independent families for every frozen label:

- `broken_payment_configuration`: 3
- `database_connection_leak`: 3
- `disk_exhaustion`: 3
- `memory_leak`: 3
- `n_plus_one_query`: 3
- `no_fault`: 3

The Visa/Cybersource payer-authentication artifact remains rejected for public research admission because its source carries confidentiality / redistribution restrictions. A public ElevenLabs payment-configuration incident replaces it, preserving the scientific requirement for three independent public families in `broken_payment_configuration` without republishing restricted material.

Research admission is downstream of preservation. The builder revalidates each admitted candidate against the committed primary-source manifest and checksum before constructing a canonical record and family.

## Leakage controls

The research lock enforces the incident family as the independence unit. Stable family IDs are derived before splitting, and the splitter operates only on families. The research audit checks:

- cross-split family overlap;
- duplicate incident IDs;
- exact and near-duplicate text across splits;
- taxonomy label/category validity;
- synthetic parent/family lineage;
- held-out identifier references in training data;
- independent-source eligibility for validation/test records;
- class distribution and difficulty-tier coverage.

The locked corpus passes the research audit with **zero errors**. Difficulty tiers remain `unknown` because the source set does not provide a defensible common difficulty rubric; the audit reports that limitation rather than inventing labels.

## Auxiliary and generated sources

OpsSentinel BenchmarkLab remains a generated engineering/controlled source and is not treated as independent held-out evidence. Its pinned semantic snapshot is used only through EvalForge's existing transport/materialization path. No OpsSentinel source-project files were modified.

UCI dataset 498 / ServiceNow remains a real operational auxiliary corpus but is not mapped into the frozen RCA taxonomy because its human-readable root-cause semantics are unavailable/anonymized. It therefore remains excluded from the primary labeled research train/validation/test benchmark.

## Reproducibility and contract validation

Guarded research-lock Run `34470939000` completed successfully and produced generated product commit `7800850abc1a4f91bd1e6e3ca7c0957ca54cd1c4`.

Within that run:

- explicit research admission passed with **18** admitted candidates;
- all **18** primary-source preservations revalidated successfully;
- candidate-artifact reproducibility passed;
- locked research-artifact reproducibility passed;
- focused Phase 03 data suite: **36 passed**;
- `app.data` coverage: **90.96%** (required threshold: 90%);
- Ruff lint and formatting passed;
- mypy reported no issues in **22 source files**;
- expanded Phase 03 contract passed;
- hard exit reported `PHASE03_EXIT_GATE=PASS`;
- research counts were exactly `18 records / 18 families / 6 train / 6 validation / 6 test`;
- `SYNTHETIC_BASE_RECORDS=0`;
- `LEAKAGE_ERRORS=0`.

Generated research artifact hashes from the guarded run:

- `incidents.jsonl`: `4d2e6f8688356b8ea57b6e113cfb34963bdc9e362b90d77d6b611c82ef5082b0`
- `families.jsonl`: `b12ff15fc5c24fb3e0ea40670e314435b36a70e69f903f589b421271e0462b5f`
- `manifest.json`: `1ec336e6ebf3d6735ba76128656d39713ce548e98964f7d436a38ae4c23faba1`
- `leakage-audit-report.json`: `f109620b7e535aa80eafeef0dc9ab52b51b454696e9953151f68c365bccb6a4f`
- `class-split-summary.json`: `8119ea207e7403244b3a0d3e650153269592a85e51a3218b21ef79f5b7263a62`
- `research-admission-summary.json`: `280555ad674869a1bfda35de5748344a42166b799dc17adf6dae8b08b8324797`

## Remaining closure gates

The scientific hard exit is passing on the generated product state, but Phase 03 is not considered complete until the temporary research-lock writer/migration helper are removed, the exact cleaned branch head passes the ordinary cumulative `make verify-all` path and fresh clean-Compose smoke, the change is merged into `phase-03-dataset-engineering`, and the exact target merge commit passes those same post-merge checks.

No hard gate has been weakened. Phase 04 has not started.