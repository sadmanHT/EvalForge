# EvalForge Incident Diagnosis Data Card — v0.1.0

**Status:** Locked Phase 03 research benchmark.  
**Research dataset version:** `evalforge-incident-diagnosis-v0.1.0`  
**Candidate/provenance dataset:** `evalforge-incident-diagnosis-v0.1.0-candidate`  
**Canonical schema:** `incident-schema-v1`  
**Label taxonomy:** `1.0.0`  
**Research admission plan:** `phase3-research-admission-v1`  
**Family split seed:** `20260908`

## Purpose

EvalForge uses this dataset as the common leakage-safe benchmark for later zero-shot, RAG, fine-tuned, and combined incident-diagnosis experiments. The primary target is the frozen `root_cause_code`. The unit of statistical independence is the **incident family**, not a row and not a synthetic variant.

Phase 03 intentionally locks the family assignment before any augmentation. Later phases may add synthetic training descendants, but they may not move families, synthesize validation/test evidence, or use test outcomes to select prompts, retrievers, calibration settings, checkpoints, or fine-tuning hyperparameters.

## Frozen label taxonomy

The research corpus contains exactly three independent public production-incident families for each frozen label:

| Root-cause code | Category | Independent families |
| --- | --- | ---: |
| `n_plus_one_query` | `database_behavior` | 3 |
| `database_connection_leak` | `database_behavior` | 3 |
| `disk_exhaustion` | `resource_exhaustion` | 3 |
| `memory_leak` | `resource_exhaustion` | 3 |
| `broken_payment_configuration` | `configuration` | 3 |
| `no_fault` | `control` | 3 |
| **Total** |  | **18** |

Changing these labels requires an explicit taxonomy version bump and comparability review.

## Source and admission policy

The candidate ledger contains **24 reviewed public-incident candidates**. Eighteen have a supported narrative root-cause mapping, all eighteen have checksum-validated primary-source preservation, and exactly those eighteen are explicitly research-admitted by `configs/phase3-research-admissions.json`.

Preservation alone is not admission. The admission gate requires all of the following simultaneously:

- a supported mapping based on narrative root-cause evidence rather than keyword/title matching;
- a checksum-validated preserved primary-source artifact;
- membership in the explicit admission manifest;
- exactly three admitted independent families for every frozen label;
- no unsupported, unpreserved, or partially admitted candidate.

Each canonical record retains the original source URL, preserved artifact path, preservation kind, SHA-256, candidate ID, and mapping basis in its provenance/metadata. The record description and evidence are conservative human-reviewed evidence summaries, not newly inferred facts.

### Republication-restricted source handling

The Visa Acceptance / Cybersource April 28, 2026 payer-authentication PIR remains in the review ledger but is **not** research-admitted. Although publicly indexed, the source explicitly restricts copying/distribution. EvalForge therefore does not republish that primary document into the public repository. It is recorded as `rejected_as_primary_mapping` with `public_republication_restricted` as the mapping basis.

A separate public ElevenLabs April 22, 2026 payment incident supplies the third `broken_payment_configuration` family. Its first-party status report explicitly attributes payment/subscription provisioning impact to a billing-system misconfiguration.

## Locked split

`backend/app/data/split.py` performs deterministic family-stratified assignment from the frozen seed `20260908`. Family IDs are created before splitting and are stable functions of the admitted candidate identities.

Because the corpus has exactly three independent families per label, the locked assignment is intentionally balanced as follows:

| Split | Independent families | Canonical records | Families per label |
| --- | ---: | ---: | ---: |
| train | 6 | 6 | 1 |
| validation | 6 | 6 | 1 |
| test | 6 | 6 | 1 |
| **Total** | **18** | **18** | **3** |

Every canonical family currently contains one independently sourced incident record. No family ID appears in more than one split.

The actual deterministic assignment is stored in:

- `datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0/incidents.jsonl`
- `datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0/families.jsonl`
- `datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0/research-admission-summary.json`
- `datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0/class-split-summary.json`

## Synthetic augmentation policy

The locked v0.1.0 base dataset contains **zero synthetic records**.

Any later synthetic descendant must be created only after this split and must:

- belong to `train`;
- name a valid `parent_incident_id`;
- inherit the parent's `incident_family_id`;
- record generator model ID and exact revision;
- record generator prompt version;
- have an independent training record as its parent.

Synthetic validation/test examples are prohibited. Synthetic volume never increases the number of independent families or the amount of independent held-out evidence.

## Leakage audit

The research-mode audit in `backend/app/data/audit.py` and `scripts/audit_dataset.py` checks:

- cross-split family overlap;
- exact duplicate and near-duplicate incident text across splits;
- canonical label/category validity;
- synthetic parent existence and train/family consistency;
- source eligibility for validation/test records;
- forbidden held-out incident/family identifier references in training artifacts;
- family membership/split consistency;
- class distribution and difficulty-tier coverage.

The committed v0.1.0 benchmark must have **zero error-severity audit findings** before `research_ready=true` is written. The generated leakage report is archived at:

`datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0/leakage-audit-report.json`

The only expected distribution warning is `MISSING_DIFFICULTY_TIERS`: the reviewed sources do not provide a defensible common difficulty rubric, so Phase 03 uses `unknown` rather than inventing easy/medium/hard labels.

## Manifest and reproducibility

The research manifest records:

- dataset version;
- canonical schema version;
- frozen taxonomy version;
- all 18 source snapshot identities and SHA-256 values;
- split seed;
- deterministic generation timestamp derived from the latest admitted primary-source capture timestamp;
- content checksum;
- manifest checksum;
- record/family and split counts;
- explicit research-ready status and limitations.

Rebuilding from the same committed configs and raw public-source snapshots must reproduce byte-identical:

- `incidents.jsonl`;
- `families.jsonl`;
- `manifest.json`;
- `leakage-audit-report.json`;
- `class-split-summary.json`;
- `research-admission-summary.json`.

Use:

```bash
python scripts/admit_phase3_research_records.py
python scripts/build_phase3_research_dataset.py
python scripts/check_phase3_research_reproducibility.py
python scripts/check_phase3_exit.py
make verify-all
```

The admission command above is verification-only in the committed state; mutation requires the explicit one-time `--apply` transition from candidate-index v7 to v8.

## Other reviewed data sources

### OpsSentinel BenchmarkLab

The pinned EvalForge input references `sadmanHT/OpsSentinel` at commit `fae661fc1634aad6a3855a1dec8dcddb16a890dd`. Its 50 BenchmarkLab scenarios are programmatically generated. They remain useful controlled engineering fixtures but are **not** independent research holdout evidence and are excluded from the 18 research families. EvalForge does not modify the OpsSentinel project.

The source also contains inferred generation-family overlap across its upstream dev/validation/hidden-test assignments, so those upstream split labels are not reused for EvalForge research evaluation.

### UCI / ServiceNow dataset 498

The auxiliary ServiceNow event log contains 141,712 events over 24,918 incidents. The canonical `incident_event_log.csv` SHA-256 is `fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef`.

It is real operational data, but the anonymized categorical fields do not provide trustworthy human-readable RCA labels for EvalForge's frozen taxonomy. It is therefore retained only as an auxiliary/robustness corpus and contributes zero independent research families to v0.1.0.

### Public postmortem discovery ledger

The `icco/postmortems` discovery corpus is pinned at commit `42bac673432f564d317dbfc30b5d400e1812c684`. postmortems.app is used for discovery and provenance review, not as automatic label ground truth. Direct external incidents are also reviewed and normalized into the same candidate ledger. The primary source for every admitted family is separately preserved and checksum-validated.

## CI smoke data

`datasets/incident_diagnosis/fixtures/ci_smoke/` is deterministic **non-research fixture data**. It exists only to exercise schema, split, audit, serialization, and augmentation invariants in CI. It must never contribute a benchmark result.

## Known limitations

- **Small independent sample:** three families per label is the minimum depth that permits one independent family for each label in train, validation, and test. Leakage safety does not make this a large statistical sample.
- **One training family per label:** later model optimization must be interpreted cautiously. Synthetic descendants may increase training examples but not independent evidence.
- **Difficulty tiers unavailable:** all v0.1.0 records use `unknown`; this limitation is surfaced by the audit rather than concealed.
- **Source heterogeneity:** reports vary in detail, publisher style, incident age, and operational context. Canonicalization preserves conservative evidence summaries and provenance instead of forcing unsupported metadata.
- **No Phase 03 model results:** this phase establishes the benchmark only. It makes no claim about which model/pipeline performs best.
- **RAG leakage boundary established, not yet indexed:** later retrieval work must exclude held-out-family historical incident material from train-time/tuning retrieval. RAG implementation belongs to a later phase.

## Evaluation restrictions

The test split is locked. Later phases may not choose prompts, retrieval settings, calibration parameters, checkpoints, adapters, or fine-tuning hyperparameters from test outcomes. All four primary pipelines must ultimately use the same frozen held-out incident IDs and deterministic evaluation code.

No primary metric may depend on an LLM judge when deterministic root-cause labels exist. No synthetic descendant may be presented as independent validation/test evidence.
