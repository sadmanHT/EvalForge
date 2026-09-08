# EvalForge Incident Diagnosis Data Card — Phase 03 Candidate

**Status:** Research benchmark **not yet lockable**. All six frozen labels now have at least one defensible independent candidate mapping, but preserved original-source evidence and per-label independent family depth are still insufficient for a credible locked holdout.

**Candidate dataset version:** `evalforge-incident-diagnosis-v0.1.0-candidate`  
**Canonical schema:** `incident-schema-v1`  
**Label taxonomy:** `1.0.0`  
**Split seed reserved for an eligible corpus:** `20260908`

## Purpose

EvalForge needs a leakage-safe incident-diagnosis benchmark for the same four primary study arms: zero-shot, RAG, fine-tuned, and combined. The independence unit is the incident family, not an individual row. Synthetic augmentation may occur only after family splitting and only from training families.

## Source inventory

Phase 03 tracks three primary provenance roles, with direct normalized public-incident snapshots attached to the candidate index.

| Source | Nature | Scale | EvalForge role | Primary labeled holdout eligible? |
| --- | --- | ---: | --- | --- |
| OpsSentinel BenchmarkLab | Programmatically generated controlled scenarios | 50 scenarios | controlled/source-candidate engineering data | No |
| UCI dataset 498 / ServiceNow incident-management event log | Real anonymized operational incident events | 141,712 events / 24,918 incidents | real-world auxiliary/robustness corpus | No |
| postmortems.app + directly reviewed public incidents | Public real-incident discovery plus normalized candidate snapshots | 11 conservatively reviewed candidates | independent incident candidate discovery | Not yet |

### OpsSentinel controlled source

The read-only snapshot of `sadmanHT/OpsSentinel` is pinned at commit:

`fae661fc1634aad6a3855a1dec8dcddb16a890dd`

The pinned BenchmarkLab release contains **50 programmatically generated scenarios**. EvalForge preserved a normalized semantic snapshot under:

`datasets/incident_diagnosis/raw/opssentinel/fae661fc1634aad6a3855a1dec8dcddb16a890dd/release-catalog.snapshot.json`

Snapshot SHA-256:

`b4828ac9793d64c7850c7bbd6c262d4e7a89db0fd09346a9ed72a48f82dc8241`

For fresh-clone transport, the exact snapshot is losslessly stored as deterministic gzip+base64 (`release-catalog.snapshot.json.gz.b64`, SHA-256 `7c00bfd3a1a2dd01f3848153c0d06f2f8f875a0588379cf02960e120754034fb`). `scripts/fetch_phase3_sources.py` reconstructs the original JSON bytes and verifies the snapshot SHA before the cumulative gate.

No OpsSentinel file was changed.

#### Upstream counts

| Upstream split | Scenarios |
| --- | ---: |
| dev | 30 |
| validation | 10 |
| hidden_test | 10 |
| **total** | **50** |

These are **upstream benchmark assignments**, not EvalForge research splits.

#### Primary root-cause distribution

| Root-cause code | Count |
| --- | ---: |
| `database_connection_leak` | 14 |
| `n_plus_one_query` | 11 |
| `broken_payment_configuration` | 9 |
| `disk_exhaustion` | 8 |
| `memory_leak` | 7 |
| `no_fault` | 1 |

#### Difficulty distribution

| Difficulty | Count |
| --- | ---: |
| easy | 10 |
| medium | 12 |
| hard | 12 |
| adversarial | 8 |
| compound | 8 |

The 50 OpsSentinel scenarios are constructed by deterministic benchmark-definition code. They are useful engineering fixtures and source candidates, but they are not independent production incident evidence. EvalForge infers generation-family identity rather than trusting individual scenario IDs. Under that stricter grouping, **three inferred generation families cross the upstream source splits**:

- `opssentinel:single_noisy_dependency:n_plus_one` — dev and validation
- `opssentinel:single_noisy_dependency:connection_leak` — dev and validation
- `opssentinel:misleading_change_temporal:connection_leak` — validation and hidden_test

Therefore the upstream dev/validation/hidden-test assignment must not be reused as the EvalForge research split.

### UCI / ServiceNow real operational auxiliary source

The reviewed UCI dataset 498 / ServiceNow archive is materialized at the following path by `python scripts/fetch_phase3_sources.py` (the user-provided equivalent archive remains preserved in the Phase 03 working evidence):

`datasets/incident_diagnosis/raw/servicenow_uci/uci-498/incident-management-process-enriched-event-log.zip`

Canonical inner `incident_event_log.csv` SHA-256:

`fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef`

Two ZIP transport envelopes were independently verified to contain that exact CSV payload:

- user-provided Kaggle/mirror ZIP: `3ea92768cb2cfada908dd601b057d7066127fe75dc8fe19e4abaa1c6766a6c13`
- official UCI download verified in GitHub Actions on 2026-09-08: `6294e29a311647306bfdfc85783f7df66517c197b9cd49aa5ee36ba9c525d1d6`

EvalForge therefore treats the inner CSV checksum as the scientific snapshot identity and the ZIP checksum as a transport-envelope integrity check. Fresh CI checkouts fetch the official reviewed envelope and reject any unreviewed transport or changed CSV payload before the cumulative gate. A different CSV payload is not accepted as the same source.

The source contains **141,712 event rows over 24,918 unique incidents** and 36 attributes. It is real operational data extracted from a ServiceNow instance and anonymized by the publishers. It is licensed CC BY 4.0 and identified by DOI `10.24432/C57S4H`.

Incident-level field availability in the preserved archive:

| Field | Incidents with a non-missing value |
| --- | ---: |
| `category` | 24,911 |
| `subcategory` | 24,910 |
| `u_symptom` | 19,405 |
| `problem_id` | 381 |
| `rfc` | 179 |
| `caused_by` | 3 |
| `closed_code` | 24,811 |
| `cmdb_ci` | 56 |

This source improves real-world coverage but **does not provide trustworthy human-readable RCA labels** for EvalForge's frozen taxonomy. The publishers omitted textual attributes and anonymized categorical values. EvalForge therefore does not guess that anonymous `Category N`, `Symptom N`, `closed_code`, `problem_id`, or RFC identifiers correspond to labels such as `memory_leak`, `n_plus_one_query`, or `database_connection_leak`.

It is retained as a real-world auxiliary/robustness corpus, not as the primary labeled research holdout.

### Public incident candidate discovery

EvalForge pins the public `icco/postmortems` / postmortems.app discovery corpus at commit:

`42bac673432f564d317dbfc30b5d400e1812c684`

The candidate decision index is `configs/postmortem-candidates.json` (index SHA-256 `7f8c2bfb0651e80aa00987bd8a938b55dc1a1fadc523b5679506338e4786fc2c`). postmortems.app remains a discovery/index source rather than automatic ground truth. Directly reviewed external candidates may instead point to normalized EvalForge snapshots under `datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/`; each such snapshot has a pinned Git blob and SHA-256 checksum and explicitly records that it is not a byte-for-byte original-source archive.

Eleven independent public-incident candidates have now been conservatively reviewed against the frozen taxonomy. Six are strong candidate mappings, one for every frozen label:

| Candidate | Proposed label | Decision | Why |
| --- | --- | --- | --- |
| Amazon EBS, 2012-10-22 | `memory_leak` | supported candidate | narrative explicitly identifies progressive agent memory consumption |
| Tarsnap, 2016-07-24 | `disk_exhaustion` | supported candidate | unbounded local log fills filesystem and service writes fail |
| Medoc, 2026-01-11 | `n_plus_one_query` | supported candidate | firsthand production narrative explicitly identifies N+1, 6,000+ DB calls, and batching as the fix |
| Dispatcharr #1416, 2026-07-06 | `database_connection_leak` | supported candidate | PostgreSQL connection is not returned on stream teardown; pool stays 8/8 and DB-dependent requests wedge until restart |
| Google Cloud payments, 2021-09-22 | `broken_payment_configuration` | supported candidate | official root cause is a payment-configuration update that prevents credit-card processing until rollback |
| Google Cloud / Mandiant, 2024-09-18 | `no_fault` | supported candidate | final official investigation reports no service degradation and no supported CrowdStrike alerts missed |
| incident.io GKE incident | `n_plus_one_query` | rejected as primary | N+1 join is a contributing issue; persistent root cause is `anetd` CPU saturation/packet loss |
| Twilio billing, 2013 | `broken_payment_configuration` | ambiguous | incorrect Redis configuration contributes, but the incident is a broader multi-causal chain |
| Elastic Cloud, 2019 | `database_connection_leak` | rejected as primary | connection leaks are Kibana/remediation issues, not established DB-connection root cause |
| Cloudflare parser incident | `memory_leak` | rejected false friend | memory disclosure from buffer over-read is not runtime memory exhaustion |
| Skyliner, 2017 | `memory_leak` | insufficient evidence | pinned index entry is too thin for a locked family/evidence record |

Candidate-level taxonomy coverage is therefore **6/6**, but this does **not** make the research benchmark ready. None of the eleven candidates is research-admitted. The three new direct snapshots are normalized semantic evidence rather than byte-for-byte originals; Amazon/Tarsnap originals are not yet preserved; Medoc is pinned to an external Git commit/blob but not yet materialized as an EvalForge original-source archive. In addition, one strong family per label is not enough to create a credible family-stratified train/validation/test split while retaining independent families in every required role.

## Current canonical research counts

| EvalForge research split | Eligible labeled records |
| --- | ---: |
| train | 0 |
| validation | 0 |
| test | 0 |

This is intentional. EvalForge does **not** manufacture held-out labels by rebranding generated OpsSentinel scenarios or by guessing semantics from anonymized ServiceNow codes.

## Required benchmark expansion

Before Phase 03 can be completed, EvalForge must expand and preserve the independent postmortem corpus until the frozen taxonomy has enough trustworthy family coverage to create a meaningful non-empty validation and locked test set while retaining enough training families. The actual counts must be determined from the acquired corpus; no predetermined “50 test cases” target is forced.

Each independent source incident must provide enough provenance to establish family identity, canonical root-cause label, evidence, and source eligibility. If multiple rows are descendants or variants of the same real incident, they must share one family and remain in one split.

## Planned split method for an eligible corpus

1. Normalize and validate canonical labels against taxonomy `1.0.0`.
2. Assign incident-family IDs before any augmentation.
3. Stratify deterministic family assignments by root-cause label using split seed `20260908`.
4. Lock train/validation/test family membership.
5. Only then generate synthetic descendants from training parents.
6. Never select prompts, retrieval settings, calibration parameters, fine-tuning hyperparameters, or checkpoints from test outcomes.

The implementation is in `backend/app/data/split.py`.

## Synthetic augmentation policy

A canonical synthetic descendant must:

- be in `train`;
- name a valid `parent_incident_id`;
- inherit the parent's `incident_family_id`;
- record generator model ID and exact revision;
- record generator prompt version;
- have a parent that is itself an independent training record.

Synthetic validation or test descendants are forbidden.

## Leakage audits

`python scripts/audit_dataset.py <dataset_dir> --research-mode` checks:

- cross-split family overlap;
- exact and near-duplicate incident text across splits;
- root-cause label/category validity;
- synthetic parent existence and split/family consistency;
- independently eligible source provenance for research validation/test records;
- forbidden held-out incident/family identifier references in training artifacts;
- class distribution and difficulty-tier coverage.

Severe imbalance and missing difficulty tiers are reported instead of silently hidden.

## CI smoke data

`datasets/incident_diagnosis/fixtures/ci_smoke/` is deterministic **non-research fixture data**. It exists only to test schemas, splitting/audit behavior, serialization, and augmentation lineage. It must never contribute a benchmark result.

## Known limitations

- Independent public-incident candidates now provide at least one strong candidate mapping for all six frozen labels, but none is research-admitted; original-source preservation and sufficient per-label independent family depth remain outstanding.
- The ServiceNow/UCI corpus is real-world operational data but lacks trustworthy human-readable root-cause semantics for the frozen taxonomy.
- The OpsSentinel BenchmarkLab catalog is generated and therefore cannot supply independent held-out evidence.
- The upstream OpsSentinel source assignments contain inferred generation-family overlap.
- RAG document provenance/eligibility is not implemented until its later phase; the Phase 03 dataset rules establish the held-out-family boundary it must respect.
- No model benchmark results are produced in Phase 03.
