# EvalForge Incident / Label Schema v1

## Purpose

This file defines the minimum semantic schema required before Phase 03 creates the versioned benchmark. It freezes names and provenance expectations without pretending the final dataset already exists.

## Canonical label

Each incident has:
- `root_cause_code`: canonical label ID from `configs/label-taxonomy.yaml`
- `root_cause_category`: EvalForge category mapped by taxonomy version
- optional `secondary_root_cause_codes` for compound cases

Unknown root-cause codes are invalid for a versioned benchmark unless the taxonomy is explicitly version-bumped first.

## Incident identity

Phase 03 records at minimum:
- `incident_id`
- `incident_family_id`
- `source`
- `source_version` / upstream commit where applicable
- `difficulty`
- `root_cause_code`
- `secondary_root_cause_codes`
- `evidence`
- `provenance`
- `split`
- `synthetic`
- `parent_incident_id` when synthetic
- schema/taxonomy version

## Split invariant

`incident_family_id` is the unit of independence. A family may occur in only one of train, validation, or test. Synthetic examples must inherit a training-family ID and may never originate from validation/test families.

## Upstream boundary

The Phase 01 label snapshot records observed root-cause codes from external repository `sadmanHT/OpsSentinel` at commit `40467b27085130c110098cac5077fb62dee4e5aa`. EvalForge does not write to or mutate that repository. Future upstream changes require an explicit EvalForge import/taxonomy version decision.
