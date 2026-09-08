# EvalForge Incident Dataset Schema — Initial Contract

**Schema version:** `incident-schema-v1`  
**Taxonomy:** `configs/label-taxonomy.yaml`

Phase 01 defines semantics only. Phase 03 implements ingestion, manifests, family-safe splitting, augmentation lineage, and leakage audits.

## Incident record

Required semantic fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `incident_id` | string | Stable unique incident identifier. |
| `incident_family_id` | string | Independence/split unit; all descendants of one family stay in one split. |
| `root_cause_code` | string | Canonical taxonomy label. |
| `root_cause_category` | string | Category implied by the taxonomy label. |
| `difficulty_tier` | string | Versioned benchmark difficulty label when available. |
| `evidence` | array/object | Ground-truth evidence references or normalized evidence fields. |
| `source_provenance` | object | Source/version information sufficient for audit. |
| `is_synthetic` | boolean | Whether the row was generated rather than sourced as an independent real incident. |
| `parent_incident_id` | string/null | Direct source incident for synthetic descendants. |

Phase 03 may add operational fields, but it may not weaken the family/split or label constraints without a schema/taxonomy version change.

## Canonical label rule

`root_cause_code` must be one of the canonical IDs in `configs/label-taxonomy.yaml`. Unknown codes are rejected; silent coercion is forbidden.

`root_cause_category` must equal the category assigned to the canonical label in the same taxonomy version.

## Family/split invariant

The split unit is `incident_family_id`. A family may belong to exactly one of train, validation, or test. Synthetic descendants inherit the family and may only be created from training families.

## Synthetic-lineage invariant

For a synthetic row:

- `is_synthetic = true`;
- `parent_incident_id` is non-null;
- the parent belongs to a training family;
- the synthetic row cannot change the canonical family identity to bypass split rules.

## Initial prediction smoke contract

Phase 01 model smoke output must be strict JSON containing at least:

```json
{
  "root_cause_code": "n_plus_one_query",
  "reasoning": "brief explanation"
}
```

The full production prediction/evaluation schema is implemented in Phase 05.
