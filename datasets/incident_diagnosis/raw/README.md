# Raw incident-diagnosis source snapshots

This directory is an immutable provenance boundary. Phase 03 may copy or normalize an external source into a versioned subdirectory, but it must never edit the external source repository or silently replace bytes under an existing source version.

Each versioned snapshot must record:

- source repository/URI and exact commit/version when available;
- source file paths and Git blob IDs when available;
- source DOI/license when applicable;
- whether the source is generated/synthetic or independently observed;
- whether it is eligible as independent research holdout evidence;
- SHA-256 of every preserved snapshot/metadata file.

A source snapshot may be useful for engineering or robustness work while still being ineligible for the primary research benchmark. Eligibility is a scientific property, not inferred from an upstream split name or from the fact that a dataset contains real incidents.

Current preserved sources:

- `opssentinel/<commit>/` — normalized semantic extraction of a deterministic generated BenchmarkLab catalog; read-only external provenance, not independent held-out evidence.
- `servicenow_uci/uci-498/` — content-identity-pinned UCI dataset 498 snapshot; real anonymized ServiceNow operational data used as auxiliary/robustness evidence, not mapped to EvalForge RCA labels because human-readable root-cause semantics are absent.
