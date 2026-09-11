# Processed incident-diagnosis datasets

Processed dataset versions are immutable once used by an experiment. Every version must contain a manifest whose identity includes the canonical schema version, label-taxonomy version, source snapshot/version, split seed, generation timestamp, content checksum, and split counts.

A `*-candidate` directory is not automatically a research-ready benchmark. `research_ready=false` means the source or split evidence has not satisfied Phase 03's hard exit. Validation/test records for the primary study must come from independently eligible incident families, and all family assignments must be frozen before synthetic augmentation.

When an eligible corpus is available, rebuild it deterministically from versioned raw snapshots/config, audit it in research mode, preserve the audit/class summaries/checksums, and only then lock the dataset version for later phases.
