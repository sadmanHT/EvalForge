# Phase 03 Evidence

Status: FINAL VALIDATION IN PROGRESS.

This directory mirrors the reproducible Phase 03 provenance, dataset, audit, and handoff evidence for the locked research benchmark `evalforge-incident-diagnosis-v0.1.0`.

The evidence set includes the candidate provenance manifest, source-candidate summary, source audit, ServiceNow/UCI auxiliary summary, public-incident candidate coverage review, locked research manifest, class/split summary, leakage audit, explicit research-admission summary, data card, Phase 03 handoff, and deterministic SHA-256 checksums for those mirrored artifacts.

Current locked research state:

- 24 conservatively reviewed public-incident candidates;
- 18 supported independent mappings;
- 18 checksum-preserved public primary sources;
- exactly 3 independent families per frozen taxonomy label;
- 18 explicitly research-admitted records/families;
- deterministic split seed `20260908`;
- train / validation / test = 6 / 6 / 6;
- one independent family per frozen label in every split;
- zero synthetic rows in the locked base research dataset;
- zero research leakage-audit errors;
- candidate and research artifact reproducibility checks passing;
- Phase 03 hard-exit check passing on the generated locked corpus.

The remaining closure requirement is operational validation of the exact cleaned branch and then the exact target merge commit through the ordinary cumulative CI and fresh clean-Compose gate.

OpsSentinel remains a generated auxiliary/engineering source and was not modified. The UCI/ServiceNow corpus remains auxiliary real-world operational evidence and is not relabeled into the frozen RCA taxonomy. Phase 04 has not started.
