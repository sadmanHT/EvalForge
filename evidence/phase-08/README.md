# Phase 08 evidence

Phase 08 validation is complete and the protocol is frozen for the single authorized locked-test run.

The accepted validation suite is the clean replacement v2 executed from repository commit
`46c187476076c4b5874267fb1fee9d07133371e9`. All seven pre-registered RAG variants are sealed
under `validation/`, with W&B tracking, one matching Phase 06 GPU/software fingerprint, and the
registered ablation-suite hash
`c08e351c61215fb96fa5965dd9a51da37042a7fbec7db78a74fea7a819a68d62`.

`ablation-report.json` deterministically selected `rag-top-k1` from validation only. The selection
policy is `phase8-validation-selection-v2-missing-ece-worst`: exact accuracy, hierarchical
accuracy, retrieval recall, ECE, p95 latency, then lexical variant ID. The policy amendment only
defines how an evaluator-undefined ECE caused by a parse failure is ordered at the already-declared
ECE tie-break position; it does not override higher-priority metrics.

`validation-comparison.json` pairs the selected validation RAG run with the frozen Phase 06
validation baseline on the same six incident IDs. The selected RAG variant scored 5/6 exact
accuracy versus 6/6 for the Phase 06 validation baseline. This tiny validation split is a protocol
selection result, not evidence of broad generalization or statistical superiority.

`protocol-freeze.json` binds the selected variant, all seven validation evidence files, the
ablation report, and the accepted GPU/software fingerprint. `configs/phase8-rag.json` is now
`state=frozen`, `selected_variant_id=rag-top-k1`, and `locked_test_authorized=true`.

The first external validation attempt from commit
`9ff1f93ac645102ea0a9ada7fe2ed1cefd17421a` remains intentionally excluded from accepted
scientific evidence. It stopped after `rag-top-k3` when the Phase 08 evidence layer treated an
evaluator-undefined ECE on a parse failure as a hard schema error. The complete v2 suite reran all
seven variants from a fresh database/runtime with distinct `kaggle-v2` run identifiers. No partial
v1 runs or W&B artifacts are mixed into the accepted suite.

Do not hand-author, backfill, or fabricate run-evidence JSON in this directory. The next scientific
action is exactly one locked-test run of `rag-top-k1` under the frozen protocol. Do not tune,
reselect, or rerun validation based on the locked-test outcome.
