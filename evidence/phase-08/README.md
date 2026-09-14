# Phase 08 evidence

Phase 08 is complete. Validation selected `rag-top-k1` under the frozen protocol, the single
authorized locked-test run was executed once, and the sealed repository exit gate passes.

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
ablation report, and the accepted GPU/software fingerprint. `configs/phase8-rag.json` is
`state=frozen`, `selected_variant_id=rag-top-k1`, and `locked_test_authorized=true`.

The single authorized locked-test execution ran from frozen repository commit
`af48887816634f444a57010a8e76881caefae7cc` as
`phase8-rag-top-k1-test-kaggle-v1`. `test-run.json.gz` is the deterministic gzip archive of the raw `test-run.json` produced by the canonical runner and contains all six held-out predictions, six retrieval traces, six cost records, configured W&B references, and metric-recomputation proof. Its evidence seal is
`bb8e12f9f63ee939e630535bd38f815779f09f23779202be19cf7c0164e0d956`. The archived raw JSON byte SHA-256 before gzip is
`c400991f84e8eadd13f7c95b2d04a2f56effd6ea12a141f321c117fd6caff028`; the deterministic gzip file SHA-256 is
`27375fca2c32abba32717547ff92341caf91e09fb81ff238e07c7b1c5970686c`. The locked-test GPU
software fingerprint exactly matches the validation freeze:
`45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff`.

The RAG locked-test exact accuracy is 5/6 (`0.833333...`) versus 6/6 (`1.0`) for the frozen Phase 06
zero-shot baseline. The paired accuracy delta is `-1/6`; the exact McNemar p-value is `1.0`, and the
95% paired bootstrap interval for the accuracy delta is `[-0.5, 0.0]`. This tiny held-out set does
not establish statistical superiority for either pipeline. The RAG run had no parse failures and no
inference retries. Retrieval precision/recall were both `1/6`, exposing a clear retrieval-quality
limitation that must not be tuned against this locked test.

The Kaggle notebook itself failed only after the canonical locked-test runner had completed, sealed,
recomputed, and uploaded the accepted run: a notebook-only diagnostic cell incorrectly looked for
`trace["provenance"]` instead of `trace["metadata"]["provenance"]`. That post-processing error did
not invalidate the already completed scientific run and was not used to justify a rerun. The paired
comparison was subsequently built deterministically from the sealed Phase 06 and Phase 08 evidence.

The first external validation attempt from commit
`9ff1f93ac645102ea0a9ada7fe2ed1cefd17421a` remains intentionally excluded from accepted
scientific evidence. It stopped after `rag-top-k3` when the Phase 08 evidence layer treated an
evaluator-undefined ECE on a parse failure as a hard schema error. The complete v2 suite reran all
seven variants from a fresh database/runtime with distinct `kaggle-v2` run identifiers. No partial
v1 runs or W&B artifacts are mixed into the accepted suite.

Do not rerun the Phase 08 locked test or tune retrieval, prompts, context budgets, or variant
selection from these held-out outcomes. Future retrieval improvements belong to a new protocol and
validation cycle.
