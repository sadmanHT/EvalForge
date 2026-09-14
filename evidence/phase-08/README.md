# Phase 08 evidence

This directory is intentionally empty of scientific run results until real Phase 08 validation is executed.

Accepted validation evidence is produced for every registered RAG ablation variant under `validation/`, using one sealed validation GPU-host record and configured W&B tracking. The canonical suite entry point is `evals/rag_ablation_runner.py`, which shares one frozen base-model load across the registered validation variants. `ablation-report.json` selects the primary RAG variant from validation only, and `validation-comparison.json` compares that selected variant with the frozen Phase 06 validation baseline on identical incident IDs.

Phase 08 index preparation reuses the Phase 07 persistence contract. CI verifies that persisted pgvector embeddings survive their database text round trip and that reindexing the same sealed plan remains logically idempotent before any external validation evidence is accepted.

After those artifacts validate, `protocol-freeze.json` binds the selected variant, validation evidence, ablation report, and GPU/software fingerprint. Only then may the single authorized locked-test run produce `test-run.json` and `test-gpu-host.json`. `baseline-vs-rag-test-comparison.json` is generated from that held-out RAG evidence and the already frozen Phase 06 test evidence.

Do not hand-author, backfill, or fabricate any run-evidence JSON in this directory. The Phase 08 exit gate treats incomplete or inconsistent evidence as a failure.

The first external validation attempt from commit
`9ff1f93ac645102ea0a9ada7fe2ed1cefd17421a` is intentionally not committed here as accepted
scientific evidence. It stopped after `rag-top-k3` when the Phase 08 evidence layer treated an
evaluator-undefined ECE on a parse failure as a hard schema error. The preserved external bundle
is diagnostic evidence for the infrastructure repair only. Accepted Phase 08 validation evidence
must come from the complete clean replacement suite under the amended missing-ECE selection rule.
