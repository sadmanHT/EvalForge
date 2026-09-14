# Phase 08 evidence

This directory is intentionally empty of scientific run results until real Phase 08 validation is executed.

Accepted validation evidence is produced for every registered RAG ablation variant under `validation/`, using one sealed validation GPU-host record and configured W&B tracking. The canonical suite entry point is `evals/rag_ablation_runner.py`, which shares one frozen base-model load across the registered validation variants. `ablation-report.json` selects the primary RAG variant from validation only, and `validation-comparison.json` compares that selected variant with the frozen Phase 06 validation baseline on identical incident IDs.

After those artifacts validate, `protocol-freeze.json` binds the selected variant, validation evidence, ablation report, and GPU/software fingerprint. Only then may the single authorized locked-test run produce `test-run.json` and `test-gpu-host.json`. `baseline-vs-rag-test-comparison.json` is generated from that held-out RAG evidence and the already frozen Phase 06 test evidence.

Do not hand-author, backfill, or fabricate any run-evidence JSON in this directory. The Phase 08 exit gate treats incomplete or inconsistent evidence as a failure.
