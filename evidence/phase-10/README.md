# Phase 10 evidence

This directory contains the preserved Phase 10 scientific evidence as the phase progresses.

Current evidence:

- `validation-run.json` — connected single-GPU validation run for the frozen Phase 09 adapter,
  exported before any locked-test authorization;
- `protocol-freeze.json` — immutable linkage from that validation evidence and the Phase 09
  training identity to the frozen Phase 10 protocol.

The locked test has not been executed yet. Test evidence must only be produced after the freeze
commit passes cumulative and clean-environment CI.

Evidence still required for Phase 10 completion includes:

- one authorized fine-tuned locked-test run with exact benchmark IDs;
- data-efficiency matrix results and subset-lineage manifests;
- paired zero-shot baseline vs fine-tuned report;
- Hugging Face repository/revision and truthful model card;
- clean download/load/inference smoke evidence.

No placeholder result, fabricated Hugging Face revision, CPU-only smoke, or locked-test-driven
retuning may satisfy the Phase 10 hard exit.
