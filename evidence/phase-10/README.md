# Phase 10 evidence

This directory contains the preserved Phase 10 scientific evidence as the phase progresses.

Current evidence:

- `validation-run.json` — connected single-GPU validation run for the frozen Phase 09 adapter,
  exported before any locked-test authorization;
- `protocol-freeze.json` — immutable linkage from that validation evidence and the Phase 09
  training identity to the frozen Phase 10 protocol.

The frozen evidence-bearing head `c59910e00f5e4fd0a722d2796da416c977753ddd` passed GitHub CI
run #532 (`35256501761`) with both cumulative `verify-all` and clean-Compose success. The Phase 10
fine-tuned locked test is therefore authorized under the frozen protocol, but it has not been
executed yet.

The one-time connected test must use the same adapter and scientific configuration, exactly one
visible CUDA GPU, and run ID `phase10-finetuned-test-v1`. The test evidence file is the one-time
consumption marker and must not be overwritten or regenerated after a completed run.

Evidence still required for Phase 10 completion includes:

- one authorized fine-tuned locked-test run with exact benchmark IDs;
- data-efficiency matrix results and subset-lineage manifests;
- paired zero-shot baseline vs fine-tuned report;
- Hugging Face repository/revision and truthful model card;
- clean download/load/inference smoke evidence.

No placeholder result, fabricated Hugging Face revision, CPU-only smoke, or locked-test-driven
retuning may satisfy the Phase 10 hard exit.
