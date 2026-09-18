# Phase 10 evidence

This directory contains the preserved Phase 10 scientific evidence as the phase progresses.

Current evidence:

- `validation-run.json` — connected single-GPU validation run for the frozen Phase 09 adapter,
  exported before any locked-test authorization;
- `protocol-freeze.json` — immutable linkage from that validation evidence and the Phase 09
  training identity to the frozen Phase 10 protocol;
- `locked-test/phase10-finetuned-test.json` — raw one-time connected fine-tuned locked-test
  evidence;
- `locked-test/LOCKED_TEST_STARTED.json` — start marker written before the one-time test
  inference process;
- `locked-test/completion-summary.json`, `README.txt`, and `SHA256SUMS` — the exact supporting
  files from the Kaggle evidence package.

The frozen scientific source head
`c59910e00f5e4fd0a722d2796da416c977753ddd` passed GitHub CI run #532
(`35256501761`) with both cumulative `verify-all` and clean-Compose success before test
consumption. The authorized fine-tuned locked test then ran once with run ID
`phase10-finetuned-test-v1`, one visible Tesla T4 GPU, the frozen adapter
`e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065`, and scientific config
`6064aaed457812a8102411eea47f02f78d812372097afe068ec53a540f9e1f7d`.

Locked-test package integrity:

- uploaded ZIP SHA-256:
  `f3a5481841d84846946f602aea90749d797408b6632ce4b9895d014eeaa8380b`;
- raw test evidence SHA-256:
  `9b253eb2e5a473812f46eccb5f90c0d8539aefb208a49266db521dc6bd137852`;
- evaluator result hash:
  `cc221f1223d65fd416fdce12bd0c6a2041dd97b1fb92b105eaf56f33d4abe71c`.

The locked test produced 6 predictions with zero inference-process failures. Exact root-cause
accuracy was 5/6 (`0.8333333333333334`). One case,
`incident-420f3a36538bb8156dd897d8`, reached the 128-token output limit with unterminated JSON and
therefore had `parse_status=INVALID_JSON`; this accounts for the 1/6 parse-failure rate and the
single primary-metric miss. The test result is a frozen observation and must not trigger
retraining, checkpoint switching, prompt/generation changes, confidence changes, or efficiency
condition selection.

The paired zero-shot baseline vs fine-tuned report is now preserved as
`baseline-finetuned-comparison.json` and is reproducibly derived from the sealed predictions.

Evidence still required for Phase 10 completion includes:

- the predeclared data-efficiency matrix and subset-lineage evidence, used only as secondary
  descriptive analysis and never to replace the already frozen primary adapter;
- Hugging Face repository/revision plus truthful model card;
- clean download/load/inference smoke evidence for that immutable Hub revision;
- a Phase 10 hard-exit gate followed by cumulative `make verify-all` and clean-Compose success.

The Phase 10 locked test is consumed. It must not be rerun based on its score, parse failure, or
any later comparison.
