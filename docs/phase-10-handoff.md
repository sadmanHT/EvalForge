# Phase 10 handoff

## Current state

Connected Phase 10 validation has been completed on the exact green source commit
`2547e3cfb54875468351d1e3810c88df8a6a6c69` with the frozen Phase 09 adapter.
The preserved validation evidence is `evidence/phase-10/validation-run.json`; it records
six validation incidents, the canonical Phase 5 evaluator, one visible Tesla T4 GPU,
and no locked-test authorization at evaluation time.

That evidence was committed in `4861ea86fd31c046d2bd9b76177659a0739b79f6`, and GitHub CI run
#528 passed both the cumulative `verify-all` gate and the clean-Compose gate.

The primary fine-tuned protocol is now frozen. `configs/phase10-finetuned.json` is
`state=frozen` with `locked_test_authorized=true`. The freeze is bound to the exact
validation evidence, Phase 09 source-training evidence, candidate adapter checksum,
and unchanged scientific configuration through `evidence/phase-10/protocol-freeze.json`.

The frozen evidence linkage was repaired in
`c59910e00f5e4fd0a722d2796da416c977753ddd` to match the preserved validation file bytes.
GitHub CI run #532 (`35256501761`) passed both `verify-all` and `clean-compose` on that exact
head. The Phase 10 fine-tuned locked test is therefore authorized, but it has **not** been
executed yet.

## Frozen candidate

- Base: `mistralai/Mistral-7B-Instruct-v0.3`
- Base revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`
- Phase 09 source run: `phase9-qlora-validation-v1`
- Adapter artifact: `phase9-qlora-validation-v1-adapter`
- Adapter SHA-256: `e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065`
- Training config SHA-256: `b322d37e9f868b251fa14efb32c8412eab2ddaf3d5cd4adcf027c0f08d51411a`
- Dataset: `evalforge-incident-diagnosis-v0.1.0`
- Prompt: `zero-shot-baseline-v1`
- Evaluator: `phase5-evaluator-v1`
- Phase 10 scientific config SHA-256:
  `6064aaed457812a8102411eea47f02f78d812372097afe068ec53a540f9e1f7d`

## Validation evidence

The connected validation run is `phase10-finetuned-validation-v1`.

- evidence file SHA-256:
  `3015ddf10b41b82be97619611d69aabd4679de6719eb3e437c461a2b20bc5385`
- evaluator result hash:
  `0af782cb9ec0dbd5b70733185423be558509182c3650da3b3d60e8b40796acea`
- exact accuracy: `5/6` (`0.8333333333333334`)
- parse failure rate: `1/6` (`0.16666666666666666`)
- locked test usage: none

These validation outcomes do not select a replacement checkpoint or adapter. The candidate was
predeclared from the Phase 09 validation-selected checkpoint, and the data-efficiency study remains
secondary analysis only.

## Next scientific action

Run exactly one connected `--split test` execution with `evals/finetuned_portable_runner.py` on
the same frozen adapter and scientific configuration. The run must be pinned to a green frozen
source head at or after `c59910e00f5e4fd0a722d2796da416c977753ddd`, use exactly one visible
CUDA GPU, verify the adapter tree SHA-256 before inference, and preserve the raw test evidence.

Use run ID `phase10-finetuned-test-v1`. Treat creation of the test evidence file as the one-time
consumption marker: a rerun must refuse to execute if that evidence already exists.

The locked test is evaluation-only. Do not use its result to retrain, switch checkpoints, change
the prompt, change generation settings, alter confidence scoring, or select a data-efficiency
condition.

After the one-time locked test, preserve the raw test evidence before building the paired
zero-shot-baseline vs fine-tuned report. Hugging Face publication and clean-download smoke remain
later Phase 10 steps.
