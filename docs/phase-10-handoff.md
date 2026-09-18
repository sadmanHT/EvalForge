# Phase 10 handoff

## Current state

Connected Phase 10 validation was completed on source commit
`2547e3cfb54875468351d1e3810c88df8a6a6c69` with the frozen Phase 09 adapter and preserved as
`evidence/phase-10/validation-run.json`. The primary fine-tuned protocol was then frozen and bound
to that validation evidence and the Phase 09 training identity.

The frozen scientific source head
`c59910e00f5e4fd0a722d2796da416c977753ddd` passed GitHub CI run #532
(`35256501761`) with both cumulative `verify-all` and clean-Compose success before the locked
test. The authorized Phase 10 fine-tuned locked test has now been **consumed exactly once**. Its
raw package is preserved under `evidence/phase-10/locked-test/`.

Do not run the Phase 10 fine-tuned locked test again. Its result may not be used to retrain,
switch checkpoints, change prompts or generation settings, alter confidence scoring, or select a
data-efficiency condition.

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
- exact accuracy: 5/6 (`0.8333333333333334`)
- parse failure rate: 1/6 (`0.16666666666666666`)
- locked-test usage at validation time: none

The validation result did not select a replacement checkpoint or adapter. The candidate remained
the Phase 09 validation-selected checkpoint.

## One-time locked-test evidence

The connected test run is `phase10-finetuned-test-v1`.

- scientific source commit:
  `c59910e00f5e4fd0a722d2796da416c977753ddd`
- evidence package ZIP SHA-256:
  `f3a5481841d84846946f602aea90749d797408b6632ce4b9895d014eeaa8380b`
- raw test evidence SHA-256:
  `9b253eb2e5a473812f46eccb5f90c0d8539aefb208a49266db521dc6bd137852`
- evaluator result hash:
  `cc221f1223d65fd416fdce12bd0c6a2041dd97b1fb92b105eaf56f33d4abe71c`
- exact accuracy: 5/6 (`0.8333333333333334`)
- parse failure rate: 1/6 (`0.16666666666666666`)
- inference failure count: 0
- locked-test consumed: yes
- selection or retuning after test: false

Five predictions parsed successfully and matched ground truth. The only primary-metric miss was
`incident-420f3a36538bb8156dd897d8` (database connection leak): the model's output was truncated at
128 generated tokens, leaving an unterminated JSON string, so the canonical parser returned
`INVALID_JSON` and no root-cause code. This is preserved as a test observation only.

The repository's Phase 06 zero-shot locked-test evidence uses the same six incident IDs and records
6/6 exact accuracy. Build the paired report directly from those already preserved predictions and
the Phase 10 raw test predictions; do not rerun either model arm for the comparison.

## Paired zero-shot vs fine-tuned comparison

The paired report is preserved as `evidence/phase-10/baseline-finetuned-comparison.json`.
It is rebuilt directly from the sealed Phase 06 zero-shot and Phase 10 fine-tuned predictions.
On the six locked-test incidents, zero-shot is 6/6 and fine-tuned is 5/6, for a fine-tuned-minus-
zero-shot difference of -1/6. The 95% paired bootstrap interval is [-0.5, 0.0] and the exact
two-sided McNemar p-value is 1.0. With only six cases, this is descriptive benchmark evidence,
not a basis for retuning or a broad claim of superiority.

## Remaining Phase 10 work

1. Execute the predeclared data-efficiency matrix (10%, 25%, 50%, 100%; seeds 20260908,
   20260909, 20260910) as secondary analysis only. Because the primary adapter and locked-test
   outcome are already frozen, efficiency runs must not alter the primary candidate or scientific
   protocol.
2. Publish the exact adapter and a truthful model card to Hugging Face, capture the immutable Hub
   revision, then download that revision into an empty environment and run load/inference smoke.
3. Add the machine-checkable Phase 10 hard-exit gate, wire it into cumulative verification and
   clean-Compose CI, and require an exact-head green run before marking Phase 10 complete.

No later Phase 10 work may reinterpret the locked test as a tuning set.


## Data-efficiency connected-run note — 2026-09-18

The first connected data-efficiency attempt on source `b733caf957befa2a5e54d516f88b15430db82530`
did not complete any matrix condition. The 10% / seed 20260908 condition reached its first
training log and the runtime aborted on `grad_norm=nan` before condition evidence, W&B artifact
publication, or validation evaluation was produced. No locked-test inference occurred.

The source-only follow-up keeps the predeclared model, data subsets, learning rate, optimizer,
epochs, gradient accumulation, quantization, prompt, evaluator, and seed matrix unchanged. It
corrects the runtime numeric guard so non-finite `loss` or `eval_loss` remains fatal, while a
transient non-finite fp16 gradient norm is recorded and left to the standard GradScaler overflow
recovery/step-skip mechanism. Every completed efficiency condition must record the recovery policy,
the exact affected global steps, and a count in its evidence. This is an infrastructure/runtime
semantics correction, not result-driven hyperparameter tuning.
