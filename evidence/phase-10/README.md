# Phase 10 evidence

Phase 10 scientific evidence is now preserved for the frozen FINETUNED arm. The one-time locked
test remains consumed and must never be rerun or treated as a tuning set.

## Frozen primary identity

- base model: `mistralai/Mistral-7B-Instruct-v0.3`
- base revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`
- primary adapter SHA-256:
  `e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065`
- Phase 10 scientific config SHA-256:
  `6064aaed457812a8102411eea47f02f78d812372097afe068ec53a540f9e1f7d`

## One-time locked test

The authorized connected test run `phase10-finetuned-test-v1` was consumed exactly once. Raw
and supporting evidence remains under `locked-test/`.

- raw test evidence SHA-256:
  `9b253eb2e5a473812f46eccb5f90c0d8539aefb208a49266db521dc6bd137852`
- evaluator result hash:
  `cc221f1223d65fd416fdce12bd0c6a2041dd97b1fb92b105eaf56f33d4abe71c`
- exact accuracy: 5/6
- parse failures: 1/6
- selection or retuning after test: false

The paired report `baseline-finetuned-comparison.json` uses the already-preserved Phase 06
zero-shot predictions on the identical six IDs. Zero-shot is 6/6 and fine-tuned is 5/6; the
fine-tuned-minus-zero-shot exact-accuracy difference is -1/6, paired-bootstrap 95% interval
[-0.5, 0.0], exact two-sided McNemar p=1.0. These six-case results are descriptive only.

## Data-efficiency study

The predeclared 10%, 25%, 50%, and 100% train-family fractions were completed for seeds 20260908,
20260909, and 20260910: 12 connected single-GPU conditions total. The study is secondary
descriptive analysis and has `primary_adapter_selection_use=false` and `test_split_used=false`.
No locked-test ID was used by an efficiency condition.

The uploaded connected-run package is sealed by SHA-256
`c1616b7fffc58e78f107de91725e706c4af410632e105513add4d700fb149139`.
Its original aggregate member SHA-256 is
`7ae6c09672e3e12992908032379dd30b431c3d2aac90b39a9d2f2ca7c463b30e`.
For repository review, `data-efficiency/aggregate.json` preserves the same JSON payload in a
canonicalized serialization; `data-efficiency/SHA256SUMS` preserves the original package's raw
condition/log checksum roster; and `data-efficiency/package-validation.json` preserves the
independent package audit plus all 12 condition/W&B identities. The hard-exit gate accepts this
sealed compact representation while retaining support for fully exploded condition files.

Mean validation exact accuracy across the three seeds was 0.5000 at 10%, 0.6111 at 25%, 0.7222
at 50%, and 0.8333 at 100%. With three seeds and six validation incidents, this is reported as a
small-sample descriptive efficiency curve, not a general scaling claim.

## Hugging Face release

The exact frozen adapter was published publicly and then clean-downloaded at immutable revision:

- repository: `sadmanht/evalforge-mistral-7b-incident-diagnosis-qlora`
- revision: `0a11104c26ed6edc2fce612a341123b9ff4e9001`
- release evidence SHA-256:
  `03a209910f21363797530f279817186f6fb2f94b6ed1704661eeaf192911dca0`
- clean-smoke evidence SHA-256:
  `3602bcafe4de301c459e6de93605309a5fb4f531b68f989eaba0e42c366f357b`
- model-card SHA-256:
  `125387a30f765cffac6487e51fc54d8f7cbe68b72ba6be8974856cda52cef20c`

The clean smoke used the validation split only, one visible Tesla T4, the FINETUNED pipeline,
no retrieval, and reproduced the exact frozen adapter SHA. No post-test retuning occurred.

With these artifacts present, the repository-level Phase 10 evidence gate is expected to resolve
to `complete`. Phase 10 is not operationally closed until the exact evidence-bearing Git head also
passes cumulative `verify-all`, the Phase 10 hard-exit check, and clean-Compose CI.
