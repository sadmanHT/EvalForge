# Phase 10 handoff

## Evidence state

The Phase 10 FINETUNED scientific protocol is frozen. The one-time locked test has been consumed
exactly once and must not be rerun. No later data-efficiency or release work changed the primary
adapter, prompt, generation settings, evaluator, confidence method, or test interpretation.

Frozen identity:

- base: `mistralai/Mistral-7B-Instruct-v0.3`
- base revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`
- primary adapter SHA-256:
  `e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065`
- Phase 09 training config SHA-256:
  `b322d37e9f868b251fa14efb32c8412eab2ddaf3d5cd4adcf027c0f08d51411a`
- Phase 10 scientific config SHA-256:
  `6064aaed457812a8102411eea47f02f78d812372097afe068ec53a540f9e1f7d`

## Primary evaluation

Validation: 5/6 exact accuracy with 1/6 parse failure. Locked test: 5/6 exact accuracy with 1/6
parse failure. The only locked-test miss was the preserved truncated/invalid JSON response for
`incident-420f3a36538bb8156dd897d8`; it is an observation, not a retuning signal.

The paired Phase 06 zero-shot comparison on the identical six test IDs is zero-shot 6/6 versus
fine-tuned 5/6. Fine-tuned minus zero-shot is -1/6; paired bootstrap 95% interval [-0.5, 0.0];
exact two-sided McNemar p=1.0. The sample is too small for a broad superiority claim.

## Data efficiency

All 12 predeclared fraction/seed conditions completed on the validation split only. Connected-run
package SHA-256:
`c1616b7fffc58e78f107de91725e706c4af410632e105513add4d700fb149139`.
The repository preserves the aggregate, original package checksum roster, and independent package
validation manifest with all 12 W&B run/artifact identities and raw condition SHA-256 values.

Mean validation exact accuracy by fraction: 10%=0.5000, 25%=0.6111, 50%=0.7222, 100%=0.8333.
This is secondary descriptive evidence only and did not select the frozen primary adapter.

## Model release

Public Hugging Face adapter:
`sadmanht/evalforge-mistral-7b-incident-diagnosis-qlora`

Immutable revision:
`0a11104c26ed6edc2fce612a341123b9ff4e9001`

The release evidence confirms the exact frozen adapter and truthful model card. A fresh-cache,
immutable-revision GPU smoke then loaded that public artifact and produced a valid FINETUNED
validation prediction with no retrieval. `locked_test_split_used=false`,
`post_test_retuning=false`, and `primary_adapter_selection_use=false` are preserved.

## Final closure rule

The evidence-level Phase 10 gate should now be `complete`. Do not begin COMBINED-arm work until
the exact evidence-bearing branch head passes cumulative `verify-all`, the Phase 10 hard-exit
check, and clean-Compose CI. The locked test remains permanently consumed.
