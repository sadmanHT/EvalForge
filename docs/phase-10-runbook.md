# Phase 10 runbook — fine-tuned evaluation, data efficiency, and release

Phase 10 starts from the Phase 09 evidence-bearing green head. The scientific identity remains
the frozen Mistral-7B-Instruct-v0.3 revision, the Phase 6 prompt/output/generation contract,
the Phase 5 evaluator, and the Phase 3 benchmark manifest.

## Protocol boundary

`configs/phase10-finetuned.json` is initially `state=validation` and
`locked_test_authorized=false`. The Phase 10 runner must refuse the test split until a
validation-selected primary-adapter record is preserved and the protocol is frozen. Phase 08
test outcomes are not inputs to Phase 10 selection.

The Phase 09 full-data adapter is the predeclared candidate because its checkpoint was selected
during Phase 09 using validation `eval_loss`. Phase 10 validation evaluates that candidate under
the common classification harness with **no retrieval**. The data-efficiency study is descriptive
and is explicitly prohibited from selecting the primary adapter.

Frozen identity inherited into Phase 10:

- base model: `mistralai/Mistral-7B-Instruct-v0.3`;
- base revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`;
- candidate adapter SHA-256:
  `e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065`;
- source training run: `phase9-qlora-validation-v1`;
- prompt: `zero-shot-baseline-v1`;
- evaluator: `phase5-evaluator-v1`;
- dataset: `evalforge-incident-diagnosis-v0.1.0`.

## Repository preflight

On the exact source commit intended for the connected run:

```bash
make phase10-contract
make test-phase10
make verify-all
```

The contract must report `PHASE10_STATE=validation` and
`LOCKED_TEST_AUTHORIZED=false` before connected validation. A clean Compose CI run is also
required on that same source commit.

## Connected portable validation

The connected GPU path is `evals/finetuned_portable_runner.py`. It exists for environments such
as Kaggle where the real PEFT model can run but PostgreSQL is not part of the runtime. This path
still reuses the canonical Phase 10 pipeline-contract validator, retry semantics, and common
`EvaluationHarness`; it does not define a second evaluator.

Use exactly one visible CUDA GPU. On Kaggle, set `CUDA_VISIBLE_DEVICES=0` before importing torch.
Install the committed backend lock and `training/requirements.gpu.txt` in an isolated Python
path/environment.

Download the Phase 09 W&B adapter artifact using the entity/project from the preserved W&B run
reference and the artifact name in `evidence/phase-09/training-run.json`. The artifact contains an
`adapter/` directory. The connected runner refuses to start unless that local directory hashes to
the frozen candidate SHA-256.

Run **validation only** while the protocol is in validation state:

```bash
python evals/finetuned_portable_runner.py \
  --split validation \
  --adapter-locator /absolute/path/to/adapter \
  --adapter-revision e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065 \
  --git-commit "$(git rev-parse HEAD)" \
  --run-id phase10-finetuned-validation-v1 \
  --hardware-runtime-descriptor '<GPU / CUDA / runtime description>' \
  --cost-rate-snapshot-version '<rate snapshot ID>' \
  --gpu-hour-usd '<hourly GPU rate>' \
  --evidence-output /tmp/phase10-finetuned-validation.json
```

The runner records every canonical prediction, evaluator payload/result hash, adapter checksum,
source-training identity, source commit, GPU/package evidence, inference failures, latency/cost
metrics, and the exact protocol hash. It explicitly records
`execution_mode=portable_connected_gpu_no_database` and `persistence_stack_used=false`; repository
integration tests separately prove database persistence/reload behavior, and preserved raw
predictions can be replayed later without re-running the model.

`--split test` is protocol-gated and must fail before the freeze commit. Do not bypass that guard.

## Data-efficiency design

Fractions are 10%, 25%, 50%, and 100%, with seeds 20260908, 20260909, and 20260910.
Sampling is by complete training family using a deterministic seeded order. Within each seed the
subsets are nested and counts use `ceil(fraction * training_family_count)`, minimum one family.

The frozen research dataset has very few independent training families. Sub-100% conditions
therefore cannot guarantee all-label coverage. This limitation must be reported; the code must not
invent stratification that the data cannot support. Validation and test families are never sampled
into training. Data-efficiency runs are secondary analysis and must not replace or select the
primary Phase 09 adapter.

## Freeze before the locked test

After connected validation evidence has been inspected and preserved, freeze the primary
fine-tuned protocol in a separate commit. That commit must link the validation evidence and
candidate adapter identity, change the protocol state to `frozen`, and set
`locked_test_authorized=true` only after all validation-controlled choices are final. The freeze
commit itself must pass cumulative repository and clean-environment CI before any locked-test
execution.

The locked test is evaluation-only. It must not select the adapter, checkpoint, hyperparameters,
prompt, confidence method, generation settings, or data-efficiency condition.

## External evidence sequence

1. Run repository contract and focused Phase 10 tests.
2. On CUDA, hydrate the exact Phase 09 adapter and verify its tree SHA-256.
3. Run connected fine-tuned validation with the portable runner and preserve its raw evidence.
4. Execute the predeclared data-efficiency matrix; log each run, seed, subset lineage,
   validation metrics, wall-clock/GPU time, and cost separately.
5. Preserve the primary-adapter freeze record from validation evidence.
6. Commit the frozen protocol and pass cumulative plus clean-environment CI.
7. Only after that freeze gate passes, run the fine-tuned locked test once.
8. Build the paired baseline-vs-fine-tuned report on identical benchmark IDs.
9. Push the exact adapter plus truthful model card to Hugging Face, capture the immutable Hub
   commit, download that revision into an empty environment, verify content, and run inference.
10. Preserve Phase 10 evidence, run the hard exit, `make verify-all`, and clean Compose CI.

Do not use locked-test performance to retrain, switch checkpoints, change prompt/generation
settings, alter confidence scoring, or choose an efficiency seed.
