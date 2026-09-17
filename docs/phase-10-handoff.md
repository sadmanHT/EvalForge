# Phase 10 handoff

## Current state

Phase 10 is in pre-connected-validation state. The repository has a frozen scientific protocol,
Phase 09 adapter identity checks, fine-tuned persistence integration tests, worker orchestration,
a database-free connected-GPU evaluation path, and cumulative CI coverage. No genuine Phase 10
validation evidence has been committed yet, the protocol remains `state=validation`, and
`locked_test_authorized=false`.

The locked test has not been consumed by Phase 10 and must remain untouched until a separate
validation-evidence freeze commit authorizes it.

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

`configs/phase10-finetuned.json` cross-checks these fields against the committed Phase 09 training
evidence every time the protocol is loaded.

## Implemented repository path

Phase 10 currently provides:

- `app.inference.finetuned_protocol` for frozen identity and locked-test authorization;
- `app.inference.finetuned_model` for content-addressed adapter loading;
- `app.inference.finetuned_runner` for canonical persisted FINETUNED experiments;
- `app.inference.finetuned_orchestration` and worker task `phase10_finetuned`;
- `app.inference.finetuned_portable` for the same evaluator/retry path without PostgreSQL;
- `evals/finetuned_portable_runner.py` for connected single-GPU evidence generation;
- deterministic train-family data-efficiency subset construction;
- Phase 10 focused unit/integration tests and cumulative CI/clean-Compose contract gates.

The portable connected path is intentionally explicit about not using the persistence stack. It
preserves raw canonical predictions and the common evaluator payload so they can be validated or
replayed after the GPU run without re-running the model.

## Next scientific action

Run the exact current green source commit on a single visible CUDA GPU. Download the preserved
Phase 09 W&B adapter artifact, verify the local adapter tree SHA-256, and execute only the
`validation` split with `evals/finetuned_portable_runner.py`.

Preserve the resulting JSON outside the repository first. Verify its source commit, protocol hash,
adapter hash, prediction coverage, evaluator result hash, GPU environment, and split before
committing anything under `evidence/phase-10/`.

Do not change `state` or `locked_test_authorized` yet. Do not run `--split test` yet.

## After validation evidence

Once connected validation evidence is valid and committed, the next source commit should freeze
the primary protocol and explicitly authorize the locked test. That freeze must be green in both
cumulative and clean-environment CI before the one-time fine-tuned locked-test execution.

Data-efficiency experiments remain train/validation-only secondary analysis and cannot select a
replacement primary adapter. Hugging Face publication is owned by Phase 10 and happens only after
the locked-test evidence and paired comparison are preserved.
