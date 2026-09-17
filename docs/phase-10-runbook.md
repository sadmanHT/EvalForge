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

## Data-efficiency design

Fractions are 10%, 25%, 50%, and 100%, with seeds 20260908, 20260909, and 20260910.
Sampling is by complete training family using a deterministic seeded order. Within each seed the
subsets are nested and counts use `ceil(fraction * training_family_count)`, minimum one family.

The frozen research dataset has very few independent training families. Sub-100% conditions
therefore cannot guarantee all-label coverage. This limitation must be reported; the code must not
invent stratification that the data cannot support. Validation and test families are never sampled
into training.

## External evidence sequence

1. Run repository contract and focused Phase 10 tests.
2. On CUDA, hydrate the exact Phase 09 adapter and verify its tree SHA-256.
3. Run fine-tuned validation through the common evaluator and persist predictions.
4. Execute the predeclared data-efficiency matrix; log each run, seed, subset lineage,
   validation metrics, wall-clock/GPU time, and cost separately.
5. Preserve the primary-adapter freeze record from validation evidence.
6. Only after the freeze gate passes, run the fine-tuned locked test once.
7. Build the paired baseline-vs-fine-tuned report on identical benchmark IDs.
8. Push the exact adapter plus truthful model card to Hugging Face, capture the immutable Hub
   commit, download that revision into an empty environment, verify content, and run inference.
9. Preserve Phase 10 evidence, run the hard exit, `make verify-all`, and clean Compose CI.

Do not use locked-test performance to retrain, switch checkpoints, change prompt/generation
settings, alter confidence scoring, or choose an efficiency seed.
