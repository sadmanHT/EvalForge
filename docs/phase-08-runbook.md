# Phase 08 — RAG pipeline and controlled ablations

Phase 08 compares a retrieval-augmented pipeline against the frozen Phase 06 zero-shot baseline
without changing the base model, dataset, taxonomy, generation settings, or deterministic primary
evaluator. Retrieval configuration is selected from **validation only** and the locked test is run
once after that selection is sealed.

## Scientific rules

- Keep `configs/phase8-rag.json` in `state=validation` while running every ablation.
- Run exactly the variants registered in `configs/phase8-rag-ablations.json`; do not add a variant
  in response to validation results without treating that as a new protocol revision.
- The deterministic selection policy is fixed before results: exact accuracy, hierarchical
  accuracy, retrieval recall, lower ECE, lower p95 latency, then lexical variant ID.
- Never use locked-test outcomes to select a RAG variant, tune retrieval, edit prompts, or change
  context budgets.
- The Phase 08 locked test may run only after the validation ablation report and protocol-freeze
  record are committed. A disappointing test score is not a reason to rerun it.
- RAGAS/faithfulness is supporting evidence only. If an LLM judge is used, preserve its exact
  evaluator revision, judge model/revision, and prompt version. Supporting judge scores never
  participate in primary RAG selection.

### Validation amendment: undefined ECE on parse failures

The first external validation attempt on commit
`9ff1f93ac645102ea0a9ada7fe2ed1cefd17421a` exposed an implementation edge case after the
`rag-top-k3` run had completed: the common Phase 05 evaluator correctly omitted
`calibration.ece` because one non-OK parse prediction had no generated-label confidence, while
the Phase 08 evidence/selection layer incorrectly required ECE unconditionally. The attempt is
classified as a technical/incomplete validation suite and is not eligible for variant selection.

Selection policy `phase8-validation-selection-v2-missing-ece-worst` completes the predeclared
ordering without changing any model, prompt, dataset, retrieval, generation, or evaluator setting.
Exact accuracy, hierarchical accuracy, and retrieval recall remain the first three ranking keys.
At the existing ECE tie-break position only, an evaluator-undefined ECE caused by a parse failure
is ranked after every defined ECE; lower p95 latency and lexical variant ID remain the final
tie-breakers. The missing-ECE rule can never improve a candidate's rank at the ECE position.

Evidence may omit `calibration.ece` only when `quality.parse_failure_rate` is positive and the
sealed predictions contain a non-OK parse outcome with no confidence value. All other required
metrics remain mandatory. After this repair, run a complete clean replacement validation suite
from one new commit; do not mix the partial first-attempt runs into the accepted ablation report.

## Required runtime

Use the same Phase 06 GPU-host contract and software fingerprint for all real Phase 08 validation
runs and the locked test. A validation session should capture one sealed GPU-host evidence file and
reuse that exact environment for all seven variants.

For Kaggle free GPU quota, the only accepted zero-direct-cost snapshot is
`kaggle-free-quota-no-direct-usd-per-gpu-hour` with `gpu_hour_usd=0.0`. Do not substitute a made-up
positive hourly price.

W&B tracking is required for accepted real Phase 08 validation and locked-test evidence. Set
`WANDB_PROJECT=evalforge-phase8`, authenticate before inference, and preserve the run and artifact
references written into each evidence file.

## Validation execution

Start from a clean checkout of the exact commit being evaluated. Migrate PostgreSQL, import the
Phase 03 dataset through the existing importer, capture/validate GPU-host evidence, and build all
registered Phase 08 knowledge-base variants before model inference.

```bash
python scripts/index_phase8_rag_variants.py \
  --output-dir /tmp/evalforge-phase8-runtime-evidence
```

Run each registered variant exactly once on the validation split. Use distinct experiment/run IDs
and write evidence to `evidence/phase-08/validation/<variant-id>.json`.

```bash
python evals/rag_runner.py \
  --split validation \
  --variant-id <variant-id> \
  --git-commit <40-hex-commit> \
  --gpu-host-evidence evidence/phase-08/validation-gpu-host.json \
  --cost-rate-snapshot-version kaggle-free-quota-no-direct-usd-per-gpu-hour \
  --gpu-hour-usd 0.0 \
  --experiment-id phase8-<variant-id>-validation \
  --run-id phase8-<variant-id>-validation \
  --evidence-output evidence/phase-08/validation/<variant-id>.json
```

Do not run the test split during this stage. If a validation run fails technically, classify and
repair the infrastructure defect before deciding whether a clean replacement run is scientifically
valid; never replace a run because its score is low.

## Select and freeze

After all registered validation evidence exists, build the deterministic ablation report. This
revalidates every evidence seal, exact split coverage, retrieval traces, leakage constraints, W&B
references, and the common GPU-host binding before selecting one variant.

```bash
python scripts/build_phase8_ablation_report.py \
  --gpu-host-evidence evidence/phase-08/validation-gpu-host.json
```

Build the validation baseline-vs-RAG comparison for the selected variant. The comparison requires
identical validation incident IDs and writes paired correctness, a paired bootstrap accuracy CI, and
an exact McNemar test.

```bash
python scripts/build_phase8_comparison.py \
  --split validation \
  --rag-evidence evidence/phase-08/validation/<selected-variant-id>.json \
  --output evidence/phase-08/validation-comparison.json
```

Verify freeze readiness without mutation first, then apply it once.

```bash
python scripts/freeze_phase8_protocol.py \
  --gpu-host-evidence evidence/phase-08/validation-gpu-host.json

python scripts/freeze_phase8_protocol.py \
  --gpu-host-evidence evidence/phase-08/validation-gpu-host.json \
  --apply
```

Commit the seven validation run files, validation GPU-host evidence, ablation report, validation
comparison, frozen `configs/phase8-rag.json`, and `evidence/phase-08/protocol-freeze.json` together.
Do not change scientific fields after this commit.

## Single locked test

Use a fresh runtime that reproduces the validation GPU/software fingerprint. Capture
`evidence/phase-08/test-gpu-host.json` and verify it matches the freeze record before invoking the
runner. The frozen protocol rejects every variant except `selected_variant_id`.

```bash
python evals/rag_runner.py \
  --split test \
  --variant-id <selected-variant-id> \
  --git-commit <frozen-commit> \
  --gpu-host-evidence evidence/phase-08/test-gpu-host.json \
  --cost-rate-snapshot-version kaggle-free-quota-no-direct-usd-per-gpu-hour \
  --gpu-hour-usd 0.0 \
  --experiment-id phase8-rag-test \
  --run-id phase8-rag-test \
  --evidence-output evidence/phase-08/test-run.json
```

Once real locked-test inference begins, treat that authorized attempt as consumed. Do not rerun for
score, calibration, retrieval tuning, or presentation quality. If the attempt fails, preserve the
logs and classify the failure before any rerun decision.

Build the paired held-out comparison only after the locked test succeeds:

```bash
python scripts/build_phase8_comparison.py \
  --split test \
  --rag-evidence evidence/phase-08/test-run.json \
  --output evidence/phase-08/baseline-vs-rag-test-comparison.json
```

## Exit gate

`python scripts/check_phase8_exit.py` passes only when the repository contains the complete sealed
validation bundle, deterministic selection report, frozen protocol record, one valid locked-test
run on the selected variant, matching GPU-host evidence, and the paired Phase 06 baseline-vs-RAG
test comparison.

Until the real external validation/freeze/test evidence is committed, Phase 08 is intentionally
incomplete and `phase8-exit` must not be added to the cumulative `verify-all` dependency list.
After the gate passes, add it to cumulative CI, run the full suite, and run the clean Compose/reindex
smoke before declaring Phase 08 complete.

## Completed locked-test record

The single authorized locked-test attempt was consumed on 2026-09-14 from frozen commit
`af48887816634f444a57010a8e76881caefae7cc` using `rag-top-k1`. The canonical runner completed
successfully, persisted all six held-out predictions and retrieval traces, recomputed metrics,
sealed raw `evidence/phase-08/test-run.json`, and uploaded the W&B result artifact.

The notebook then encountered a **post-run diagnostic bug**: its local sanity-check cell accessed
`trace["provenance"]` even though the canonical retrieval-trace schema stores provenance at
`trace["metadata"]["provenance"]`. Because this happened only after the canonical runner printed
`PHASE08_LOCKED_TEST_RUN=PASS` and emitted sealed evidence, it is classified as notebook
post-processing failure rather than model/inference invalidation. The locked test was therefore not
rerun. The deterministic paired comparison was generated later from the sealed evidence. The raw
`test-run.json` emitted by the runner is preserved in-repo as deterministic `test-run.json.gz`;
repository validators transparently decompress it and revalidate the original evidence seal.

Final locked-test facts:

- selected variant: `rag-top-k1`
- run ID: `phase8-rag-top-k1-test-kaggle-v1`
- test evidence SHA-256: `bb8e12f9f63ee939e630535bd38f815779f09f23779202be19cf7c0164e0d956`
- environment fingerprint: `45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff`
- RAG exact accuracy: `5/6`
- frozen Phase 06 baseline exact accuracy: `6/6`
- paired delta: `-1/6`
- exact McNemar p-value: `1.0`
- paired bootstrap 95% interval: `[-0.5, 0.0]`
- comparison SHA-256: `f763e9ffa94b7fecbe7e445e55d4249f6d81cc1fd729a8f7e0a96097a24ee13f`

These held-out outcomes are final Phase 08 evidence, not tuning feedback.
