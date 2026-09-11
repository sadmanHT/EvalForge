# EvalForge evaluation methodology — Phase 05

Version: `phase5-evaluator-v1`

Phase 05 defines one evaluator contract for all four primary study arms. It does not run the real zero-shot, RAG, fine-tuned, or combined benchmark; those later phases must emit the canonical `Prediction` contract and are scored by this same harness.

## Primary correctness

The primary metric is exact root-cause-code accuracy on aligned incident IDs. A prediction is correct only when `parse_status == OK` and `predicted_root_cause_code` exactly equals the frozen ground-truth root-cause code. Invalid JSON, missing labels, unknown labels, and partial outputs are recorded as prediction rows and count as incorrect; they never crash the whole evaluation run.

Hierarchical accuracy is supporting only. It maps both the predicted and true root-cause codes through frozen taxonomy categories and checks category equality. Top-3 accuracy is supporting only and checks whether the true code appears in the first three ranked labels. When no ranked list is present, the single predicted label is the only rank considered.

## Probability and confidence contract

A number is not reported as confidence unless it has a named reproducible source. Phase 05 supports:

- `normalized_label_sequence_log_likelihood`: compute one sequence log-likelihood for every allowed root-cause code, apply log-sum-exp normalization over exactly the frozen label set, and use the predicted label probability as confidence.
- `temperature_scaled_label_sequence_log_likelihood`: divide those label sequence log-likelihoods by one scalar temperature fitted on validation predictions only, then normalize identically.

The scoring helper rejects missing/extra labels and non-finite scores. Temperature fitting rejects any source split other than `validation`. The fitted calibration version hashes the method, label set, fitted temperature, source split, and validation NLL.

## Calibration metrics

For K root-cause labels, normalized multiclass Brier score is

`mean_i [ 0.5 * sum_k (p_ik - y_ik)^2 ]`.

The factor 0.5 makes the multiclass score bounded by `[0,1]` for valid probability vectors while preserving zero for a perfect forecast.

Expected calibration error uses deterministic fixed-width confidence bins on `[0,1]`. Confidence exactly equal to `1.0` belongs to the final bin. For each non-empty bin, reliability data stores count, empirical accuracy, mean confidence, and absolute gap. ECE is the sample-count-weighted mean absolute gap across non-empty bins. Reliability-diagram data is persisted in ECE metric metadata.

## Latency and cost

Latency p50 and p95 use the deterministic R-7 linear quantile definition (`position = (n-1)q`). Metrics require latency for every prediction in the scored run.

Marginal cost/query is `sum(per_prediction_cost) / N`. Amortized cost/query is `(upfront_cost + sum(per_prediction_cost)) / N`. Later fine-tuning phases must supply training/upfront cost when making amortized economic comparisons.

## Retrieval metrics

Deterministic context precision/recall use reference-relevant chunk IDs when such references exist. For each evaluable incident:

- context precision = relevant retrieved chunks / retrieved chunks, with zero when nothing is retrieved;
- context recall = relevant retrieved chunks / reference-relevant chunks.

The reported values are macro averages over incidents that have reference-relevant chunks. Incidents without retrieval references are not silently treated as zero-relevance cases.

RAGAS or LLM-judge metrics enter only through the `supporting.ragas.*` namespace and carry an evaluator version. They cannot replace or decide the deterministic primary root-cause metric.

## Paired statistics

Two experiments can be compared only when their correctness vectors cover exactly the same held-out incident IDs. Phase 05 implements:

- paired bootstrap confidence intervals for accuracy difference `B - A`, resampling incident pairs together with a recorded deterministic seed;
- two-sided exact McNemar/binomial testing on discordant correctness pairs.

Point-estimate differences alone are never labeled statistically meaningful.

## Failure taxonomy

Taxonomy version: `phase5-failure-taxonomy-v1`.

Deterministic rules may assign:

- `HALLUCINATED_EVIDENCE`: a cited chunk ID was not present in retrieved context;
- `CORRECT_CATEGORY_WRONG_CAUSE`: exact label wrong but taxonomy category correct;
- `OVERCONFIDENT_WRONG`: incorrect at confidence >= 0.80;
- `UNDERCONFIDENT_CORRECT`: correct at confidence <= 0.50;
- `RETRIEVAL_MISS`: reference-relevant chunks exist and none were retrieved;
- `OTHER_UNCLASSIFIED`: incorrect/parse-failed prediction matched no more specific deterministic rule.

`ANCHORING` and `INSUFFICIENT_CONTEXT` are intentionally not inferred from weak heuristics. They require a manual or secondary annotation with explanatory notes. Manual/secondary annotations always require notes.

## Persistence and reproducibility

Phase 05 reuses the Phase 04 PostgreSQL schema. Canonical prediction payloads are stored in `predictions.output_json`; scalar confidence/latency remain duplicated in typed columns for queryability. Metrics are persisted in `metrics`, failure labels in `failure_annotations`, and all rows receive deterministic IDs derived from run/incident/metric semantics. Every persisted metric records the evaluator result hash; ECE also records reliability-bin data.

A stored run can be reloaded into canonical predictions and rescored by the same harness. Phase completion requires the persisted hash and metric values to match the live evaluation and an idempotent second persistence attempt to return the same stored snapshot.

## Golden fixture

`backend/tests/fixtures/phase5/golden_metrics.json` is the independent hand-worked fixture. Its expected values are documented in `evidence/phase-05/golden-fixture-calculations.md`. Tests compare the implementation against those values; expected values must not be rewritten merely to fit implementation output.
