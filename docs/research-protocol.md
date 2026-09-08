# EvalForge Research Protocol v1.0.0

**Study ID:** `evalforge-incident-diagnosis-primary-v1`  
**Domain:** production incident diagnosis / root-cause classification  
**Status:** Phase 01 protocol frozen; concrete benchmark manifests are created and locked in Phase 03.

## 1. Scoped research question

Within production incident diagnosis, how do four strategies—zero-shot inference, retrieval-augmented generation (RAG), domain LoRA/QLoRA fine-tuning, and fine-tuning plus RAG—trade off exact root-cause accuracy, calibration, retrieval quality, latency, marginal cost, amortized cost, and failure modes when evaluated under one common benchmark and evaluator?

This is a domain-specific study. Results must not be generalized into universal claims about RAG or fine-tuning across unrelated tasks, datasets, model families, or deployment environments.

## 2. Pre-specified hypotheses

The hypotheses are recorded before the held-out benchmark is used for model/prompt/retrieval selection.

- **H1 — Adaptation benefit:** at least one adapted arm (`RAG`, `FINETUNED`, `COMBINED`) will improve exact root-cause-code accuracy over `ZERO_SHOT` on the locked incident-diagnosis benchmark.
- **H2 — Retrieval mediation:** RAG-family errors will be materially associated with retrieval failures; cases with relevant eligible evidence retrieved should outperform cases with retrieval misses.
- **H3 — Fine-tuning efficiency trade-off:** `FINETUNED` may reduce marginal inference overhead relative to RAG-family arms, but any economic advantage must be assessed after amortizing training cost.
- **H4 — Combined trade-off:** `COMBINED` may achieve the strongest exact accuracy, but it is not assumed to dominate latency or cost; superiority claims require metric-specific evidence.
- **H5 — Calibration is not assumed to track accuracy:** an arm may improve accuracy while worsening Brier score/ECE. Calibration is therefore evaluated independently rather than inferred from accuracy.

These are falsifiable study hypotheses, not promised outcomes. Null or contrary findings are valid results.

## 3. Primary comparison arms

| Arm | Frozen base model | Retrieval | Adapter |
| --- | --- | --- | --- |
| `ZERO_SHOT` | yes | no | no |
| `RAG` | yes | yes | no |
| `FINETUNED` | yes | no | yes |
| `COMBINED` | yes | yes | yes |

The base-model ID and exact revision are identical across all four primary arms. Retrieval state and adapter state are the intended independent variables.

## 4. Ground truth and labels

The primary target is the canonical structured field `root_cause_code`. The EvalForge taxonomy is versioned independently in `configs/label-taxonomy.yaml`.

Phase 01 records a read-only external upstream label snapshot from the incident benchmark source. EvalForge does not mutate that external project. Category metadata in EvalForge is EvalForge-owned and versioned.

Unknown labels are invalid. Adding, removing, renaming, merging, splitting, or recategorizing canonical labels requires a taxonomy version bump and a comparability review.

## 5. Outcomes

### 5.1 Primary outcome

- **Exact root-cause-code accuracy** on the locked held-out evaluation set.

### 5.2 Secondary classification outcomes

- hierarchical/category accuracy
- top-3 accuracy

### 5.3 Calibration outcomes

- Brier score
- expected calibration error (ECE)
- reliability-diagram data

### 5.4 Retrieval-supporting outcomes

For RAG-capable arms only:

- context precision
- context recall
- retrieval miss rate
- faithfulness / RAGAS outputs as supporting evidence

RAGAS or LLM-judge outputs may not decide the primary winner when deterministic ground-truth root-cause labels exist.

### 5.5 Efficiency outcomes

- latency p50 and p95
- marginal cost/query
- queries/dollar
- amortized total cost/query at declared query volumes

### 5.6 Failure taxonomy

At minimum:

- `HALLUCINATED_EVIDENCE`
- `ANCHORING`
- `INSUFFICIENT_CONTEXT`
- `CORRECT_CATEGORY_WRONG_CAUSE`
- `OVERCONFIDENT_WRONG`
- `UNDERCONFIDENT_CORRECT`
- `RETRIEVAL_MISS`
- `OTHER`

Failure labels support analysis; they do not replace deterministic primary grading.

## 6. Dataset split and leakage policy

1. The unit of independence is the **incident family**, not the row.
2. Incident families are partitioned into train/validation/test before any synthetic augmentation.
3. Synthetic descendants may be generated only from training families.
4. Validation controls prompt design, retrieval configuration, calibration fitting, checkpoint selection, and training hyperparameters.
5. The held-out test set is locked once Phase 03 creates its manifest. Test outcomes may not choose prompts, retriever settings, calibration parameters, checkpoints, thresholds, or training hyperparameters.
6. Research-mode RAG must exclude documents derived from held-out incident families where they reveal the target answer.
7. If the benchmark is too small for a defensible holdout, the limitation must be reported or additional independent real incidents collected. Synthetic held-out variants may not be presented as independent real evidence.

## 7. Fairness controls for the primary study

Unless a protocol amendment explicitly states otherwise, the following are frozen across primary arms:

- base model ID and exact revision
- label taxonomy version
- held-out incident IDs and test-manifest checksum
- prompt/output-schema family
- decoding policy and generation limits
- confidence method
- evaluator version and metric implementation
- paired case alignment
- hardware/runtime timing methodology

Pipeline-specific differences are limited to retrieval configuration and/or adapter revision as appropriate.

## 8. Confidence source

Preferred confidence is model-derived rather than an unconstrained self-report:

1. Score every allowed root-cause label with normalized sequence log-likelihood under the active pipeline context.
2. Normalize label scores into a probability distribution.
3. If calibration is applied, fit temperature scaling on validation predictions only.
4. Apply the frozen calibration transform unchanged to held-out test predictions.

If a runtime cannot expose suitable token scores, a self-reported confidence value may be stored only with source `self_reported`; it must never be described as equivalent to a calibrated model probability.

## 9. Statistical analysis

Because every primary pipeline evaluates the same held-out incidents, comparisons are paired.

Required reporting for the final four-way study:

- point estimate for exact accuracy
- 95% confidence interval
- paired accuracy delta for key comparisons
- paired bootstrap confidence interval for deltas
- McNemar test, paired permutation, or another pre-documented paired procedure for binary correctness when appropriate
- effect size/delta alongside any p-value

Overlapping or non-overlapping marginal confidence intervals alone do not establish paired significance. A larger percentage alone is not sufficient to claim a meaningful win.

## 10. Cost accounting

Report both:

- **marginal inference cost/query**, and
- **amortized total cost/query** at declared query volumes.

Fine-tuning economics include training compute and material adapter/model hosting costs. RAG economics include retrieval, embedding/reranking, and additional prompt/inference cost. Pricing assumptions are versioned and timestamped.

## 11. Experiment identity and reproducibility

An experiment may not enter `RUNNING` unless it contains every field declared in `configs/study.yaml`. Fields that are genuinely not applicable remain explicit `null`; omission is not allowed.

Pipeline-specific required fields are enforced by the Phase 01 contract validator. Completed experiment identities are immutable; meaningful changes create a new version/config hash.

## 12. Primary-run invalidation rules

A primary experiment is invalid if any of the following is discovered:

- cross-split incident-family leakage
- held-out-family answer leakage through research-mode RAG
- wrong/mismatched base-model revision
- different held-out case set without a documented protocol amendment
- test-guided tuning
- corrupted, missing, or duplicated predictions that break paired alignment
- evaluator or metric bug that affects reported results
- manually substituted or untraceable dashboard metrics

Invalid runs remain auditable but may not support final claims.

## 13. Allowed and disallowed claims

Allowed form:

> On EvalForge incident-diagnosis benchmark version X, pipeline A changed exact root-cause-code accuracy by Y percentage points versus pipeline B, with Z uncertainty under protocol v1.0.0.

Disallowed form:

> Fine-tuning is better than RAG for LLMs.

All claims must identify the benchmark/version, metric, comparison arm, and uncertainty where applicable.

## 14. Protocol-change rule

Any change that can alter scientific meaning requires:

1. a protocol/version bump,
2. an ADR or protocol amendment,
3. a comparability assessment for prior experiments,
4. rerunning affected validation/primary experiments when necessary.

The locked test set must never be retrospectively redefined to rescue a result.

## 15. Phase boundaries

Phase 01 freezes scientific and architecture contracts only. It does not implement the database, API, worker, retrieval, training, or UI systems assigned to later phases.
