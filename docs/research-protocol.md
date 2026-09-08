# EvalForge Research Protocol v1.0.0

## Scoped research question

Within **production incident diagnosis/root-cause classification**, how do four strategies—zero-shot inference, retrieval-augmented generation (RAG), domain LoRA/QLoRA fine-tuning, and fine-tuning plus RAG—trade off correctness, calibration, retrieval quality, latency, marginal cost, amortized cost, and failure modes when evaluated under the same benchmark and common evaluator?

This is a domain-specific study. It must not be presented as proving universal superiority of one strategy for LLM applications.

## Primary comparison arms

| Arm | Frozen base model | Retrieval | Adapter |
|---|---|---|---|
| ZERO_SHOT | yes | no | no |
| RAG | yes | yes | no |
| FINETUNED | yes | no | yes |
| COMBINED | yes | yes | yes |

The **base model ID and exact revision are identical across all four primary arms**. Retrieval state and adapter state are the intended independent variables.

## Ground truth and label contract

The primary target is a structured `root_cause_code`. The EvalForge label taxonomy is independently versioned and may only change through an explicit taxonomy version bump.

Phase 01 records a read-only external upstream label snapshot observed from `sadmanHT/OpsSentinel` at commit `40467b27085130c110098cac5077fb62dee4e5aa`. EvalForge does not modify that repository. Observed codes are:

- `n_plus_one_query`
- `database_connection_leak`
- `disk_exhaustion`
- `broken_payment_configuration`
- `memory_leak`
- `no_fault`

EvalForge-owned hierarchy/category metadata is not claimed to be upstream OpsSentinel taxonomy.

## Outcomes

### Primary metric
- exact root-cause-code accuracy on the locked held-out evaluation set

### Secondary classification metrics
- hierarchical/category accuracy
- top-3 accuracy

### Calibration
- Brier score
- expected calibration error (ECE)
- reliability-diagram data

### Retrieval-only supporting metrics
- context precision
- context recall
- retrieval miss rate
- faithfulness/RAGAS outputs as supporting evidence only

### Efficiency
- latency p50/p95
- marginal cost/query
- queries/dollar
- amortized total cost/query including fine-tuning cost where applicable

### Failure analysis
At minimum: `HALLUCINATED_EVIDENCE`, `ANCHORING`, `INSUFFICIENT_CONTEXT`, `CORRECT_CATEGORY_WRONG_CAUSE`, `OVERCONFIDENT_WRONG`, `UNDERCONFIDENT_CORRECT`, `RETRIEVAL_MISS`, `OTHER`.

LLM judges and RAGAS may support explanation/retrieval analysis but may not determine the primary winner when deterministic root-cause labels exist.

## Dataset and split policy

1. The unit of independence is the **incident family**, not an individual row.
2. Incident families are partitioned into train/validation/test **before synthetic augmentation**.
3. Synthetic descendants may be generated only from training families.
4. Validation controls prompt design, retrieval configuration, calibration fitting, checkpoint selection, and training hyperparameters.
5. The test set is locked. Test outcomes may not select prompts, retriever settings, calibration parameters, checkpoints, thresholds, or training hyperparameters.
6. RAG knowledge-base provenance must prevent documents derived from held-out incident families from leaking answers in research mode.
7. If the available real benchmark is too small for a defensible holdout, document that limitation or expand independent real cases. Synthetic test cases may not be used to create the appearance of independent evidence.

## Fairness controls

For the primary four-way study, the following must match unless a protocol amendment explicitly says otherwise:

- base model ID and exact revision
- label taxonomy version
- held-out incident IDs
- dataset/test manifest checksum
- prompt/output schema family
- decoding policy and generation limits
- confidence method
- evaluator version
- metric implementation
- case alignment for paired comparisons
- latency measurement method and declared hardware/runtime descriptor

Pipeline-specific differences are limited to retrieval configuration and/or adapter revision as appropriate.

## Confidence source

Preferred confidence is model-derived, not a free-form self-report:

1. Score every allowed root-cause label with normalized sequence log-likelihood under the active pipeline context.
2. Normalize label scores into a probability distribution.
3. If calibration is applied, fit temperature scaling on **validation predictions only**.
4. Apply the frozen calibration transform unchanged to held-out test predictions.

If a runtime cannot expose suitable token scores, self-reported confidence may be stored only with source `self_reported`; it must not be described as equivalent to a calibrated model probability.

## Statistical analysis

Because all primary pipelines evaluate the same held-out incidents, comparisons are paired.

Required reporting:
- point estimate for exact accuracy
- 95% confidence interval
- paired delta for key comparisons
- paired bootstrap confidence interval for deltas
- McNemar test or another pre-documented paired procedure for binary correctness when assumptions are satisfied
- effect sizes/deltas, not p-values alone

Overlapping or non-overlapping marginal confidence intervals alone do not determine significance.

No claim may say a strategy “wins” or is “better” without naming the metric, benchmark version, comparison arm, and uncertainty.

## Cost accounting

Report both:
- **marginal inference cost/query**, and
- **amortized total cost/query** at declared query volumes.

Fine-tuning economic claims must include training compute and any material hosting/artifact costs. RAG costs must include retrieval/embedding/reranking and increased prompt/inference cost where applicable. Pricing assumptions are versioned with timestamps.

## Experiment identity and reproducibility

An experiment may not enter `RUNNING` unless its identity contains every required field declared in `configs/study.yaml`. Non-applicable fields are still present with `null`, so absence cannot be confused with omission.

Completed experiment identities are immutable. A meaningful change creates a new config/version.

## Invalidation rules

A primary experiment is invalid if any of these are discovered:
- cross-split incident-family leakage
- held-out-family answer leakage through RAG
- wrong or mismatched base-model revision
- different held-out case set without protocol justification
- test-guided tuning
- corrupted/missing/duplicated predictions that break paired alignment
- evaluator/metric bug affecting the result
- untraceable manually entered metrics

Invalid runs remain auditable but may not be used for final claims.

## Allowed claims

Allowed: “On EvalForge incident-diagnosis benchmark version X, pipeline A changed exact accuracy by Y percentage points versus pipeline B, with Z uncertainty under the frozen protocol.”

Not allowed: “Fine-tuning is better than RAG for LLMs” or another universal conclusion from this single domain/model family.

## Protocol-change rule

Any change that could alter scientific meaning requires:
1. a version bump,
2. an ADR or protocol amendment,
3. an explanation of whether previous experiments remain comparable,
4. rerunning affected validation/primary experiments when necessary.

The locked test must never be retrospectively redefined to rescue a result.
