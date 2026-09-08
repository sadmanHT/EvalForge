# ADR-006 — RAG and Evaluation Integrity

**Status:** Accepted

## Context
RAG can leak held-out answers through historical incidents/documents, and LLM judges can confound deterministic classification evaluation.

## Decision
Research-mode retrieval is provenance-aware and excludes held-out-family answer leakage. Structured root-cause-code exact match is the primary evaluator. RAGAS/LLM judges are supporting evaluators only.

## Consequences
Phase 07 must version KB documents/chunks and enforce eligibility filters. Primary leaderboard results cannot be manually overridden by judge scores.
