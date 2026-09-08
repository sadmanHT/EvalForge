# ADR-006 — RAG and Evaluation Integrity

**Status:** Accepted

RAG uses versioned pgvector retrieval with document/chunk provenance and a configurable reranker interface. Research mode must exclude documents derived from held-out incident families. The primary ground truth is deterministic root-cause codes; RAGAS and LLM judges are supporting evaluators only. All four pipelines enter one common evaluation harness.
