# Phase 07 Handoff — Knowledge-base retrieval foundation

## Completion state

Phase 07 reaches the repository `complete` state when the preserved KB/index/leakage/CI evidence validates, the project-owner retrieval review is approved, and `scripts/check_phase7_exit.py` passes. The cumulative `make verify-all` and fresh Compose paths now execute the Phase 07 hard exit as part of their normal guardrails.

## Frozen retrieval identity

- KB version: `evalforge-kb-v0.1.0`
- Dataset version: `evalforge-incident-diagnosis-v0.1.0`
- KB manifest checksum: `a55b666c6508c4a1190b70c5fcae1694c54275afc5380f6bed967c95894c51b5`
- Embedding adapter: `evalforge/feature-hash-embedding`
- Embedding revision: `phase7-feature-hash-v1`
- Embedding dimension: `1536`
- Distance: cosine
- Default top-k: `5`
- Chunker: `phase7-token-window-v1`
- Chunk size: `96` tokens
- Chunk overlap: `24` tokens
- Research leakage policy: `phase7-leakage-policy-v1`
- Reranker: none in the Phase 07 baseline

The Phase 07 embedding is intentionally deterministic and CPU-only. It is a reproducibility-oriented retrieval baseline, not a claim that feature hashing is the preferred production semantic embedding model.

## Knowledge-base contents

The canonical index contains 24 provenance-bearing documents and 25 logical chunks:

- 18 historical incident documents from the frozen research dataset;
- 6 authored operational runbooks, one for each root-cause label.

Historical validation/test incident documents remain indexable for integrity and unrestricted diagnostic tests, but research-mode retrieval rejects them. Only train-split historical incidents and generic runbooks are research-eligible.

## Retrieval and leakage guardrails

Research-mode retrieval enforces all of the following at query time:

- `research_eligible=true`;
- no `validation` or `test` source split;
- exclusion of a matching source family when a query family is supplied;
- optional source-type filtering;
- deterministic cosine nearest-neighbor ordering with chunk-ID tie breaking.

The preserved leakage audit covers all 12 held-out incidents spanning 12 held-out families. Every query returns only research-eligible records, returned historical incident splits are train-only, and known held-out-family leakage is zero.

## Index and persistence evidence

Phase 07 adds pgvector/HNSW retrieval indexes through Alembic revision `0003_phase07`. Reindexing is logically idempotent: document/chunk counts and logical chunk IDs remain stable across repeated persistence of the same manifest.

The preserved index evidence records:

- document count: `24`;
- chunk count: `25`;
- logical chunk-ID SHA-256: `e00913ac1249c340bdb079eeb7651f9f3ff3ac398791333f912bbc26fef0009f`;
- reindex verified: `true`.

## Preserved CI evidence

`evidence/phase-07/ci-validation.json` binds a real green GitHub Actions execution:

- workflow run: `34719281051` (run number `320`);
- evidence head: `9085f916f1fd51c1cab76fb808b34c0554bc22d4`;
- `verify-all`: success;
- clean Compose: success;
- retrieval evidence artifact ID: `10305159236`;
- clean-environment artifact ID: `10305847179`.

That evidence proves the Phase 07 retrieval contract, indexing path, leakage audit, earlier Phase 04–06 exits, five-service health, and clean-environment behavior before manual review. The final branch head must also be green after the review/exit-gate closeout; this is checked in GitHub Actions rather than self-referentially rewriting the preserved CI evidence on every final commit.

## Manual retrieval review

The preserved sample set contains:

- train: `incident-0557d490e1a492950c4ac3e4` (`Payment Issues`);
- validation: `incident-215133387692e86f737df1e9` (`Payment configuration update prevented U.S. credit-card processing`).

The validation sample ranks the correct `evalforge://runbooks/payment-configuration` document first. The shorter train sample ranks that same correct runbook second, behind an unrelated N+1-query historical incident.

The project owner explicitly approved the traces and acknowledged that ranking limitation. Approval is recorded in `evidence/phase-07/retrieval-review.json` with reviewer `project-owner`. The weak train-sample ranking is treated as a known quality limitation of the deterministic feature-hash baseline, not as a leakage defect and not as a reason to tune against held-out outcomes.

## Interpretation limits

Phase 07 proves a reproducible, leakage-safe retrieval foundation with versioned provenance, deterministic indexing, pgvector persistence, clean-environment verification, and explicit human inspection. It does not establish state-of-the-art retrieval quality.

The feature-hash adapter is intentionally simple, there are only 24 documents, and only two sample traces receive manual relevance inspection. Controlled retriever, embedding, top-k, and reranker comparisons should be performed in the next experimental phase using predeclared validation procedures rather than tuning from locked test outcomes.

## Final gate contract

Phase 07 is complete only when the final repository head passes all of the following:

```bash
make verify-all
python scripts/check_phase7_contract.py
python scripts/index_kb.py
python scripts/audit_phase7_leakage.py
python scripts/check_phase7_exit.py
```

The final GitHub Actions run must also pass the fresh no-cache Compose job, including the Phase 07 hard-exit check, before the phase is considered closed.
