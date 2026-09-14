# Phase 08 handoff — RAG pipeline, retrieval evaluation, and controlled ablations

## Completion state

Phase 08 is complete. The RAG-only arm was implemented, seven pre-registered retrieval variants
were executed on validation, `rag-top-k1` was selected by the frozen validation-only policy, the
protocol was frozen, and exactly one authorized locked-test RAG execution was consumed. The final
repository gate validates the complete validation bundle, freeze record, locked-test evidence, GPU
fingerprint parity, and paired Phase 06 baseline comparison.

## Canonical scientific identity

- protocol: `phase8-rag-protocol-v1`
- scientific config SHA-256: `be9f7431892ca7d1f1568894a9d769147730593f7f1d51dffbf656b25510ae63`
- ablation-suite SHA-256: `c08e351c61215fb96fa5965dd9a51da37042a7fbec7db78a74fea7a819a68d62`
- selected variant: `rag-top-k1`
- base model: `mistralai/Mistral-7B-Instruct-v0.3`
- base model revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`
- prompt: `rag-context-v1`
- evaluator: `phase5-evaluator-v1`
- research leakage policy: `phase7-leakage-policy-v1`
- selected KB: `evalforge-kb-v0.1.0`
- selected embedding: `evalforge/feature-hash-embedding@phase7-feature-hash-v1`
- selected chunking: `phase7-token-window-v1`, size `96`, overlap `24`
- selected retrieval depth: `top_k=1`
- selected reranker: none
- GPU/software fingerprint: `45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff`

## Validation and selection

The accepted validation evidence is the complete clean replacement v2 suite from commit
`46c187476076c4b5874267fb1fee9d07133371e9`. It contains all seven registered one-factor-at-a-time
variants with W&B tracking and the same accepted GPU/software fingerprint. The deterministic
selection policy is `phase8-validation-selection-v2-missing-ece-worst`: exact accuracy,
hierarchical accuracy, retrieval recall, lower ECE, lower p95 latency, then lexical variant ID.
The selection report SHA-256 is
`68bdd0435bc60a431343bf6eed2cc0b0465e99e443bb387ee7da97f46fa62ce1`.

The first external validation attempt from commit
`9ff1f93ac645102ea0a9ada7fe2ed1cefd17421a` is explicitly excluded. It was technically incomplete
because the Phase 08 evidence layer initially required ECE even when the common evaluator properly
omitted it for a parse-failure prediction without confidence. The repair only completed the
predeclared ECE tie-break behavior; the accepted v2 suite reran every variant from a fresh runtime.

## Locked-test execution

The protocol was frozen before test access. The one authorized locked-test execution used frozen
commit `af48887816634f444a57010a8e76881caefae7cc`, selected variant `rag-top-k1`, and run ID
`phase8-rag-top-k1-test-kaggle-v1`. The run completed all six held-out incidents with zero inference
retries, zero parse failures, six retrieval traces, six cost records, configured W&B tracking, and
metric recomputation verified.

The sealed test evidence SHA-256 is
`bb8e12f9f63ee939e630535bd38f815779f09f23779202be19cf7c0164e0d956`. The test host fingerprint
exactly matches the frozen validation host.

The Kaggle notebook failed only in a post-run sanity-check cell after the canonical runner had
already printed `PHASE08_LOCKED_TEST_RUN=PASS`: the cell used `trace["provenance"]` instead of the
canonical `trace["metadata"]["provenance"]` location. This did not alter or invalidate model
inference, persistence, metric recomputation, the W&B artifact, or the sealed test evidence. The
locked test was not rerun.

## Final paired result

On the six held-out incidents, the frozen Phase 06 zero-shot baseline scored `6/6` exact accuracy
while the selected Phase 08 RAG pipeline scored `5/6` (`0.833333...`). RAG was wrong only on
`incident-3d1ad2d09d88df9038f46fd6`, predicting `memory_leak` for a `disk_exhaustion` incident; its
retrieved top-1 context was an unrelated train memory-leak incident, so the failure is also tagged
`RETRIEVAL_MISS`.

The paired accuracy delta (RAG minus baseline) is `-1/6`. The 95% paired bootstrap interval is
`[-0.5, 0.0]`, and the exact McNemar p-value is `1.0` with one discordant pair. The comparison
SHA-256 is `f763e9ffa94b7fecbe7e445e55d4249f6d81cc1fd729a8f7e0a96097a24ee13f`. Given six held-out
examples, these figures do not establish broad statistical superiority.

RAG test retrieval precision and recall are both `1/6`. This confirms the Phase 07/08 lexical
retrieval foundation remains the principal quality limitation. The locked-test outcome must not be
used to tune it. Any semantic embedding, reranker, chunking, top-k, prompt, or context-policy
change belongs to a new protocol and validation cycle.

## Evidence and reproduction

Canonical Phase 08 evidence lives under `evidence/phase-08/`:

- `validation/*.json` — seven accepted validation run seals
- `validation-gpu-host.json` — accepted validation host
- `ablation-report.json` — deterministic ranking and selected variant
- `validation-comparison.json` — paired validation baseline/RAG comparison
- `protocol-freeze.json` — immutable freeze record
- `test-gpu-host.json` — locked-test host evidence
- `test-run.json.gz` — deterministic gzip archive of the complete raw locked-test RAG evidence
  - raw JSON SHA-256 before gzip: `c400991f84e8eadd13f7c95b2d04a2f56effd6ea12a141f321c117fd6caff028`
  - evidence seal inside the JSON: `bb8e12f9f63ee939e630535bd38f815779f09f23779202be19cf7c0164e0d956`
- `baseline-vs-rag-test-comparison.json` — final paired held-out comparison

Primary verification commands:

```bash
python scripts/check_phase8_exit.py
make test-phase8
make verify-all
make fresh-smoke
```

`make verify-all` and the clean-environment smoke both execute `phase8-exit` after final wiring.

## Implemented, proven, deferred

Implemented: deterministic RAG context construction, leakage-safe retrieval, provenance-bearing
traces, variant indexing, configurable reranking boundary, validation ablations, deterministic
selection, freeze enforcement, locked-test authorization, W&B/cost evidence, paired bootstrap and
McNemar comparison, and the Phase 08 completion gate.

Proven: exact validation/test split coverage, no held-out-family retrieval leakage under the
canonical evidence validator, selected-variant enforcement, GPU/software fingerprint parity,
metric recomputation, one locked-test execution, and paired Phase 06 comparison.

Deferred: retrieval-quality improvements. The accepted top-1 lexical retriever has low held-out
context precision/recall and one clear retrieval-induced error. Do not repair it inside Phase 08;
carry that limitation into the next research protocol. There are no remaining Phase 08 blocker or
critical items after the final cumulative CI and clean-environment gates are green.
