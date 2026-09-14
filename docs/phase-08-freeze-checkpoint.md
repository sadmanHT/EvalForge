# Phase 08 validation freeze checkpoint

The complete clean replacement Phase 08 validation suite has been accepted and the protocol is now frozen for the single authorized locked-test run.

- Selected variant: `rag-top-k1`
- Selection source: validation only
- Accepted execution source commit: `46c187476076c4b5874267fb1fee9d07133371e9`
- Accepted environment fingerprint: `45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff`
- Frozen protocol record: `evidence/phase-08/protocol-freeze.json`
- Locked-test authorization: enabled for the frozen selected variant only

The first incomplete validation attempt remains excluded from scientific selection evidence. The next scientific action is exactly one RAG locked-test execution of `rag-top-k1`; the Phase 06 locked test must not be rerun. Test outcomes must not be used to tune retrieval, prompts, context budgets, or variant selection.
