# UCI / ServiceNow incident-management source snapshot

This directory preserves a byte-equivalent-content transport archive for UCI dataset 498, **Incident management process enriched event log**. EvalForge verified that the user-provided Kaggle/mirror ZIP and the current official UCI ZIP have different envelope hashes but contain the exact same sole `incident_event_log.csv` payload.

The source is real operational data extracted from a ServiceNow instance and anonymized by its publishers. It is retained as an **auxiliary real-world robustness/provenance corpus**. It is **not** mapped into EvalForge's frozen root-cause taxonomy because the source intentionally omits textual attributes and exposes anonymized categorical codes rather than trustworthy human-readable RCA labels.

Never infer labels such as `memory_leak`, `n_plus_one_query`, or `database_connection_leak` from anonymous `Category N`, `Symptom N`, or closure codes.

See `SOURCE.json` and `SHA256SUMS.txt` for provenance and integrity metadata.

## Stable source identity

EvalForge keys the scientific source identity to the inner CSV SHA-256 `fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef`. `SOURCE.json` records both accepted ZIP-envelope hashes. A rebuild must match the canonical CSV payload and one accepted transport envelope; a changed CSV is rejected.
