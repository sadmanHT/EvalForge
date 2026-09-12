# Phase 06 Evidence Directory

Phase 06 is complete. The repository now preserves the accepted real-model validation run, the human validation review, the protocol freeze, and the single authorized locked-test baseline with matching sealed GPU/software evidence and configured W&B tracking.

Committed evidence:

- `validation-gpu-host.json` — sealed GPU/software host evidence captured before validation.
- `validation-run.json` — deterministic validation export from PostgreSQL.
- `validation-review.json` — human review covering every validation prediction.
- `protocol-freeze.json` — freeze record binding the accepted validation result, review, scientific configuration, and GPU/software fingerprint before test authorization.
- `test-gpu-host.json` — sealed locked-test host evidence. Its stable environment fingerprint exactly matches validation.
- `test-run.json` — deterministic export from the single authorized locked-test baseline, including raw predictions, metrics, one cost record per prediction, scientific identity, and W&B references.

The locked test used run ID `phase6-zero-shot-test-kaggle-v1`. Its evidence seal is `76ee0677d3b3eebc51169fbbe21f323e6c6b23cb3dffabcffbc2f8a0685dd3c8`, and the frozen scientific configuration hash is `15fdda8fdd9f33e3b3d59f7d3f7971751124a25d321728576d85af8cb690a18e`. Validation and test both bind to GPU/software environment fingerprint `45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff`.

Locked-test metrics:

| Metric | Value |
| --- | ---: |
| Exact accuracy | 1.0 |
| Top-3 accuracy | 1.0 |
| Hierarchical accuracy | 1.0 |
| Parse failure rate | 0.0 |
| ECE | 0.057461652760390294 |
| Normalized multiclass Brier | 0.015977908490571832 |
| p50 latency | 6110.058298500007 ms |
| p95 latency | 7539.785869499894 ms |
| Marginal mean direct cost | $0.00 |
| Amortized mean direct cost | $0.00 |

W&B tracking is configured in the locked-test export. The run reference is `https://wandb.ai/sadman-hasan-t-islamic-university-of-technology/evalforge-phase6/runs/1xlznr26`, and the artifact reference is `phase6-zero-shot-test-kaggle-v1-result`.

The zero direct cost is intentional rather than missing accounting. Paid-rate snapshots remain strictly positive. The explicitly named `kaggle-free-quota-no-direct-usd-per-gpu-hour` snapshot is the sole zero-direct-cost exception: every cost record uses that snapshot, records a zero GPU hourly rate and zero marginal amount, preserves positive inference latency, and reports zero marginal mean cost.

Repository completion is machine-checkable:

```bash
python scripts/check_phase6_contract.py
python scripts/check_phase6_exit.py
```

Both must report PASS, and `phase6-exit` is part of the cumulative `make verify-all` gate after completion.

The 100% validation and locked-test accuracy should not be interpreted as broad generalization. The frozen benchmark is only 18 incidents total (six per split), with one example per class in validation and test, and several narratives are unusually diagnosis-explicit. The evidence supports reproducibility and protocol completionce for this benchmark; it does not by itself establish robustness to subtler production incidents.
