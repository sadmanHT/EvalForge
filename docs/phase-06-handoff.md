# Phase 06 Handoff — Zero-shot baseline

## Completion state

Phase 06 reached the repository `complete` state after one accepted real-model validation run, manual review, protocol freeze, and exactly one authorized locked-test run. The final evidence is machine-checkable with `scripts/check_phase6_contract.py` and `scripts/check_phase6_exit.py`.

## Frozen scientific identity

- Protocol: `phase6-zero-shot-baseline-v1`
- Dataset: `evalforge-incident-diagnosis-v0.1.0`
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Model revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`
- Seed: `20260908`
- Scientific config hash: `15fdda8fdd9f33e3b3d59f7d3f7971751124a25d321728576d85af8cb690a18e`
- GPU/software environment fingerprint: `45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff`
- Locked-test frozen code commit: `b3c2b88a96b72937ca91ba36e4138b9b82ae34d7`

The real runtime used Python 3.12.13, `torch==2.10.0+cu128`, `transformers==4.57.6`, `accelerate==1.13.0`, `bitsandbytes==0.50.2`, `huggingface-hub==0.36.2`, `safetensors==0.7.0`, one visible Tesla T4, CUDA 12.8 through PyTorch, 4-bit NF4 quantization, and deterministic decoding.

## Validation evidence

- Run ID: `phase6-zero-shot-validation-kaggle-v2`
- Validation evidence seal: `184ed6f973bfc69ff5bc7becac5f60b1650ce7368cb36cfec89bcd6e1e344e1f`
- Validation evidence file SHA-256: `9de27a3ea03d7b05c91180546d02eb384ce9143be078f5a932ce08592235fdb0`
- Manual reviewer: `project-owner`
- Review accepted all six validation predictions for schema compliance, allowed-label compliance, and absence of prompt/parser defects.
- Validation exact accuracy: `1.0`
- Validation parse failure rate: `0.0`

The validation review explicitly records a limitation: several validation narratives are unusually diagnosis-explicit, so the perfect validation score is not interpreted as broad generalization.

## Protocol freeze

`evidence/phase-06/protocol-freeze.json` binds the accepted validation evidence, human review, scientific config hash, and sealed validation GPU host before locked-test authorization. The protocol remained frozen for the test execution, and the test host fingerprint exactly matched the frozen validation fingerprint.

## Single locked-test result

- Experiment ID: `phase6-zero-shot-test-kaggle-v1`
- Run ID: `phase6-zero-shot-test-kaggle-v1`
- Test evidence seal: `76ee0677d3b3eebc51169fbbe21f323e6c6b23cb3dffabcffbc2f8a0685dd3c8`
- Test evidence file SHA-256: `9e5471525e44863e182b9a5d126f879f3934fcf46d0d6c043df53b4aed6416f9`
- Test GPU-host evidence seal: `c4d0668ad4df68b7f50c13a0c030a98e7345c2e72a9212446d6a42e2025b32cb`
- Test GPU-host file SHA-256: `a1c37799a40170666a5d36d7e3bbba57fbbcab27dd76ec3072946af50b6812a2`
- Prediction count: `6`
- Inference failures: `0`
- Cost records: `6`
- Metric recomputation verified: `true`

| Metric | Locked-test value |
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

All six held-out predictions matched their frozen gold labels, each prediction parsed successfully, and no technical inference failure was recorded.

## Experiment tracking

W&B tracking was required and enabled for the locked test.

- Run: `https://wandb.ai/sadman-hasan-t-islamic-university-of-technology/evalforge-phase6/runs/1xlznr26`
- Artifact: `phase6-zero-shot-test-kaggle-v1-result`

The tracking references are embedded in `test-run.json`, so the repository completion gate verifies their presence rather than relying on a dashboard claim.

## Cost semantics

The Kaggle execution used `gpu_hour_usd=0.0` with snapshot `kaggle-free-quota-no-direct-usd-per-gpu-hour`. This is the repository's explicit zero-direct-cost sentinel for Kaggle quota execution; it is not a fabricated provider price. Every test prediction has a versioned cost row, positive inference latency, zero GPU hourly rate, and zero direct marginal amount. Other snapshots retain the strict-positive paid-rate rule.

## Portable execution bundle

The Kaggle notebook produced `EvalForge_Phase6_Kaggle_Locked_Test_Artifacts_v1.zip` with SHA-256 `763c93bbfe02c2b5f028f5afab553bd62c99f0113bdf8edb803351276c71c716`. It contains the validation/freeze/test evidence copies, runtime package list, NVIDIA snapshot, PostgreSQL dump, zero-cost provenance note, auxiliary runtime dependency pins, and the completed single-run sentinel. The committed `test-run.json` and `test-gpu-host.json` are byte-identical to the copies in that portable bundle.

## Interpretation limits

The locked test is only six incidents, one per taxonomy class. The dataset is intentionally family-disjoint across train/validation/test, but multiple incident narratives directly expose or strongly imply the diagnosis. Therefore the result demonstrates that the frozen zero-shot baseline, evidence pipeline, calibration/cost accounting, reproducibility controls, and completion gates work on the Phase 06 benchmark. It should not be presented as evidence that the model has solved general production-incident diagnosis.

## Final gate contract

Phase 06 is complete only when the final repository head passes all of the following:

```bash
make verify-all
python scripts/check_phase6_contract.py
python scripts/check_phase6_exit.py
```

The cumulative gate now includes `phase6-exit`. CI must also pass the fresh no-cache Compose smoke on the same final head before the phase is considered closed.
