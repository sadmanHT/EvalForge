# Phase 01 Handoff

## Current status

**COMPLETE** — the exact frozen model/revision passed a real GPU load + strict structured-output smoke, the generated evidence bundle is preserved unchanged, the machine-checkable exit gate passes, and the cumulative Phase 01 test/static/clean-environment gates are green.

## Implemented and proven outcomes

- scoped research question and pre-specified hypotheses;
- primary/secondary metrics, calibration/confidence, paired statistics, cost accounting, allowed claims, and invalidation rules;
- versioned canonical root-cause taxonomy and change policy;
- base-model selection rule, candidate record, exact frozen model ID/revision;
- architecture contract and nine ADRs;
- project-wide Definition of Done, defect severities, and mandatory repair/retest loop;
- Phase 01 config/label/experiment validators and cumulative tests;
- strict connected-model smoke contract;
- real Kaggle GPU evidence proving the exact frozen revision can load and emit canonical strict JSON;
- machine-checkable integrity validation for the model-smoke JSON, raw/parsed output agreement, taxonomy agreement, runtime metadata, and evidence-bundle checksums.

## Canonical versions

- Study: `evalforge-incident-diagnosis-primary-v1`
- Protocol: `1.0.0`
- Taxonomy: `1.0.0`
- Incident schema: `incident-schema-v1`
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Base revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`
- Research protocol commit: `1f56e7c86b09fa3c467b7728c2e5e013a1e4b1da`
- Phase 01 GPU load mode: `bitsandbytes_4bit_nf4`

## Real GPU smoke evidence

The Phase 01 Kaggle run resolved and loaded the exact frozen Hugging Face revision on Tesla T4 GPU hardware and produced:

```json
{"root_cause_code": "n_plus_one_query", "reasoning": "..."}
```

The output was strict JSON, contained exactly the required keys, used a canonical taxonomy label, and had non-empty reasoning. The generated evidence records the requested, resolved, and loaded config revision as the same immutable SHA.

Preserved generated artifacts:

- `evidence/phase-01/model-smoke.json`
- `evidence/phase-01/model-smoke.txt`
- `evidence/phase-01/environment.txt`
- `evidence/phase-01/SHA256SUMS.txt`

Generated-file checksums:

- `model-smoke.json`: `2e2c0a6b8f69c24e8128733cf0d2de73fcc95884460b752517d7d89599d3fdad`
- `model-smoke.txt`: `e9380ad62c5c2e8db1831e91f64606b0bd18c865afd758a06d059e97081be1e4`
- `environment.txt`: `5ed2cc672af7b73b89906905bf4172e48f5c11b2cebba071c69edeff3eec5350`
- internal `evidence_sha256`: `53cf1617d3d52b48105d267e1ce54d3b26a0c9bb952aed249a48da717f3990cb`

## Phase 01 verification commands

Focused schema/contract suite:

```bash
python -m unittest tests.test_phase1_contracts -v
```

Focused hard-exit/evidence suite:

```bash
python -m unittest tests.test_phase1_exit_gate -v
```

Full Phase 01 suite:

```bash
python -m unittest discover -s tests -p 'test_phase1_*.py' -v
```

Static documentation/config gate:

```bash
python scripts/validate_phase1.py
```

Machine-checkable hard exit gate:

```bash
python scripts/check_phase1_exit.py
```

Required successful marker:

```text
PHASE01_EXIT_GATE=PASS
```

## Clean-environment rule

Phase 02 will establish Docker/Makefile/CI infrastructure. For Phase 01, clean-environment verification means copying/checking out only the repository files into a fresh directory with no developer-local generated state, then running the declared Phase 01 tests and validators. Once Phase 02 exists, `make verify-all` supersedes this temporary Phase 01 command surface.

## Important interpretation boundary

The real smoke used a 4-bit NF4 load path because that is appropriate for Kaggle T4-class compatibility testing. This proves the frozen model/revision is accessible, loadable in the intended GPU development class, and compatible with the strict output contract. It does **not** freeze later benchmark quantization, latency methodology, or training/inference hardware; those remain governed by the fairness and reproducibility rules in the research protocol and later phase contracts.

## Remaining limitations

There are no known blocker or critical Phase 01 defects. Phase 02 infrastructure (containers, Makefile gates, CI, service skeletons) remains intentionally unimplemented because it belongs to the next phase.

## Phase 02 handoff rule

Treat the study/model/taxonomy/protocol contracts and Phase 01 evidence as canonical. Do not change the frozen primary model revision, canonical label IDs, or research rules casually; any such change requires explicit versioning and a documented validity decision.
