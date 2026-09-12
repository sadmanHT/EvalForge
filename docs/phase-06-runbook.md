# Phase 06 real baseline runbook

This runbook is for the two **real** Phase 06 executions only: validation first, then one locked-test baseline after validation review and protocol freeze. CI fixtures and mock backends never satisfy the real-model gate.

## 1. Use the exact study configuration

The canonical protocol is `configs/phase6-baseline.json`. Before validation it must remain:

- state: `validation`
- locked-test authorization: `false`
- model: `mistralai/Mistral-7B-Instruct-v0.3`
- revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`
- runtime: bitsandbytes 4-bit NF4, double quantization enabled, float16 compute
- deterministic generation: sampling disabled, temperature 0
- seed: `20260908`

Do not change prompt, model, decoding, confidence method, evaluator, or label taxonomy after inspecting locked-test outcomes.

## 2. Prepare and prove the Linux CUDA GPU host

`configs/phase6-gpu-host.json` pins the real-host compatibility contract to the Phase 01 proven stack. The real Phase 06 host must use Python 3.12.x and these exact package versions:

- torch `2.10.0+cu128`
- transformers `4.57.6`
- accelerate `1.13.0`
- bitsandbytes `0.50.2`
- huggingface-hub `0.36.2`
- safetensors `0.7.0`

The contract also requires CUDA, compute capability at least 7.5, and at least 14 GiB VRAM on a visible GPU. Phase 01 proved this path on Tesla T4 hardware. Do not substitute a tiny model, CPU-only runtime, or unrecorded dependency stack for the real validation/test runs.

After installing the repository backend lock plus the pinned GPU packages, capture the machine-checkable host evidence:

```bash
python scripts/capture_phase6_gpu_host.py \
  --output evidence/phase-06/validation-gpu-host.json
```

The command verifies the Phase 01 reference evidence, exact package versions, CUDA availability, GPU capability/VRAM, captures `nvidia-smi`, seals the evidence JSON, and prints a stable environment fingerprint. The fingerprint excludes volatile capture time and `nvidia-smi` text but includes Python, exact packages, CUDA runtime, and the visible GPU identities/capabilities.

The canonical worker/CLI needs PostgreSQL because PostgreSQL is the experiment system of record. On a GPU host with Docker available, start the same image used by EvalForge:

```bash
docker run -d --name evalforge-phase6-postgres \
  -e POSTGRES_DB=evalforge \
  -e POSTGRES_USER=evalforge \
  -e POSTGRES_PASSWORD=evalforge \
  -p 5432:5432 \
  pgvector/pgvector:pg16

export DATABASE_URL=postgresql://evalforge:evalforge@localhost:5432/evalforge
cd backend
alembic upgrade head
cd ..
```

## 3. Configure tracking and cost evidence

For the final locked-test hard exit, configure W&B and install the `wandb` package in the GPU environment. Validation may be run without W&B while debugging, but the final locked-test evidence must preserve real W&B run and artifact references.

```bash
export WANDB_PROJECT=<real-project-name>
export WANDB_ENTITY=<real-entity-if-used>
export WANDB_API_KEY=<secret-from-environment>
```

Never commit the API key. The resulting W&B run/artifact references are copied into PostgreSQL and then into the deterministic Phase 06 evidence export.

Use the GPU provider's actual rate for `--gpu-hour-usd`. Give that rate source a stable `--cost-rate-snapshot-version`; do not invent a price. The runner computes per-query marginal inference cost as active inference wall time multiplied by the supplied GPU hourly rate and persists a versioned `cost_records` row for each priced prediction.

## 4. Run the real validation split

The runner no longer accepts a hand-written hardware descriptor. It validates the sealed host evidence and derives the experiment runtime descriptor from its stable environment fingerprint.

```bash
python evals/runner.py \
  --split validation \
  --git-commit "$(git rev-parse HEAD)" \
  --gpu-host-evidence evidence/phase-06/validation-gpu-host.json \
  --cost-rate-snapshot-version "<actual rate snapshot version>" \
  --gpu-hour-usd <actual provider rate> \
  --experiment-id phase6-zero-shot-validation-v1 \
  --run-id phase6-zero-shot-validation-v1 \
  --evidence-output evidence/phase-06/validation-run.json
```

The run must store exactly one prediction per validation incident. The evidence exporter reloads the stored raw predictions, recomputes metrics through the Phase 5 evaluator, verifies result identity, checks the exact validation incident IDs, and seals the JSON with `evidence_sha256`.

A retry with the same completed experiment/run identity does **not** rerun inference: EvalForge first reloads and recomputes the completed run and returns it only if the scientific identity and metrics still match.

## 5. Manually inspect validation predictions

Copy `evidence/phase-06/validation-review.template.json` to `evidence/phase-06/validation-review.json`. Inspect every prediction in `validation-run.json`; populate the real run ID, scientific hash, all reviewed incident IDs, reviewer, timestamp, and notes. Set the three pass booleans to true only if inspection actually confirms:

- the output/schema contract is functioning,
- predicted labels stay inside the frozen taxonomy,
- no prompt or parser defect was observed that should be repaired before the locked test.

Validation outcomes may be used to repair technical/prompt defects. Locked-test outcomes may not be used to tune the study.

Check readiness without modifying the protocol:

```bash
python scripts/freeze_phase6_protocol.py \
  --evidence evidence/phase-06/validation-run.json \
  --review evidence/phase-06/validation-review.json \
  --gpu-host-evidence evidence/phase-06/validation-gpu-host.json
```

Only after that prints `PHASE06_VALIDATION_FREEZE_READINESS=PASS`, freeze and authorize:

```bash
python scripts/freeze_phase6_protocol.py \
  --evidence evidence/phase-06/validation-run.json \
  --review evidence/phase-06/validation-review.json \
  --gpu-host-evidence evidence/phase-06/validation-gpu-host.json \
  --apply
```

The freeze record binds the accepted validation run to the sealed GPU evidence and its stable environment fingerprint. Commit the validation evidence, GPU-host evidence, review, generated `protocol-freeze.json`, and frozen protocol before executing the locked test. The scientific configuration hash must remain unchanged by the freeze.

## 6. Run the locked test once

On the locked-test host, capture host evidence again:

```bash
python scripts/capture_phase6_gpu_host.py \
  --output evidence/phase-06/test-gpu-host.json
```

Then run the same baseline with `--split test`, a distinct experiment/run ID, and `--gpu-host-evidence evidence/phase-06/test-gpu-host.json`. The runner refuses the locked test unless the stable GPU/software environment fingerprint exactly matches the validation fingerprint preserved in `protocol-freeze.json`. Use the frozen commit SHA as `--git-commit`.

Do not rerun the locked test because its score is disappointing or because a prompt/config change looks attractive. A rerun is valid only after documenting a non-model technical invalidation such as corrupted data, failed persistence, or incomplete inference.

## 7. Preserve final evidence

The final Phase 06 handoff must retain the real test experiment ID/config hash, scientific hash, W&B run/artifact references, exact stored prediction count versus test count, raw prediction export, cost records, metric recomputation proof, validation and test GPU-host evidence, the frozen environment fingerprint, and cumulative Phase 1–6 + clean-environment CI results. No dashboard/README claim may be substituted for those artifacts.

## 8. Pass the machine-checkable completion gates

At every repository transition, run:

```bash
python scripts/check_phase6_contract.py
```

It must report one internally consistent stage: `pre_validation`, `validation_ready_to_freeze`, `frozen_waiting_test`, or `complete`. This contract check is already part of `make verify-all`.

Only after the real locked-test evidence is committed should this pass:

```bash
python scripts/check_phase6_exit.py
```

The hard-exit checker requires exact held-out prediction coverage with no duplicates, raw-prediction metric recomputation, zero technical inference failures, one positive versioned cost row per prediction, frozen/recoverable scientific identity, validation/test GPU-software fingerprint equality, and configured W&B run/artifact references for the locked test.

When the repository reaches `complete`, deliberately update the Phase 06 completion-state regression test from the pre-validation expectation, add `phase6-exit` to the cumulative `verify-all` dependency list, then rerun the full Phase 1–6 gate and fresh no-cache Compose smoke. Do not mark Phase 06 complete before those final green results are preserved.
