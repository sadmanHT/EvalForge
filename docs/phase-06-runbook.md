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

## 2. Prepare a Linux CUDA GPU host

Use a GPU environment capable of loading the frozen model runtime. Phase 01 proved the compatibility path on Tesla T4 hardware. Install the repository backend lock plus the GPU model stack used by the real host. Record the exact installed versions and `nvidia-smi` output with the run evidence. Do not substitute a tiny or mock model for the real validation/test runs.

The canonical worker/CLI needs PostgreSQL because PostgreSQL is the experiment system of record. On a GPU host with Docker available, a minimal database can be started with the same image used by EvalForge:

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

Install the repository package and the real inference stack in the same Python environment. The model runtime will fail explicitly if torch/transformers/bitsandbytes support is unavailable.

## 3. Configure tracking and cost evidence

For portfolio/public experiment tracking, configure W&B before the real run and install the `wandb` package in the GPU environment:

```bash
export WANDB_PROJECT=<real-project-name>
export WANDB_ENTITY=<real-entity-if-used>
export WANDB_API_KEY=<secret-from-environment>
```

Never commit the API key. The resulting W&B run/artifact references are copied into PostgreSQL and then into the deterministic Phase 06 evidence export.

Use the GPU provider's actual rate for `--gpu-hour-usd`. Give that rate source a stable `--cost-rate-snapshot-version`; do not invent a price. The runner computes per-query marginal inference cost as active inference wall time multiplied by the supplied GPU hourly rate and persists a versioned `cost_records` row for each priced prediction.

## 4. Run the real validation split

Capture the exact repository commit and a factual hardware/runtime descriptor. Then run:

```bash
python evals/runner.py \
  --split validation \
  --git-commit "$(git rev-parse HEAD)" \
  --hardware-runtime-descriptor "<actual GPU/runtime description>" \
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
  --review evidence/phase-06/validation-review.json
```

Only after that prints `PHASE06_VALIDATION_FREEZE_READINESS=PASS`, freeze and authorize:

```bash
python scripts/freeze_phase6_protocol.py \
  --evidence evidence/phase-06/validation-run.json \
  --review evidence/phase-06/validation-review.json \
  --apply
```

Commit the validation evidence, review, generated `protocol-freeze.json`, and frozen protocol before executing the locked test. The scientific configuration hash must remain unchanged by the freeze.

## 6. Run the locked test once

After the frozen/authorized commit is on the Phase 06 branch, run the exact same real-model command with `--split test`, a distinct experiment/run ID, and `--evidence-output evidence/phase-06/test-run.json`. Use the frozen commit SHA as `--git-commit`.

Do not rerun the locked test because its score is disappointing or because a prompt/config change looks attractive. A rerun is valid only after documenting a non-model technical invalidation such as corrupted data, failed persistence, or incomplete inference.

## 7. Preserve final evidence

The final Phase 06 handoff must retain the real test experiment ID/config hash, scientific hash, W&B reference when configured, exact stored prediction count versus test count, raw prediction export, cost records, metric recomputation proof, hardware/runtime evidence, and cumulative Phase 1–6 + clean-environment CI results. No dashboard/README claim may be substituted for those artifacts.
