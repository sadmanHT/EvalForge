# ADR-005 — Frozen Primary Model

**Status:** Accepted, pending connected weight-load smoke

Primary model: `mistralai/Mistral-7B-Instruct-v0.3`

Frozen revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`

License: Apache-2.0.

The Phase 01 plan prefers the Mistral 7B Instruct family when compatible. This revision is open-weight, instruction-tuned, and suitable for the common four-arm comparison. Every primary arm must use this exact base ID/revision. A future model change requires a new study/model version rather than silently changing this study.

The full weight-load/generation smoke is a hard Phase 01 gate and is performed by `scripts/model_smoke.py` on suitable hardware.
