# ADR-005 — Primary Base Model for the Four-Way Study

**Status:** Accepted, pending hard smoke evidence

## Context
The central experiment is invalid if different base-model families/revisions are compared across arms.

## Decision
Freeze `mistralai/Mistral-7B-Instruct-v0.3` at revision `e8737b84b4470b28db3a0be719b362b1bd39a14d` for `ZERO_SHOT`, `RAG`, `FINETUNED`, and `COMBINED` primary runs.

The selection rule and candidate comparison are documented in `docs/model-selection.md`.

## Required hard gate
The exact revision must load in the intended GPU development/training environment and return strict JSON with a canonical `root_cause_code`. If that fails because of access, license, output-contract, or hardware incompatibility, Phase 01 must reopen this ADR, select another compatible 7B–8B instruction model, freeze a new exact revision, and rerun all Phase 01 checks.

## Consequences
No later primary run may silently update to `main` or switch model family.
