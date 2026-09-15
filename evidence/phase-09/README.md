# Phase 09 evidence

Repository-side Phase 09 evidence starts here. Do not add synthetic W&B references, GPU metadata, adapter checksums, or training results.

The preferred connected path is `training/scripts/complete_phase9.py`. It generates the completion records from executable checks and only copies them into this directory after training, reload, resume, export, and the Phase 09 hard-exit check succeed.

Expected completion evidence after a real connected GPU run:

- `training-run.json` — exact training-config hash, dataset/prepared-data checksums, lineage hash, frozen model identity, git commit, CUDA/software metadata, selected checkpoint, adapter checksum, W&B run reference, and W&B adapter-artifact reference;
- `adapter-reload.json` — validation-only schema-compatible inference through the canonical `FINETUNED` adapter pipeline;
- `resume-training-run.json` — a second CUDA/W&B training record resumed from a checkpoint that appears in the source run;
- `resume.json` — source/resumed run linkage, checkpoint identity, resumed adapter checksum, and checksum of `resume-training-run.json`;
- `adapter-export.json` — checksum-preserving adapter-copy proof and export-manifest identity. Phase 09 evidence must not request the Phase 10 public Hub release.

The three supporting PASS records contain the SHA-256 of the exact `training-run.json`. The hard-exit gate therefore rejects stale reload/resume/export evidence if the primary training record changes. The resume gate separately hashes and validates `resume-training-run.json` as a real CUDA + W&B record.

Until those external facts exist, the repository may prove the training-system contracts but Phase 09 remains pre-completion. A CPU smoke adapter, fabricated JSON, or a decreasing loss curve is not completion evidence.
