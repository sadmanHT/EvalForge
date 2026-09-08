# Primary Base-Model Selection Record

## Decision status

**Selected:** `mistralai/Mistral-7B-Instruct-v0.3`  
**Frozen revision:** `e8737b84b4470b28db3a0be719b362b1bd39a14d`  
**Study:** `evalforge-incident-diagnosis-primary-v1`

The selection is provisional only with respect to the Phase 01 hard load/inference smoke: the model/revision is frozen for the study, but Phase 01 remains incomplete until the exact revision is proven to load and produce the required structured smoke output in the intended GPU development/training environment.

## Selection rule

The primary study model must satisfy all mandatory criteria:

1. open-weight instruction model in the 7B–8B range;
2. exact immutable Hub revision can be pinned;
3. usable through Hugging Face Transformers and compatible with PEFT/LoRA workflows;
4. license/access conditions are practical for a public portfolio project;
5. supports deterministic chat/instruction inference and token-level scoring needed for the planned confidence method;
6. fits the intended single-GPU development/training path using full/bfloat16 inference where practical and QLoRA/4-bit training where needed;
7. no model-family switching across the four primary comparison arms.

Desirable criteria: non-gated access, permissive license, mature Transformers support, and sufficient context length for incident evidence.

## Candidate review

| Candidate | 7B–8B instruct | Hub revision pinning | Access/license | Transformers path | Decision |
| --- | --- | --- | --- | --- | --- |
| Mistral-7B-Instruct-v0.3 | yes | yes | Apache-2.0; public model page | official Transformers examples available | **selected** |
| Llama-3.1-8B-Instruct | yes | yes | Llama 3.1 license; gated/approval-based Hub access | Transformers supported | not selected: avoid gated-access dependency |
| Qwen2.5-7B-Instruct | yes | yes | Apache-2.0; public model page | Transformers supported | viable fallback if Mistral hard smoke fails |

## Why Mistral is selected

Mistral satisfies the required size, instruction-tuning, public-access, permissive-license, revision-pinning, and Transformers/PEFT compatibility criteria while avoiding gated model access. The exact revision is frozen in `configs/model.yaml`.

The repository hard gate intentionally treats runtime compatibility separately from paper selection. If the frozen Mistral revision cannot be loaded in the intended GPU environment or cannot satisfy the strict structured-output smoke, Phase 01 requires a new explicit model-selection decision, ADR update, new frozen revision, and rerun of all Phase 01 tests.

## Intended development/training environment contract

The primary intended environment is a Linux CUDA GPU workspace suitable for 7B-class Transformers/PEFT work. Phase 01 model smoke evidence must record:

- GPU model and VRAM
- CUDA/PyTorch/Transformers versions
- load dtype/quantization mode
- exact model revision
- smoke prompt
- raw generated text
- parsed structured output

The current lightweight ChatGPT execution container is not the intended training environment and is not used as proof of model compatibility.

## Source records checked during Phase 01

- Mistral official model card: <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>
- Frozen Mistral revision: <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/e8737b84b4470b28db3a0be719b362b1bd39a14d>
- Llama 3.1 8B Instruct model card: <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- Qwen2.5 7B Instruct model card: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
