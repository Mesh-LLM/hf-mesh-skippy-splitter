# Mesh-LLM Layer Splitter (HF Jobs)

Split large GGUF models into per-layer files on Hugging Face compute.
No local download needed — the source model is mounted directly from HF storage.

Each layer becomes a separate GGUF file that nodes in a [mesh-llm](https://github.com/Mesh-LLM/mesh-llm)
cluster can download independently. A node running layers 70–94 of a 235B model
only downloads ~35 GB instead of the full 134 GB.

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│  HF Job (cpu-xl, 16 vCPU, no GPU needed)                       │
│                                                                 │
│  1. Mount source GGUF repo as read-only volume (instant)        │
│  2. Build llama-model-slice from mesh-llm source (~5 min)       │
│  3. Split GGUF into per-layer files (~1 GB/min throughput)      │
│  4. Validate tensor coverage                                    │
│  5. Upload layer package to target HF repo                      │
└─────────────────────────────────────────────────────────────────┘
```

The splitter (`llama-model-slice`) is built from the
[mesh-llm](https://github.com/Mesh-LLM/mesh-llm) repo which includes it as a
workspace crate (see [PR #422](https://github.com/Mesh-LLM/mesh-llm/pull/422)).

## Quick start

```bash
# 1. Set your HF token (needs write access to target org)
export HF_TOKEN="hf_..."

# 2. One-time: upload the mesh-llm source tarball to a bucket
./scripts/prepare-source.sh

# 3. Split a model
./scripts/run-split-job.sh \
  unsloth/Qwen3-235B-A22B-GGUF \
  "UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf" \
  meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
```

That's it. The job runs on HF infrastructure (~$0.75 for a 134 GB model),
and the result is published to the target repo.

## Prerequisites

- [HF CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (`pip install huggingface_hub[cli]`)
- `HF_TOKEN` with write access to the target org/repo
- The mesh-llm source tarball uploaded to a HF bucket (one-time setup)

## Usage

```
./scripts/run-split-job.sh <source_repo> <source_file> <target_repo> [model_id]
```

| Argument | Description | Example |
|---|---|---|
| `source_repo` | HF repo with the source GGUF | `unsloth/Qwen3-235B-A22B-GGUF` |
| `source_file` | Path to first shard within the repo | `UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf` |
| `target_repo` | HF repo to publish layer package to | `meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers` |
| `model_id` | (optional) Model ID for manifest | auto-derived from source |

For sharded GGUFs (multiple files), point `source_file` at the **first shard**
(`-00001-of-NNNNN.gguf`). The splitter finds siblings automatically.

## Examples

### Qwen3-235B-A22B (134 GB, 3 shards → 94 layers, ~$0.75)

```bash
./scripts/run-split-job.sh \
  unsloth/Qwen3-235B-A22B-GGUF \
  "UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf" \
  meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
```

**Already done** — result at [meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers](https://huggingface.co/meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers).
Tested across 2 machines (Mac Studio M3 Ultra 256 GB + M4 Max 64 GB) at **16 tok/s**.

### Qwen3.6-72B (40 GB, single file → 80 layers, ~$0.15)

```bash
./scripts/run-split-job.sh \
  unsloth/Qwen3.6-72B-GGUF \
  "Q4_K_M/Qwen3.6-72B-Q4_K_M.gguf" \
  meshllm/Qwen3.6-72B-Q4_K_M-layers
```

### DeepSeek-V3 (350 GB, 5 shards → 61 layers, ~$1.50)

```bash
./scripts/run-split-job.sh \
  unsloth/DeepSeek-V3-0324-GGUF \
  "Q4_K_M/DeepSeek-V3-0324-Q4_K_M-00001-of-00005.gguf" \
  meshllm/DeepSeek-V3-Q4_K_M-layers
```

### LLaMA 3.1 405B (200 GB, 4 shards → 126 layers, ~$1.00)

```bash
./scripts/run-split-job.sh \
  unsloth/Llama-3.1-405B-GGUF \
  "Q4_K_M/Llama-3.1-405B-Q4_K_M-00001-of-00004.gguf" \
  meshllm/Llama-3.1-405B-Q4_K_M-layers
```

### Qwen3-Coder-480B (260 GB → 96 layers, ~$1.25)

```bash
./scripts/run-split-job.sh \
  unsloth/Qwen3-Coder-480B-A35B-GGUF \
  "Q4_K_M/Qwen3-Coder-480B-A35B-Q4_K_M-00001-of-00004.gguf" \
  meshllm/Qwen3-Coder-480B-Q4_K_M-layers
```

## Cost

| Model size | Split time | Upload time | Total | Cost (cpu-xl @ $1/hr) |
|---:|---:|---:|---:|---:|
| 40 GB | ~5 min | ~3 min | ~10 min | ~$0.15 |
| 134 GB | ~15 min | ~10 min | ~45 min | ~$0.75 |
| 200 GB | ~20 min | ~15 min | ~60 min | ~$1.00 |
| 350 GB | ~35 min | ~25 min | ~90 min | ~$1.50 |

Build time (~5 min) is included in the first run. Subsequent runs reuse the
cached binary if the bucket source hasn't changed.

## Output format

```
target-repo/
├── model-package.json          # Manifest (layer count, checksums, provenance)
├── shared/
│   ├── metadata.gguf           # Model config (vocab size, hidden dim, etc.)
│   ├── embeddings.gguf         # Token embeddings
│   └── output.gguf             # Output head + final norm
└── layers/
    ├── layer-000.gguf          # Layer 0 (attention + FFN/MoE experts)
    ├── layer-001.gguf
    ├── ...
    └── layer-093.gguf
```

Each layer file contains all tensors for that layer (attention weights, FFN/expert
weights, norms). For MoE models, each layer includes all experts — they're large
(~1.5 GB for Qwen3-235B at Q4) but self-contained.

## Using the layer package

Once published, any mesh-llm node can serve the model:

```bash
# Single node (downloads all layers)
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers

# Multi-node (each downloads only its assigned layers)
# Node A:
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers --split
# Node B:
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers --split --join <token>
```

## Supported model families

The splitter works with any GGUF model. The stage runtime (for multi-node inference)
currently supports these architectures:

| Family | Architecture | Example models |
|---|---|---|
| Qwen 2/3 | `QWEN2`, `QWEN3` | Qwen3-30B, Qwen3.6-72B |
| Qwen MoE | `QWEN35MOE`, `QWEN3MOE` | Qwen3-235B, Qwen3.6-397B, Qwen3-Coder-480B |
| LLaMA | `LLAMA` | LLaMA 3.1 405B |
| DeepSeek | `DEEPSEEK2` | DeepSeek-V3 |
| Gemma | `GEMMA2`, `GEMMA3`, `GEMMA4` | Gemma 2/3/4 |
| GLM | `GLM4` | GLM-4, GLM-5 |
| Falcon | `FALCON_H1` | Falcon-H1 |
| MiniMax | `MINIMAX_M2` | MiniMax-M2.7 |
| OLMo | `OLMO` | OLMo |

## How the splitter is built

The `llama-model-slice` binary is built from the mesh-llm workspace. It links
against a patched llama.cpp that understands GGUF tensor structure and can
decompose a model into per-layer files with correct metadata.

The HF Job builds it from source inside the container:
1. Extracts the mesh-llm source tarball from the bucket
2. Runs `scripts/prepare-llama.sh` to clone + patch llama.cpp
3. Runs `scripts/build-llama.sh` to compile the C++ static libraries
4. Runs `cargo build --release -p llama-model-slice` to build the Rust binary

Total build time: ~5 minutes on cpu-xl (16 vCPU).
