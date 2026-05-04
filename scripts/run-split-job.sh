#!/bin/bash
set -euo pipefail

# Submit an HF Job to split a GGUF model into per-layer files.
#
# Usage:
#   ./scripts/run-split-job.sh <source_repo> <source_file> <target_repo> [model_id]
#
# The source GGUF repo is mounted read-only (no download needed).
# The job clones mesh-llm from GitHub and builds the splitter inside the container.
#
# Examples:
#   ./scripts/run-split-job.sh \
#     unsloth/Qwen3-235B-A22B-GGUF \
#     "UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf" \
#     meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers

SOURCE_REPO="${1:?Usage: $0 <source_repo> <source_file> <target_repo> [model_id]}"
SOURCE_FILE="${2:?Usage: $0 <source_repo> <source_file> <target_repo> [model_id]}"
TARGET_REPO="${3:?Usage: $0 <source_repo> <source_file> <target_repo> [model_id]}"

# Auto-derive model_id from source repo + quant dir
QUANT_DIR="$(dirname "$SOURCE_FILE")"
MODEL_ID="${4:-${SOURCE_REPO}:${QUANT_DIR}}"

# Git ref for mesh-llm (PR #422 branch, switch to "main" once merged)
MESH_LLM_REF="${MESH_LLM_REF:-jd/feat/skippy}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUCKET="meshllm/layer-split-output"

# Verify prerequisites
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN not set. Export a write-access token."
    exit 1
fi

if ! command -v hf &>/dev/null; then
    echo "Error: 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Layer Package Split Job                                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Source:  ${SOURCE_REPO}"
echo "║  File:    ${SOURCE_FILE}"
echo "║  Target:  ${TARGET_REPO}"
echo "║  ModelID: ${MODEL_ID}"
echo "║  Build:   mesh-llm @ ${MESH_LLM_REF}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Upload the split script to the bucket (the job needs to run it)
echo "Uploading job script to bucket..."
hf bucket upload "$BUCKET" "$SCRIPT_DIR/split-model-job.sh" split-model-job.sh 2>/dev/null

echo "Submitting HF Job (cpu-xl, ~\$1/hr)..."
echo ""

hf jobs run \
    --flavor cpu-xl \
    --timeout 10800 \
    --env "SOURCE_REPO=$SOURCE_REPO" \
    --env "SOURCE_FILE=$SOURCE_FILE" \
    --env "TARGET_REPO=$TARGET_REPO" \
    --env "MODEL_ID=$MODEL_ID" \
    --env "SOURCE_REVISION=main" \
    --env "MESH_LLM_REF=$MESH_LLM_REF" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --volume "$SOURCE_REPO:/source:ro" \
    --volume "$BUCKET:/bucket:ro" \
    -- bash /bucket/split-model-job.sh

echo ""
echo "Job submitted!"
echo ""
echo "  Monitor:  hf jobs logs <job_id>"
echo "  Result:   https://huggingface.co/$TARGET_REPO"
echo "  Use with: mesh-llm serve --model hf://${TARGET_REPO} --split"
