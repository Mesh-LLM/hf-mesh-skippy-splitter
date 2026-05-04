#!/bin/bash
set -euo pipefail

# This script runs inside an HF Job container.
# It clones mesh-llm, builds the splitter, splits the model, validates, and publishes.
#
# Environment variables (set by run-split-job.sh):
#   SOURCE_REPO, SOURCE_FILE, TARGET_REPO, MODEL_ID, SOURCE_REVISION, HF_TOKEN
#   MESH_LLM_REF — git ref to build from (default: jd/feat/skippy, then main once merged)
#
# Volumes:
#   /source  — source GGUF repo (read-only mount)

MESH_LLM_REF="${MESH_LLM_REF:-jd/feat/skippy}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Layer Package Split Job                                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Source: ${SOURCE_REPO}/${SOURCE_FILE}"
echo "║  Target: ${TARGET_REPO}"
echo "║  Model:  ${MODEL_ID}"
echo "║  Build:  mesh-llm @ ${MESH_LLM_REF}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Build tools ──────────────────────────────────────────────────────────
echo "=== [1/7] Installing build dependencies ==="
apt-get update -qq && apt-get install -y -qq \
    cmake git curl build-essential pkg-config libssl-dev \
    python3-pip python3-venv > /dev/null 2>&1

echo "=== [2/7] Installing Rust ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
source /root/.cargo/env

echo "=== [3/7] Cloning mesh-llm and building llama-model-slice ==="
git clone --depth 1 --branch "$MESH_LLM_REF" \
    https://github.com/Mesh-LLM/mesh-llm.git /tmp/build
cd /tmp/build

# Full clone needed for git-am patches in prepare-llama
sed -i 's/--filter=blob:none //' scripts/prepare-llama.sh
scripts/prepare-llama.sh pinned > /dev/null 2>&1
scripts/build-llama.sh > /dev/null 2>&1

# Build the splitter binary
SKIPPY_LLAMA_BUILD_DIR=.deps/llama.cpp/build-stage-abi-static \
    cargo build --release -p llama-model-slice 2>&1 | tail -3
SLICER=/tmp/build/target/release/llama-model-slice
echo "  ✓ Built: $SLICER"

# ─── Split ────────────────────────────────────────────────────────────────
echo ""
echo "=== [4/7] Splitting model ==="
SOURCE_PATH="/source/${SOURCE_FILE}"
PACKAGE_DIR="/tmp/package"
mkdir -p "$PACKAGE_DIR"

if [ ! -f "$SOURCE_PATH" ]; then
    echo "ERROR: Source file not found at $SOURCE_PATH"
    echo ""
    echo "Available GGUF files in /source:"
    find /source -name "*.gguf" -type f | sort | head -30
    exit 1
fi

echo "  Source: $SOURCE_PATH ($(du -h "$SOURCE_PATH" | cut -f1))"
time $SLICER write-package "$SOURCE_PATH" \
    --out-dir "$PACKAGE_DIR" \
    --model-id "$MODEL_ID" \
    --source-repo "$SOURCE_REPO" \
    --source-revision "${SOURCE_REVISION:-main}" \
    --source-file "$SOURCE_FILE"

LAYER_COUNT=$(ls "$PACKAGE_DIR"/layers/ | wc -l)
TOTAL_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
echo "  ✓ Split into $LAYER_COUNT layers ($TOTAL_SIZE total)"

# ─── Validate ─────────────────────────────────────────────────────────────
echo ""
echo "=== [5/7] Validating package ==="
time $SLICER validate-package "$SOURCE_PATH" "$PACKAGE_DIR"
echo "  ✓ Validation passed — all tensors accounted for"

# ─── Publish ──────────────────────────────────────────────────────────────
echo ""
echo "=== [6/7] Publishing to HuggingFace ==="
python3 -m venv /tmp/venv > /dev/null
/tmp/venv/bin/pip install -q huggingface_hub

/tmp/venv/bin/python3 << PYTHON
from huggingface_hub import HfApi
import os, json

api = HfApi(token=os.environ['HF_TOKEN'])
target_repo = os.environ['TARGET_REPO']
source_repo = os.environ['SOURCE_REPO']
model_id = os.environ.get('MODEL_ID', '')

# Create repo (idempotent)
api.create_repo(target_repo, exist_ok=True)

# Upload the entire package
api.upload_folder(
    repo_id=target_repo,
    folder_path='/tmp/package',
    commit_message=f'Layer package from {source_repo} ({model_id})',
)

# Print summary
manifest = json.load(open('/tmp/package/model-package.json'))
print(f'  ✓ Published: https://huggingface.co/{target_repo}')
print(f'    Model:  {manifest["model_id"]}')
print(f'    Layers: {manifest["layer_count"]}')
print(f'    Schema: {manifest["schema_version"]}')
PYTHON

# ─── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== [7/7] Done ==="
echo ""
echo "  Published:  https://huggingface.co/${TARGET_REPO}"
echo "  Layers:     ${LAYER_COUNT}"
echo "  Total size: ${TOTAL_SIZE}"
echo ""
echo "  Use with mesh-llm:"
echo "    mesh-llm serve --model hf://${TARGET_REPO} --split"
