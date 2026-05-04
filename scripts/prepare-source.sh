#!/bin/bash
set -euo pipefail

# Prepare and upload the mesh-llm source tarball to a HF bucket.
# Run this once, or whenever the splitter code changes.
#
# Prerequisites:
#   - HF_TOKEN set with write access
#   - mesh-llm repo cloned locally
#
# Usage:
#   ./scripts/prepare-source.sh [/path/to/mesh-llm]

MESH_LLM_DIR="${1:-$(dirname "$0")/../../mesh-llm}"
BUCKET="meshllm/layer-split-output"
TARBALL="/tmp/mesh-llm-splitter-source.tar.gz"

if [ ! -f "$MESH_LLM_DIR/Cargo.toml" ]; then
    echo "Error: mesh-llm repo not found at $MESH_LLM_DIR"
    echo "Usage: $0 [/path/to/mesh-llm]"
    exit 1
fi

echo "=== Creating source tarball from $MESH_LLM_DIR ==="
cd "$MESH_LLM_DIR"

# Include only what's needed to build llama-model-slice
tar czf "$TARBALL" \
    --exclude='.deps' \
    --exclude='target' \
    --exclude='.git' \
    --exclude='docs/family' \
    --exclude='docs/skippy/family' \
    --exclude='ui-preview' \
    --exclude='sdk' \
    --exclude='eval' \
    --exclude='*.log' \
    .

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  Tarball: $TARBALL ($SIZE)"

echo "=== Uploading to bucket $BUCKET ==="
hf bucket upload "$BUCKET" "$TARBALL" source.tar.gz

echo "=== Done ==="
echo "Source tarball uploaded. You can now run split jobs."
