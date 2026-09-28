#!/usr/bin/env bash
set -Eeuo pipefail

# Prepared launcher only. Invoke explicitly with either `official` or `bialign`.
ARM="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
case "$ARM" in
  official) CONFIG="exps/RSIAT_BiAlign_official.json" ;;
  bialign)  CONFIG="exps/RSIAT_BiAlign.json" ;;
  *) echo "Usage: PYTHON_BIN=/path/to/python $0 {official|bialign}" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
# [IMPLEMENTATION DEVIATION] allocator setting is runtime-only and shared by arms.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$PYTHON_BIN" tools/verify_bialign_configs.py
exec "$PYTHON_BIN" -u main.py --config "$CONFIG"
