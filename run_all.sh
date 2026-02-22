#!/bin/bash
# Run all 4 correctness experiments.
#
# Each experiment is a separate process because VLLM_BATCH_INVARIANT
# is read once at import time and cannot be toggled within a process.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4 bash run_all.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_all.sh 2>&1 | tee run_all.log

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is not set."
    echo "Usage: CUDA_VISIBLE_DEVICES=4 bash run_all.sh"
    exit 1
fi

export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Correctness Test Suite"
echo "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  $(date)"
echo "============================================================"

echo ""
echo ">>> [1/4] Batch invariance ON"
echo "============================================================"
python test_batch_invariance.py --batch-invariant

echo ""
echo ">>> [2/4] Batch invariance OFF"
echo "============================================================"
python test_batch_invariance.py --no-batch-invariant

echo ""
echo ">>> [3/4] Spec decode + batch invariance ON"
echo "============================================================"
python test_spec_decode.py --batch-invariant

echo ""
echo ">>> [4/4] Spec decode + batch invariance OFF"
echo "============================================================"
python test_spec_decode.py --no-batch-invariant

echo ""
echo "============================================================"
echo "  All 4 experiments completed.  $(date)"
echo "============================================================"
