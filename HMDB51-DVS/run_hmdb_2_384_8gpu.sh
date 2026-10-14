#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python3}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_DIR:-$PROJECT_DIR/data}}"
DATA_DIR="${DATA_DIR:-$CACHE_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
command -v "$PYTHON" >/dev/null || { echo "Python executable is missing: $PYTHON" >&2; exit 2; }
[[ -d "$CACHE_ROOT/train_cache_16" && -d "$CACHE_ROOT/test_cache_16" ]] || {
  echo "Required HMDB temporal caches are missing under: $CACHE_ROOT" >&2; exit 2;
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Batch size 8 per GPU: global batch size 64, no gradient accumulation.
exec "$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train.py -c HMDB51DVS.yml \
  --dim 384 --time-step 16 \
  --batch-size 8 --val-batch-size 8 \
  --cache-root "$CACHE_ROOT" -data-dir "$DATA_DIR" \
  --seed 42 --resume '' --native-amp --log-interval 10 \
  --output "$OUTPUT_DIR" \
  --experiment hmdb_d2_dim384_T16_bs8_seed42
