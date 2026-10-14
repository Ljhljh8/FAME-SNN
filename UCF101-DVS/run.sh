#!/usr/bin/env bash
# Launch the UCF101-DVS FAME_SNN job on both hosts (8 GPUs per host, 16 GPUs total).
# Start this script concurrently on both machines:
#   node 0: MASTER_ADDR=<node-0-address> bash run.sh 0
#   node 1: MASTER_ADDR=<node-0-address> bash run.sh 1
# Optional overrides, which must be identical on both hosts:
#   EXPERIMENT=name PER_GPU_BATCH=32 ./run.sh <node_rank>
# Set DATA_DIR only when deliberately using raw train/ and test/ event files.
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <node_rank: 0|1> [extra train.py arguments...]" >&2
    exit 64
fi

NODE_RANK="$1"
shift
case "$NODE_RANK" in
    0|1) ;;
    *) echo "ERROR: node_rank must be 0 or 1, got: $NODE_RANK" >&2; exit 64 ;;
esac
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the address of node 0 reachable from both nodes}"
HOST="$(hostname -s)"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-$PROJECT_DIR/UCF101DVS.yml}"
# Raw data is optional: the transferred HDF5 frame caches are the default input.
DATA_DIR="${DATA_DIR:-}"
CACHE_TIME_STEP="${CACHE_TIME_STEP:-8}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
EXPERIMENT="${EXPERIMENT:-ucf101dvs_ddp16}"
MASTER_PORT="${MASTER_PORT:-29517}"
NNODES=2
GPUS_PER_NODE=8
PER_GPU_BATCH="${PER_GPU_BATCH:-32}"
VAL_PER_GPU_BATCH="${VAL_PER_GPU_BATCH:-32}"
WORKERS="${WORKERS:-8}"

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

command -v "$PYTHON" >/dev/null || fail "Python executable is missing: $PYTHON"
[[ -f "$CONFIG" ]] || fail "Config is missing: $CONFIG"
# train.py creates its rotating log before it parses arguments or initializes DDP.
mkdir -p "$PROJECT_DIR/logs"
if [[ -n "$DATA_DIR" ]]; then
    [[ -d "$DATA_DIR/train" ]] || fail "Raw training data is missing: $DATA_DIR/train"
    [[ -d "$DATA_DIR/test" ]] || fail "Raw test data is missing: $DATA_DIR/test"
    DATA_MODE="raw_events"
else
    [[ -d "$CACHE_ROOT/train_cache_$CACHE_TIME_STEP" ]] || fail "Cached training frames are missing"
    [[ -d "$CACHE_ROOT/test_cache_$CACHE_TIME_STEP" ]] || fail "Cached test frames are missing"
    DATA_MODE="cached_frames"
fi

"$PYTHON" - <<'PY'
import torch
if torch.__version__ != '2.0.0+cu118':
    raise SystemExit('unexpected torch version: ' + torch.__version__)
if torch.version.cuda != '11.8':
    raise SystemExit('unexpected CUDA runtime: ' + str(torch.version.cuda))
if torch.cuda.device_count() != 8:
    raise SystemExit('expected exactly 8 visible GPUs, got ' + str(torch.cuda.device_count()))
print('ENVIRONMENT_OK torch=%s cuda=%s gpus=%d' % (torch.__version__, torch.version.cuda, torch.cuda.device_count()))
PY

# Set NCCL_SOCKET_IFNAME externally if a specific network interface is needed.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ "$NODE_RANK" == 0 ]]; then
    "$PYTHON" - "$MASTER_ADDR" "$MASTER_PORT" <<'PY'
import socket
import sys
addr, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((addr, port))
finally:
    s.close()
print('RENDEZVOUS_PORT_OK %s:%d' % (addr, port))
PY
fi

GLOBAL_BATCH=$((NNODES * GPUS_PER_NODE * PER_GPU_BATCH))
printf 'LAUNCH host=%s node_rank=%s nnodes=%s gpus_per_node=%s global_batch=%s\n' \
    "$HOST" "$NODE_RANK" "$NNODES" "$GPUS_PER_NODE" "$GLOBAL_BATCH"
printf 'DATA_MODE=%s DATA_DIR=%s CONFIG=%s OUTPUT_DIR=%s EXPERIMENT=%s\n' \
    "$DATA_MODE" "${DATA_DIR:-<cache-only>}" "$CONFIG" "$OUTPUT_DIR" "$EXPERIMENT"
printf 'NCCL_SOCKET_IFNAME=%s MASTER=%s:%s\n' \
    "${NCCL_SOCKET_IFNAME:-<auto>}" "$MASTER_ADDR" "$MASTER_PORT"

cd "$PROJECT_DIR"
exec "$PYTHON" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    -c "$CONFIG" \
    -data-dir "$DATA_DIR" \
    --cache-root "$CACHE_ROOT" \
    --batch-size "$PER_GPU_BATCH" \
    --val-batch-size "$VAL_PER_GPU_BATCH" \
    --workers "$WORKERS" \
    --output "$OUTPUT_DIR" \
    --experiment "$EXPERIMENT" \
    --amp \
    "$@"
