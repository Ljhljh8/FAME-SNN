#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Batch size 8 per GPU: global batch size 64, no gradient accumulation.
exec /home/think/anaconda3/envs/snn_ql_1/bin/python \
  -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py \
  -c HMDB51DVS.yml \
  --dim 384 \
  --time-step 16 \
  --batch-size 8 \
  --val-batch-size 8 \
  --cache-root /home/think/ljh_datasets/HMDBDVS \
  -data-dir /home/think/ljh_datasets/HMDBDVS \
  --seed 42 \
  --resume '' \
  --native-amp \
  --log-interval 10 \
  --output /media/think/data/ljh/FAME_SNN_NEW/outputs/HMDB51-DVS \
  --experiment hmdb_d2_dim384_T16_bs8_seed42
