#!/usr/bin/env bash
# 用已有 checkpoint 只跑评估、生成图片（不训练）
# 用法:
#   ./run_eval_only.sh <checkpoint.pth> [output_dir]
# 示例:
#   ./run_eval_only.sh outputs/cifar10_8gpu/20260316_235343/checkpoint-799.pth outputs/cifar10_8gpu/20260316_235343
set -e

PROJECT_DIR="/mnt/data/conan/flow_matching"
ENV_DIR="$PROJECT_DIR/.venv"
DATA_PATH="$PROJECT_DIR/data/image_generation"

CHECKPOINT="${1:?Usage: $0 <path/to/checkpoint.pth> [output_dir]}"
OUTPUT_DIR="${2:-$(dirname "$CHECKPOINT")}"

cd "$PROJECT_DIR"
source "$ENV_DIR/bin/activate"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
# torch-fidelity/torch.hub 会把 Inception 权重下载到 $TORCH_HOME/hub/checkpoints
mkdir -p "$TORCH_HOME/hub/checkpoints"

# 单卡即可；加载后 start_epoch=checkpoint.epoch+1，epochs 设大一些保证会跑一步 eval
torchrun --nproc_per_node=1 examples/image/train.py \
  --dataset cifar10 \
  --batch_size 128 \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --resume "$CHECKPOINT" \
  --eval_only \
  --epochs 10000 \
  --eval_frequency 1 \
  --fid_samples 10000 \
  --compute_fid

echo "Done. Check snapshots under: $OUTPUT_DIR/snapshots/"
