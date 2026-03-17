#!/usr/bin/env bash
# cd /mnt/data/conan/flow_matching
# ./run_cifar10.sh
set -e

# 一键多卡训练 CIFAR10 的脚本

# 1. 基本配置（如需修改，改下面这些变量即可）
PROJECT_DIR="/mnt/data/conan/flow_matching"
ENV_DIR="$PROJECT_DIR/.venv"
DATA_PATH="$PROJECT_DIR/data/image_generation"

# 是否只做快速测试（跑极少量 step / epoch）
TEST_RUN=false

# 多卡“保守模式”：禁用 SHM/P2P/IB，避免多人共享机器下 NCCL 卡死
SAFE_DDP=true

# 为本次实验生成一个时间戳目录，避免多次运行互相覆盖
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ "$TEST_RUN" = true ]; then
  OUTPUT_ROOT="$PROJECT_DIR/outputs/cifar10_8gpu/test_$TIMESTAMP"
else
  OUTPUT_ROOT="$PROJECT_DIR/outputs/cifar10_8gpu/$TIMESTAMP"
fi

# 只使用指定的 GPU
# 共享机器上常见 0/7 被他人占用或状态异常，默认先避开它们提高多卡启动成功率
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

if [ "$TEST_RUN" = true ]; then
  # 测试模式：单卡、小 batch、很少 epoch
  NGPUS=1
  BATCH_SIZE=64
  EPOCHS=2           # 跑很少的 epoch，看流程是否正常
  EVAL_FREQ=1        # 每个 epoch 都评估一次
  FID_SAMPLES=512    # 少量样本即可
else
  # 正式训练配置
  # 使用的 GPU 数量（单机，需要与上面的 CUDA_VISIBLE_DEVICES 个数一致）
  NGPUS=8

  # 每张 GPU 的 batch size，有效 batch = BATCH_SIZE * NGPUS * ACCUM_ITER
  BATCH_SIZE=128

  # 训练轮数和评估频率
  EPOCHS=800    # 总 epoch 数
  EVAL_FREQ=50      # 每多少个 epoch 评估并生成一次图像
  FID_SAMPLES=10000  # 评估时用于 FID 的样本数
fi

# 2. 进入工程目录并激活虚拟环境
cd "$PROJECT_DIR"
source "$ENV_DIR/bin/activate"

# 3. 设置 TORCH_HOME 到项目内缓存目录，避免使用系统级缓存
export TORCH_HOME="$PROJECT_DIR/.cache/torch"

# 4. 创建数据目录（若 CIFAR10 未下载，torchvision 会自动下载到此目录）
mkdir -p "$DATA_PATH"

echo "Running training with:"
echo "  GPUs        : $NGPUS"
echo "  Batch/GPU   : $BATCH_SIZE"
echo "  Epochs      : $EPOCHS"
echo "  Eval freq   : $EVAL_FREQ"
echo "  Output root : $OUTPUT_ROOT"
echo "  Test run    : $TEST_RUN"
echo "  Safe DDP    : $SAFE_DDP"
echo

# 5. 启动多卡训练
EXTRA_ARGS=()
if [ "$TEST_RUN" = true ]; then
  EXTRA_ARGS+=(--test_run)
fi

# --- 核心优化：确保环境纯净 ---
if [ "$SAFE_DDP" = true ]; then
  export NCCL_SHM_DISABLE=1   # 禁用 /dev/shm 通信（多人共享机器最常见冲突点）
  export NCCL_P2P_DISABLE=1   # 禁用 P2P/NVLink
  export NCCL_IB_DISABLE=1    # 禁用 IB/RDMA
  # 不限制网卡名（很多机器网卡命名/策略各异），让 NCCL 自己选可用接口
  unset NCCL_SOCKET_IFNAME
  # 强制 IPv4，避免 IPv6/解析导致的握手异常
  export NCCL_SOCKET_FAMILY=AF_INET
  export GLOO_SOCKET_FAMILY=AF_INET
  # 打开更细的 NCCL 初始化/网络日志，方便定位 Abort 原因
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,NET
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
fi

# 自动生成一个 20000-30000 之间的随机端口，彻底避开冲突
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)

echo "Using Random Master Port: $MASTER_PORT"

# 启动训练
# 增加 --master_port 参数确保 torchrun 使用我们定义的随机端口
torchrun \
  --nproc_per_node="$NGPUS" \
  --master_port="$MASTER_PORT" \
  examples/image/train.py \
  --dataset cifar10 \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --eval_frequency "$EVAL_FREQ" \
  --fid_samples "$FID_SAMPLES" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_ROOT" \
  --decay_lr \
  "${EXTRA_ARGS[@]}"