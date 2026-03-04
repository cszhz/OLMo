#!/bin/bash
################################################################################
# OLMo Neuron 训练启动脚本
#
# 使用方法:
#   bash scripts/start-training.sh [步数] [进程数] [保存间隔]
#
# 示例:
#   bash scripts/start-training.sh 100 2 20     # 100步, 2进程, 每20步保存
#   bash scripts/start-training.sh 1000 4 100   # 1000步, 4进程, 每100步保存
################################################################################

set -e  # 遇到错误立即退出

# 默认参数
MAX_STEPS=${1:-100}        # 默认100步
NPROC=${2:-2}              # 默认2进程
SAVE_INTERVAL=${3:-20}     # 默认每20步保存

# 配置文件
CONFIG="configs/tiny/OLMo-20M-neuron-optimum.yaml"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}OLMo Neuron 训练启动${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "配置:"
echo "  - 训练步数: ${MAX_STEPS}"
echo "  - 并行进程: ${NPROC}"
echo "  - 保存间隔: ${SAVE_INTERVAL}"
echo "  - 配置文件: ${CONFIG}"
echo ""

# 检查环境
echo -e "${YELLOW}检查环境...${NC}"

if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}激活虚拟环境...${NC}"
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
else
    echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
fi

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}✗ 配置文件不存在: ${CONFIG}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件存在${NC}"

# 检查训练脚本
if [ ! -f "scripts/train-optimum-neuron.py" ]; then
    echo -e "${RED}✗ 训练脚本不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 训练脚本存在${NC}"

# 检查 Neuron 设备
echo ""
echo -e "${YELLOW}检查 Neuron 设备...${NC}"
NEURON_COUNT=$(neuron-ls 2>/dev/null | grep -c "NA      | NA" || echo 0)
if [ "$NEURON_COUNT" -lt "$NPROC" ]; then
    echo -e "${YELLOW}⚠ 可用 Neuron 设备: ${NEURON_COUNT}, 需要: ${NPROC}${NC}"
    echo -e "${YELLOW}  部分设备可能被占用，继续训练可能失败${NC}"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "训练已取消"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 可用设备充足: ${NEURON_COUNT} >= ${NPROC}${NC}"
fi

# 设置环境变量
echo ""
echo -e "${YELLOW}设置环境变量...${NC}"
export WANDB_MODE=disabled
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type transformer --retry_failed_compilation}"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export MALLOC_ARENA_MAX=64

echo -e "${GREEN}✓ 环境变量已设置${NC}"

# 创建日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}开始训练...${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "日志文件: ${LOG_FILE}"
echo ""
echo -e "${YELLOW}提示: 首次运行会编译模型（较慢），后续运行会使用缓存${NC}"
echo ""

# 启动训练
torchrun --nproc_per_node=${NPROC} \
    scripts/train-optimum-neuron.py \
    "${CONFIG}" \
    --wandb=null \
    --max_duration=${MAX_STEPS} \
    --save_interval=${SAVE_INTERVAL} \
    2>&1 | tee "${LOG_FILE}"

# 检查退出状态
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}训练成功完成！${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "日志文件: ${LOG_FILE}"
    echo "Checkpoint: workspace/OLMo-20M-neuron-optimum/"
    echo ""

    # 显示最后的性能指标
    echo "最终性能指标:"
    grep "summary/" "${LOG_FILE}" | tail -10 || true
else
    echo -e "${RED}================================${NC}"
    echo -e "${RED}训练失败 (退出码: ${EXIT_CODE})${NC}"
    echo -e "${RED}================================${NC}"
    echo ""
    echo "请检查日志: ${LOG_FILE}"
    echo ""
    echo "常见问题:"
    echo "  1. 设备不足 -> 减少进程数或清理进程: pkill -9 python3"
    echo "  2. 内存不足 -> 减小批次大小"
    echo "  3. 配置错误 -> 检查 num_workers=0"
    echo ""
    echo "详细故障排查请查看: TRAIN.md#故障排查"
    exit $EXIT_CODE
fi
