# OLMo Neuron 训练指南（中文）

使用 optimum-neuron 在 AWS Trainium 上训练 OLMo 模型的快速指南。

## 快速开始

### 1. 激活环境

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd ~/OLMo
```

### 2. 运行训练

```bash
# 基础训练（100步）
export WANDB_MODE=disabled
torchrun --nproc_per_node=2 scripts/train-optimum-neuron.py \
    configs/tiny/OLMo-20M-neuron-optimum.yaml \
    --wandb=null \
    --max_duration=100
```

### 3. 带checkpoint保存的训练

```bash
# 每20步保存checkpoint
torchrun --nproc_per_node=2 scripts/train-optimum-neuron.py \
    configs/tiny/OLMo-20M-neuron-optimum.yaml \
    --wandb=null \
    --max_duration=100 \
    --save_interval=20
```

## 性能指标

在 trn2.48xlarge 上使用 OLMo-20M (2进程):
- **吞吐量**: ~800K tokens/秒
- **MFU**: 85%
- **步骤耗时**: 0.041秒

## 关键配置

### 必须设置的参数

```yaml
# configs/tiny/OLMo-20M-neuron-optimum.yaml

data:
  num_workers: 0              # ⚠️ 必须为0（避免死锁）
  prefetch_factor: null
  persistent_workers: false

precision: amp_bf16            # 使用BF16精度
```

### 重要参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `device_train_microbatch_size` | 微批次大小 | 1-2 |
| `global_train_batch_size` | 全局批次 | 16-32 |
| `max_duration` | 训练步数 | 根据需求 |
| `save_interval` | 保存间隔 | 100-1000 |

## 常见命令

### 查看 Neuron 设备

```bash
neuron-ls
```

### 清理僵尸进程

```bash
pkill -9 python3
```

### 监控训练

```bash
# 实时监控
neuron-top

# 查看日志
tail -f /tmp/train.log
```

## 故障排查

### 1. 训练挂起

**问题**: 编译后挂起

**解决**: 确保 `num_workers: 0`

### 2. 设备不足

**问题**: `not enough NeuronCores`

**解决**:
```bash
# 清理进程
pkill -9 python3
# 检查设备
neuron-ls
```

### 3. 内存不足

**解决**: 减小批次大小
```bash
--device_train_microbatch_size=1 \
--global_train_batch_size=8
```

## Checkpoint 管理

### Checkpoint 位置

```
workspace/OLMo-20M-neuron-optimum/
├── checkpoint-20/
│   ├── pytorch_model.bin    # 模型权重 (51MB)
│   ├── config.yaml          # 配置
│   ├── optimizer.pt         # 优化器
│   └── scheduler.pt         # 调度器
└── checkpoint-40/
    └── ...
```

### 加载 Checkpoint

```python
import torch
from olmo.config import ModelConfig
from olmo_neuron_wrapper import OLMoForCausalLM

# 加载
config = ModelConfig.load('workspace/.../checkpoint-20/config.yaml')
model = OLMoForCausalLM(config)
state = torch.load('workspace/.../checkpoint-20/pytorch_model.bin')
model.model.load_state_dict(state)
```

## 核心文件

| 文件 | 说明 |
|------|------|
| `scripts/train-optimum-neuron.py` | 训练脚本 |
| `olmo_neuron_wrapper.py` | Neuron包装器 |
| `configs/tiny/OLMo-20M-neuron-optimum.yaml` | 配置 |
| `scripts/train-optimum-neuron.sh` | 启动脚本 |

## 技术亮点

### 1. OLMo Neuron 包装器

- 使 OLMo 兼容 optimum-neuron
- 继承 `PreTrainedModel` 支持checkpoint
- 自动处理配置映射

### 2. 自动修复

- 自动修复 peft 导入问题
- 自动初始化并行组
- 自动禁用有问题的特性

### 3. 优化特性

- 编译缓存加速
- 数据并行训练
- BF16 混合精度
- ZeRO-1 优化器

## 进阶使用

### 增加并行度

```bash
# 32进程训练
torchrun --nproc_per_node=32 scripts/train-optimum-neuron.py \
    configs/your-config.yaml \
    --wandb=null \
    --max_duration=1000
```

### 自定义配置

```bash
# 命令行覆盖配置
torchrun --nproc_per_node=2 scripts/train-optimum-neuron.py \
    configs/your-config.yaml \
    --max_duration=500 \
    --optimizer.learning_rate=1e-5 \
    --global_train_batch_size=32
```

### 恢复训练

```bash
# 从checkpoint恢复
torchrun --nproc_per_node=2 scripts/train-optimum-neuron.py \
    configs/your-config.yaml \
    --resume_from_checkpoint=workspace/.../checkpoint-100
```

## 性能优化建议

1. **增加进程数**: 2 → 4 → 8 → 32
2. **增大批次**: 提高硬件利用率
3. **使用缓存**: 不要删除编译缓存
4. **调整精度**: 使用 BF16 平衡速度和精度

## 完整示例

```bash
#!/bin/bash
# 完整的训练启动脚本

# 1. 激活环境
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# 2. 进入目录
cd ~/OLMo

# 3. 设置环境变量
export WANDB_MODE=disabled
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1

# 4. 启动训练
torchrun --nproc_per_node=2 \
    scripts/train-optimum-neuron.py \
    configs/tiny/OLMo-20M-neuron-optimum.yaml \
    --wandb=null \
    --max_duration=1000 \
    --save_interval=100 \
    --optimizer.learning_rate=2e-5 \
    --global_train_batch_size=16

echo "训练完成！"
```

## 监控指标

训练过程中会显示：

```
{'loss': 0.0,
 'learning_rate': 2e-06,
 'train/tokens_per_sec': 800000,      # 吞吐量
 'train/mfu': 85.0,                   # 硬件利用率
 'train/step_time': 0.041,            # 步骤时间
 'train/efficiency': 10.5,            # 效率
 'epoch': 0.5}
```

## 资源链接

- **详细文档**: `TRAIN.md`
- **AWS Neuron**: https://awsdocs-neuron.readthedocs-hosted.com/
- **Optimum Neuron**: https://huggingface.co/docs/optimum-neuron/
- **OLMo**: https://github.com/allenai/OLMo

---

## 快速命令参考

```bash
# 激活环境
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# 基础训练
cd ~/OLMo && export WANDB_MODE=disabled
torchrun --nproc_per_node=2 scripts/train-optimum-neuron.py \
    configs/tiny/OLMo-20M-neuron-optimum.yaml --wandb=null --max_duration=100

# 查看设备
neuron-ls

# 清理进程
pkill -9 python3

# 监控
neuron-top
```

---

**提示**: 首次运行会编译模型（较慢），后续运行使用缓存（快速）。

*最后更新：2026-03-04*
