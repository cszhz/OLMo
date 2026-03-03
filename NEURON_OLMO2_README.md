# NeuronOLMo2ForCausalLM

OLMo-2 模型的 AWS Neuron 推理优化实现，基于 `neuronx-distributed-inference` 框架。

## 📁 文件结构

```
/home/ubuntu/OLMo/
├── neuronx_olmo2/
│   ├── __init__.py              # 模块初始化
│   └── modeling_olmo2.py        # NeuronOlmo2 核心实现
├── compile_olmo2_neuron.py      # 编译和推理脚本
├── test_neuron_olmo2.py         # 测试脚本
└── NEURON_OLMO2_README.md       # 本文档
```

## ✅ 已完成

### 1. 核心实现 (`neuronx_olmo2/modeling_olmo2.py`)

- ✅ **Olmo2InferenceConfig**: 推理配置类
- ✅ **Olmo2NeuronConfig**: Neuron 配置类
- ✅ **NeuronOlmo2Attention**: Neuron 优化的注意力层
  - 使用 `NeuronAttentionBase`
  - 支持 RoPE 位置编码
  - 支持 QK norm（OLMo2 特性）
- ✅ **NeuronOlmo2DecoderLayer**: 解码器层
  - 重用 `NeuronLlamaMLP`（SwiGLU 激活）
  - RMSNorm 归一化
- ✅ **NeuronOlmo2Model**: 基础模型
  - 并行 Embedding (`ParallelEmbedding`)
  - 并行线性层 (`ColumnParallelLinear`)
  - 支持张量并行 (TP)
- ✅ **NeuronOlmo2ForCausalLM**: 因果语言模型
  - HF 权重转换
  - 权重共享处理

### 2. 测试验证

运行测试：
```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python3 /home/ubuntu/OLMo/test_neuron_olmo2.py
```

结果：
```
✓ 模块导入成功
✓ 配置创建成功
✓ Attention 初始化成功
✓ DecoderLayer 初始化成功
✓ 模型类定义正确
```

## 🚀 使用方法

### 方式 1：编译模型

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

python3 compile_olmo2_neuron.py \
    --model-path allenai/OLMo-2-0425-1B \
    --compiled-model-path /tmp/olmo2_neuron_compiled \
    --tp-degree 2 \
    --batch-size 1 \
    --n-positions 2048 \
    --torch-dtype float32 \
    --buckets "[128, 256, 512, 1024, 2048]"
```

**注意**：编译后不会自动运行推理，因为需要先加载模型到 Neuron 设备。

**参数说明：**
- `--model-path`: HuggingFace 模型路径
- `--compiled-model-path`: 编译后模型保存路径
- `--tp-degree`: 张量并行度（2, 4, 8, 16...）
- `--batch-size`: 批次大小
- `--n-positions`: 最大序列长度
- `--torch-dtype`: 数据类型 (float32/float16/bfloat16)
- `--buckets`: 序列长度 buckets（优化编译）
- `--test-inference`: 编译后测试推理
- `--prompt`: 测试 prompt
- `--max-new-tokens`: 生成 token 数量

### 方式 2：加载并推理（使用已编译模型）

```bash
python3 compile_olmo2_neuron.py \
    --inference-only \
    --model-path allenai/OLMo-2-0425-1B \
    --compiled-model-path /tmp/olmo2_neuron_compiled \
    --tp-degree 2 \
    --batch-size 1 \
    --n-positions 2048 \
    --prompt "Language modeling is" \
    --max-new-tokens 20
```

**推理方法说明**：
- `NeuronOlmo2ForCausalLM` 没有 `generate()` 方法
- 使用自定义的 `greedy_generate()` 函数，直接调用 `model.forward()`
- 需要手动构造 `attention_mask`、`position_ids`、`seq_ids`、`sampling_params` 等输入
- 使用自回归循环逐个生成 token

## 🔍 推理实现细节

### NeuronBaseForCausalLM 的推理方法

`neuronx-distributed-inference` 的模型**不提供** HuggingFace 风格的 `generate()` 方法。需要：

1. **手动调用 forward()**: 直接使用 `model.forward()` 或 `model()`
2. **准备输入参数**:
   - `input_ids`: Token IDs [batch_size, seq_len]
   - `seq_ids`: 序列 IDs [batch_size, 1]
   - `attention_mask`: 注意力掩码 [batch_size, seq_len]
   - `position_ids`: 位置 IDs [batch_size, seq_len]
   - `sampling_params`: 采样参数 [batch_size, 3] - [top_k, top_p, temperature]
3. **自回归循环**: 手动实现 token-by-token 生成循环

### 推理流程

```python
# 1. Context Encoding（首次 forward）
outputs = model(
    input_ids=input_ids,  # 初始 prompt tokens
    seq_ids=seq_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    sampling_params=sampling_params,
)

# 2. Token Generation（循环生成）
for i in range(max_new_tokens):
    logits = outputs[0]  # or outputs.logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

    # 更新 attention_mask 和 position_ids
    # 再次调用 forward
    outputs = model(input_ids, seq_ids, attention_mask, position_ids, sampling_params)
```

## 🔧 关键技术细节

### 1. OLMo2 与 Qwen3 的架构差异

| 特性 | OLMo2 | Qwen3 |
|------|-------|-------|
| QK Norm 维度 | 整个投影维度 (num_heads × head_dim) | 每个头独立 (head_dim) |
| KV 头 | 16 (MHA) | 通常 < num_heads (GQA) |
| RoPE theta | 500000 | 10000 |
| 激活函数 | SiLU (SwiGLU) | SiLU |

### 2. Neuron 优化

- **并行化**:
  - Embedding: `ParallelEmbedding` (vocab 并行)
  - Linear: `ColumnParallelLinear` (列并行)
  - Attention: 张量并行 (TP)

- **自定义算子**:
  - `CustomRMSNorm`: Neuron 优化的 RMSNorm
  - `NeuronAttentionBase`: Neuron 优化的注意力

- **Buckets**:
  - 预编译多个序列长度
  - 运行时选择最接近的 bucket
  - 减少动态编译开销

### 3. 权重转换

HF 格式 → Neuron 格式：
```python
# q_norm / k_norm 重命名
layers.{i}.self_attn.q_norm.weight
  → layers.{i}.self_attn.q_layernorm.weight

# 添加 rank 信息（张量并行）
layers.{i}.self_attn.rank_util.rank = [0, 1, ..., tp_degree-1]

# 权重共享
lm_head.weight = embed_tokens.weight.clone()
```

## ⚙️ 编译流程

1. **加载 HF 配置** → 创建 `Olmo2InferenceConfig`
2. **初始化 Neuron 模型** → `NeuronOlmo2ForCausalLM`
3. **转换权重** → `convert_hf_to_neuron_state_dict()`
4. **编译到 Neuron** → `model.compile()`
   - 为每个 bucket 编译
   - 生成 NEFF 文件
   - 保存编译结果
5. **序列化** → 保存到 `compiled_model_path`

## 📊 预期性能

### 编译时间（预估）

| 配置 | 时间 |
|------|------|
| TP=2, buckets=5 | 15-30 分钟 |
| TP=4, buckets=5 | 20-40 分钟 |
| TP=8, buckets=5 | 30-60 分钟 |

### 推理性能

- **首次推理**: 需要加载编译模型 (~10秒)
- **后续推理**: 快速 (毫秒级)
- **吞吐量**: 取决于 TP degree 和硬件

## 🐛 已知问题

1. **没有 generate() 方法**:
   ```python
   AttributeError: 'NeuronOlmo2ForCausalLM' object has no attribute 'generate'
   ```
   - **原因**: `NeuronBaseForCausalLM` 不提供 HuggingFace 风格的 `generate()` 方法
   - **解决方案**: 使用自定义的 `greedy_generate()` 函数（见 `compile_olmo2_neuron.py`）
   - **替代方案**: 直接调用 `model.forward()` 实现自己的生成循环

2. **TP degree 和 KV heads 不整除**:
   ```
   WARNING: TP degree (1) and KV heads (16) are not divisible
   ```
   - 解决方案：使用 TP=2, 4, 8, 16

3. **内存占用**:
   - 1.48B 模型需要足够的 Neuron 内存
   - 建议使用 trn1.32xlarge 或更大

4. **编译缓慢**:
   - 首次编译很慢（每个 bucket 需要几分钟）
   - 使用较少的 buckets 可以加快编译

5. **推理需要加载到 Neuron 设备**:
   - 编译后不能直接推理
   - 必须先通过 `model.load(compiled_model_path)` 加载到 Neuron 设备
   - 加载过程需要在 Neuron 实例（Trn1/Inf2）上运行

## 📝 当前状态和下一步

### ✅ 已完成
- [x] 核心实现完成（`NeuronOlmo2ForCausalLM`）
- [x] 编译脚本完成（`compile_olmo2_neuron.py`）
- [x] 单元测试通过（`test_neuron_olmo2.py`）
- [x] 编译测试成功（TP=2, bucket=128）
- [x] 解决 RMSNorm 维度问题
- [x] 实现自定义 `greedy_generate()` 函数

### 🔄 待测试
- [ ] **实际推理测试**（需要在 Neuron 实例上运行 `--inference-only` 模式）
- [ ] 多 TP degree 测试 (2, 4, 8, 16)
- [ ] 完整 bucket 配置测试（多个 buckets）
- [ ] 推理精度验证（与 HuggingFace 对比）
- [ ] 性能基准测试（延迟、吞吐量）

### 🚀 可能的优化
- [ ] 改进生成循环性能
- [ ] 添加 beam search 支持
- [ ] 添加 top-p / top-k sampling 支持
- [ ] 优化权重加载速度
- [ ] 添加量化支持（INT8）
- [ ] 集成 HuggingFace transformers 的 GenerationMixin（如果可能）

## 🔗 参考实现

基于以下文件：
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/qwen3/modeling_qwen3.py`
- `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../neuronx_distributed_inference/models/llama/modeling_llama.py`

## 📄 License

基于 Apache 2.0 许可证
