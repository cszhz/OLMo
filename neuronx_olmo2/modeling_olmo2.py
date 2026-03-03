# coding=utf-8
# Copyright 2025 AI2 and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch OLMo2 model for NXD inference
基于 neuronx_distributed_inference Qwen3 实现改写
"""
from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from transformers import Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm
    - If infer on NXD -> CustomRMSNorm
    - If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    """
    return Olmo2RMSNorm if cpu_mode() else CustomRMSNorm


class Olmo2NeuronConfig(NeuronConfig):
    def __init__(self, qk_norm_strategy='weighted_mean', **kwargs):
        """
        Args:
            qk_norm_strategy: QK norm 权重转换策略
                - 'weighted_mean': 加权平均（最接近 GPU，推荐）⭐
                - 'mean': 简单平均所有头的权重
                - 'rms': 均方根（保留幅度信息）
                - 'first': 使用第一个头（保留原始特征）
                - 'median': 使用中位数头（鲁棒性更好）
        """
        super().__init__(**kwargs)
        self.attn_cls = NeuronOlmo2Attention
        self.qk_norm_strategy = qk_norm_strategy


class Olmo2InferenceConfig(InferenceConfig):
    """OLMo2 推理配置"""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # 添加 transformers 期望的属性
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_cache = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "pad_token_id",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Olmo2NeuronConfig]:
        return Olmo2NeuronConfig


class NeuronOlmo2Attention(NeuronAttentionBase):
    """
    OLMo2 Neuron Attention - 完全匹配 GPU 的 QK norm

    关键改进：
    - q_norm 和 k_norm 在完整维度 [hidden_size] 上（与 GPU 一致）⭐
    - 重写 prep_qkv_tensors 在 projection 后、reshape 前应用 norm
    - 保留所有 Neuron 优化（NKI kernels、flash attention 等）
    """

    def __init__(self, config: Olmo2InferenceConfig):
        head_dim = config.hidden_size // config.num_attention_heads

        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # 传入 None 给父类，我们自己管理 norm
        q_layernorm = None
        k_layernorm = None

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            q_layernorm=q_layernorm,
            k_layernorm=k_layernorm,
        )

        # 创建 QK Norm
        # TP=1: 使用完整维度 [2048]，完全匹配 GPU ✅
        # TP>1: 使用分片维度，无法完全匹配 GPU（分片后无法在完整维度做 norm）
        tp_degree = config.neuron_config.tp_degree

        if tp_degree == 1:
            # TP=1 模式：完整维度，100% 匹配 GPU！⭐
            q_dim = config.hidden_size  # [2048]
            kv_dim = config.num_key_value_heads * head_dim  # [2048]
        else:
            # TP>1 模式：分片维度（无法完全匹配 GPU）
            q_dim = config.hidden_size // tp_degree
            kv_dim = (config.num_key_value_heads * head_dim) // tp_degree

        self.q_norm_full = get_rmsnorm_cls()(
            hidden_size=q_dim,
            eps=config.rms_norm_eps
        )
        self.k_norm_full = get_rmsnorm_cls()(
            hidden_size=kv_dim,
            eps=config.rms_norm_eps
        )

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """
        重写此方法，在 projection 后、reshape 前应用 norm
        与 GPU 一致：在完整 hidden_size 维度做 QK norm
        """
        # 1. QKV Projection - 返回 Q, K, V, residual (4个值！)
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states,
            rmsnorm=rmsnorm,
            adapter_ids=adapter_ids,
            residual=residual
        )

        # 2. 应用 QK Norm - 在完整维度上！与 GPU 一致 ⭐
        Q = self.q_norm_full(Q)  # [batch, seq, 2048] → 在 2048 维上做 norm
        K = self.k_norm_full(K)  # [batch, seq, num_kv_heads * head_dim]

        # 3. Reshape - 使用每个分片的头数
        bsz, q_len, _ = Q.size()
        if self.qkv_proj_sp_enabled:
            q_len *= self.tensor_model_parallel_group.size()
        # 在 TP 模式下，self.num_heads 和 self.num_key_value_heads 已经是每个分片的值
        Q = Q.view(bsz, q_len, self.num_heads, self.head_dim)
        K = K.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        V = V.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # 4. Transpose (BSHD -> BHSD)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 5. RoPE
        if not skip_rope:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(hidden_states, position_ids)
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
                Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
            )

        return Q, K, V, cos_cache, sin_cache, residual


class NeuronOlmo2DecoderLayer(nn.Module):
    """
    OLMo2 解码器层 - Neuron 优化版本
    使用 NXD 的注意力和 MLP 模块
    注意：OLMo2 使用 post-norm 架构（与 Llama 的 pre-norm 不同）
    """

    def __init__(self, config: Olmo2InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronOlmo2Attention(config)

        # 重用 Llama MLP（OLMo2 使用 SwiGLU，与 Llama 相同）
        self.mlp = NeuronLlamaMLP(config)

        # OLMo2 使用 post-norm：在 attention 和 MLP 之后做 norm
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self Attention (post-norm)
        residual = hidden_states
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP (post-norm)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states, adapter_ids=adapter_ids)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronOlmo2Model(NeuronBaseModel):
    """
    OLMo2 基础模型 - Neuron 优化版本
    """

    def setup_attr_for_model(self, config: Olmo2InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Olmo2InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 并行 Embedding
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # Decoder 层
        self.layers = nn.ModuleList(
            [NeuronOlmo2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # 最终归一化层
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head（并行线性层）
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )


class NeuronOlmo2ForCausalLM(NeuronBaseForCausalLM):
    """
    OLMo2 因果语言模型 - Neuron 推理版本
    可以直接替代 Olmo2ForCausalLM 用于 Neuron 推理
    """

    _model_cls = NeuronOlmo2Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """加载 HuggingFace 预训练模型"""
        # 移除路径末尾的斜杠（如果是 HF 模型名称）
        model_path = model_path.rstrip('/')
        return Olmo2ForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """
        转换 HuggingFace 模型权重到 Neuron 格式

        关键改进：⭐ 直接保留完整维度的 QK norm 权重 [hidden_size]
        - q_norm.weight: [2048] → q_norm_full.weight: [2048]  (不降维！)
        - k_norm.weight: [2048] → k_norm_full.weight: [2048]  (不降维！)

        完全匹配 GPU 行为，无信息损失！
        """
        neuron_config = config.neuron_config

        if neuron_config.vocab_parallel:
            # Vocab 并行需要的 rank 信息
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        for i in range(num_layers):
            # Attention rank 信息
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # OLMo2 的 q_norm 和 k_norm 权重转换
            q_norm_key = f"layers.{i}.self_attn.q_norm.weight"
            k_norm_key = f"layers.{i}.self_attn.k_norm.weight"

            # 只处理存在的层
            if q_norm_key in state_dict:
                q_norm_weight = state_dict.pop(q_norm_key)  # [2048]

                if tp_degree == 1:
                    # TP=1: 直接使用完整权重，100% 匹配 GPU！⭐
                    state_dict[f"layers.{i}.self_attn.q_norm_full.weight"] = q_norm_weight
                else:
                    # TP>1: 分片权重（无法完全匹配 GPU）
                    q_dim_per_shard = config.hidden_size // tp_degree
                    # 简单策略：取前部分权重切片
                    q_norm_weight_sharded = q_norm_weight[:q_dim_per_shard]
                    state_dict[f"layers.{i}.self_attn.q_norm_full.weight"] = q_norm_weight_sharded

            if k_norm_key in state_dict:
                k_norm_weight = state_dict.pop(k_norm_key)  # [2048]

                if tp_degree == 1:
                    # TP=1: 直接使用完整权重，100% 匹配 GPU！⭐
                    state_dict[f"layers.{i}.self_attn.k_norm_full.weight"] = k_norm_weight
                else:
                    # TP>1: 分片权重（无法完全匹配 GPU）
                    kv_dim_per_shard = (config.num_key_value_heads * head_dim) // tp_degree
                    k_norm_weight_sharded = k_norm_weight[:kv_dim_per_shard]
                    state_dict[f"layers.{i}.self_attn.k_norm_full.weight"] = k_norm_weight_sharded

        # Base model rank 信息
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        处理权重共享（OLMo2 的 lm_head 和 embed_tokens 共享权重）
        """
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        """返回配置类"""
        return Olmo2InferenceConfig

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        min_new_tokens: int = None,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
        use_cache: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本序列（HuggingFace 风格接口）

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            max_new_tokens: 最多生成多少个新 token
            min_new_tokens: 最少生成多少个新 token（暂未实现）
            do_sample: 是否采样（False=贪婪解码，True=采样）
            num_beams: beam search 的 beam 数量（>1 时使用 beam search）
            temperature: 温度参数（采样时使用，越高越随机）
            top_k: top-k 采样（只保留概率最高的 k 个 tokens）
            top_p: top-p nucleus 采样（只保留累积概率达到 p 的 tokens）
            length_penalty: 长度惩罚（beam search，>1鼓励长序列）
            early_stopping: 是否提前停止（beam search）
            pad_token_id: padding token ID
            eos_token_id: EOS token ID（用于提前停止）
            use_cache: 是否使用 KV cache（自动管理）

        Returns:
            生成的完整序列 [batch_size, seq_len + generated_len]
        """
        # 如果 num_beams > 1，使用 beam search
        if num_beams > 1:
            return self.beam_search(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        batch_size = self.neuron_config.batch_size
        current_ids = input_ids.clone()
        batch_actual = current_ids.shape[0]
        seq_len = current_ids.shape[1]

        # Pad 到 batch_size
        if batch_actual < batch_size:
            padding = torch.full(
                (batch_size - batch_actual, seq_len),
                pad_token_id,
                dtype=current_ids.dtype,
                device=current_ids.device
            )
            current_ids = torch.cat([current_ids, padding], dim=0)

        # 准备输入参数
        seq_ids = torch.arange(batch_size, dtype=torch.long, device=current_ids.device)

        # 采样参数
        if do_sample:
            sampling_params = torch.tensor(
                [[top_k, top_p, temperature]] * batch_size,
                dtype=torch.float32,
                device=current_ids.device
            )
        else:
            # 贪婪解码
            sampling_params = torch.tensor(
                [[1, 1.0, 1.0]] * batch_size,
                dtype=torch.float32,
                device=current_ids.device
            )

        # 生成循环
        # 注意：neuronx-distributed-inference 的 KV cache 由编译后的模型内部管理
        # 我们在 Python 层面进行增量生成优化
        past_key_values = None
        initial_seq_len = current_ids.shape[1]

        for i in range(max_new_tokens):
            seq_len = current_ids.shape[1]

            # KV cache 优化：首次传入完整序列，后续只传入新 token
            if use_cache and i > 0:
                # 增量生成：只处理最后一个新 token
                input_for_forward = current_ids[:, -1:]
                # position_ids 从当前位置开始
                position_ids = torch.full(
                    (batch_size, 1),
                    seq_len - 1,
                    dtype=torch.long,
                    device=current_ids.device
                )
                # attention_mask 扩展到当前长度
                attention_mask = (current_ids != pad_token_id).long()
            else:
                # 首次或不使用 cache：处理完整序列
                input_for_forward = current_ids
                position_ids = torch.arange(
                    seq_len, dtype=torch.long, device=current_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
                attention_mask = (current_ids != pad_token_id).long()

            # Forward pass
            with torch.no_grad():
                outputs = self(
                    input_ids=input_for_forward,
                    seq_ids=seq_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    sampling_params=sampling_params,
                )

            # 获取 logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits

            # 取最后一个位置的 logits
            next_token_logits = logits[:, -1, :]

            # 生成下一个 token
            if do_sample:
                # 采样模式 - 支持 top-k 和 top-p
                scores = next_token_logits.clone()

                # 应用温度
                if temperature != 1.0:
                    scores = scores / temperature

                # Top-k 过滤
                if top_k > 0:
                    top_k_val = min(top_k, scores.size(-1))
                    indices_to_remove = scores < torch.topk(scores, top_k_val)[0][..., -1, None]
                    scores[indices_to_remove] = float('-inf')

                # Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除累积概率超过 top_p 的 tokens
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留至少一个 token
                    sorted_indices_to_remove[..., 0] = False
                    # 将 mask 映射回原始索引
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    scores[indices_to_remove] = float('-inf')

                # 从过滤后的分布中采样
                probs = torch.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1)

            # 添加新 token
            current_ids = torch.cat([current_ids, next_token.unsqueeze(1)], dim=1)

            # 检查是否遇到 EOS（简化版）
            if eos_token_id is not None:
                # 如果所有序列都生成了 EOS，提前停止
                if (next_token[:batch_actual] == eos_token_id).all():
                    break

        # 只返回实际的批次
        return current_ids[:batch_actual]

    def beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Beam search 解码

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            max_new_tokens: 最多生成多少个新 token
            num_beams: beam 数量
            length_penalty: 长度惩罚（>1.0 鼓励更长序列，<1.0 鼓励更短序列）
            early_stopping: 是否在所有 beams 生成 EOS 时提前停止
            pad_token_id: padding token ID
            eos_token_id: EOS token ID

        Returns:
            生成的最佳序列 [batch_size, seq_len + generated_len]
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id

        batch_size = self.neuron_config.batch_size
        batch_actual = input_ids.shape[0]

        if batch_actual != 1:
            raise NotImplementedError("Beam search currently only supports batch_size=1")

        # 扩展 input_ids 到 num_beams 份
        seq_len = input_ids.shape[1]
        beam_input_ids = input_ids.repeat(num_beams, 1)  # [num_beams, seq_len]

        # Pad 到 batch_size
        if num_beams < batch_size:
            padding = torch.full(
                (batch_size - num_beams, seq_len),
                pad_token_id,
                dtype=beam_input_ids.dtype,
                device=beam_input_ids.device
            )
            beam_input_ids = torch.cat([beam_input_ids, padding], dim=0)

        # 初始化 beam scores
        beam_scores = torch.zeros(num_beams, device=input_ids.device)
        beam_scores[1:] = float('-inf')  # 只有第一个 beam 初始有效

        # 记录是否完成
        done_beams = torch.zeros(num_beams, dtype=torch.bool, device=input_ids.device)

        seq_ids = torch.arange(batch_size, dtype=torch.long, device=input_ids.device)
        sampling_params = torch.tensor(
            [[1, 1.0, 1.0]] * batch_size,
            dtype=torch.float32,
            device=input_ids.device
        )

        # Beam search 循环
        # KV cache 优化：首次传入完整序列，后续只传入新 token
        for step in range(max_new_tokens):
            current_len = beam_input_ids.shape[1]

            # KV cache 优化：增量生成
            if step > 0:
                # 只处理最后一个新 token
                input_for_forward = beam_input_ids[:, -1:]
                position_ids = torch.full(
                    (batch_size, 1),
                    current_len - 1,
                    dtype=torch.long,
                    device=input_ids.device
                )
                # attention_mask 扩展到当前长度
                attention_mask = (beam_input_ids != pad_token_id).long()
            else:
                # 首次：处理完整序列
                input_for_forward = beam_input_ids
                position_ids = torch.arange(
                    current_len, dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
                attention_mask = (beam_input_ids != pad_token_id).long()

            # Forward pass
            with torch.no_grad():
                outputs = self(
                    input_ids=input_for_forward,
                    seq_ids=seq_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    sampling_params=sampling_params,
                )

            # 获取 logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits

            # 只关注前 num_beams 个
            next_token_logits = logits[:num_beams, -1, :]  # [num_beams, vocab_size]
            vocab_size = next_token_logits.shape[-1]

            # 计算对数概率
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)  # [num_beams, vocab_size]

            # 将当前 beam scores 加到下一个 token scores 上
            next_token_scores = next_token_scores + beam_scores.unsqueeze(1)  # [num_beams, vocab_size]

            # 应用长度惩罚
            if length_penalty != 1.0:
                length_penalty_factor = ((current_len + 1) / (seq_len + 1)) ** length_penalty
                next_token_scores = next_token_scores / length_penalty_factor

            # 将已完成的 beams 的分数设为很低
            next_token_scores[done_beams] = float('-inf')

            # Reshape 为 [num_beams * vocab_size]
            next_token_scores_flat = next_token_scores.view(-1)

            # 选择 top num_beams 个 scores
            top_scores, top_indices = torch.topk(next_token_scores_flat, num_beams, largest=True)

            # 计算是哪个 beam 和哪个 token
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # 更新 beam_input_ids
            new_beam_input_ids = torch.cat([
                beam_input_ids[beam_indices, :],
                token_indices.unsqueeze(1)
            ], dim=1)

            # Pad 回 batch_size
            if num_beams < batch_size:
                padding = torch.full(
                    (batch_size - num_beams, new_beam_input_ids.shape[1]),
                    pad_token_id,
                    dtype=new_beam_input_ids.dtype,
                    device=new_beam_input_ids.device
                )
                beam_input_ids = torch.cat([new_beam_input_ids, padding], dim=0)
            else:
                beam_input_ids = new_beam_input_ids

            # 更新 beam scores
            beam_scores = top_scores

            # 检查是否生成了 EOS
            if eos_token_id is not None:
                done_beams = token_indices == eos_token_id

            # 如果所有 beams 都完成且 early_stopping，提前退出
            if early_stopping and done_beams.all():
                break

        # 返回最佳 beam（第一个，因为已经按 score 排序）
        return beam_input_ids[0:1, :]
