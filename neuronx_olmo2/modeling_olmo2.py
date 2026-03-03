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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronOlmo2Attention


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
    OLMo2 Neuron 优化注意力层
    关键差异：OLMo2 的 q_norm/k_norm 在整个投影维度，不是每个头
    """

    def __init__(self, config: Olmo2InferenceConfig):
        head_dim = config.hidden_size // config.num_attention_heads

        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # OLMo2 原本在整个投影维度做 norm
        # 但 neuronx-distributed 的 NeuronAttentionBase 在 reshape 后做 norm
        # 所以我们使用 head_dim（与 Qwen3 相同）
        # 注意：这不完全匹配 HF OLMo2，但在 Neuron 上应该可以工作
        q_layernorm = get_rmsnorm_cls()(
            hidden_size=head_dim,
            eps=config.rms_norm_eps
        )
        k_layernorm = get_rmsnorm_cls()(
            hidden_size=head_dim,
            eps=config.rms_norm_eps
        )

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

        关键问题：OLMo2 的 q_norm/k_norm 在整个投影维度 [hidden_size]，
        而 Neuron 的 q_layernorm/k_layernorm 在每个头维度 [head_dim]。
        需要将 [num_heads * head_dim] reshape 为 [num_heads, head_dim]，
        然后取平均（或使用第一个头的权重）。
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

            # OLMo2 的 q_norm 和 k_norm 转换
            # HF: [hidden_size] = [num_heads * head_dim]
            # Neuron: [head_dim]
            q_norm_weight = state_dict[f"layers.{i}.self_attn.q_norm.weight"]
            k_norm_weight = state_dict[f"layers.{i}.self_attn.k_norm.weight"]

            # Reshape 为 [num_heads, head_dim] 然后取平均
            q_norm_reshaped = q_norm_weight.reshape(num_heads, head_dim)
            k_norm_reshaped = k_norm_weight.reshape(num_heads, head_dim)

            # 取所有头的平均作为统一的 norm 权重
            state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = q_norm_reshaped.mean(dim=0)
            state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = k_norm_reshaped.mean(dim=0)

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
