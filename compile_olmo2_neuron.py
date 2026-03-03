#!/usr/bin/env python3
"""
OLMo2 Neuron 编译脚本
使用 neuronx_distributed_inference 框架编译 OLMo2 模型
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# 添加 neuronx_olmo2 到路径
sys.path.insert(0, '/home/ubuntu/OLMo')

from neuronx_olmo2.modeling_olmo2 import (
    NeuronOlmo2ForCausalLM,
    Olmo2InferenceConfig,
)


def create_config(args):
    """创建 OLMo2 推理配置"""

    # 从 HF 加载基础配置
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model_path)

    # 先创建 NeuronConfig
    from neuronx_olmo2.modeling_olmo2 import Olmo2NeuronConfig

    # 转换 torch_dtype
    if isinstance(args.torch_dtype, str):
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[args.torch_dtype]
    else:
        torch_dtype = args.torch_dtype

    neuron_config = Olmo2NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        seq_len=args.n_positions,
        torch_dtype=torch_dtype,
        qk_norm_strategy=args.qk_norm_strategy,
    )

    # 设置 buckets
    if args.buckets:
        neuron_config.buckets = eval(args.buckets)
    else:
        # 默认 buckets：[128, 256, 512, 1024, 2048]
        neuron_config.buckets = [128, 256, 512, 1024, 2048]

    # 创建推理配置
    config_kwargs = {
        # Neuron config
        "neuron_config": neuron_config,

        # 从 HF config 复制
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "vocab_size": hf_config.vocab_size,
        "max_position_embeddings": hf_config.max_position_embeddings,
        "rope_theta": hf_config.rope_theta,
        "rms_norm_eps": hf_config.rms_norm_eps,
        "hidden_act": hf_config.hidden_act,
        "pad_token_id": hf_config.pad_token_id,
        "intermediate_size": hf_config.intermediate_size,
    }

    config = Olmo2InferenceConfig(**config_kwargs)

    return config


def compile_model(args):
    """编译 OLMo2 模型到 Neuron"""

    print("=" * 60)
    print("OLMo2 Neuron 编译")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  输出路径: {args.compiled_model_path}")
    print(f"  TP degree: {args.tp_degree}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  序列长度: {args.n_positions}")
    print(f"  数据类型: {args.torch_dtype}")

    # 1. 创建配置
    print("\n1. 创建推理配置...")
    config = create_config(args)
    print(f"   ✓ 配置已创建")
    print(f"   Buckets: {config.neuron_config.buckets}")

    # 2. 创建 Neuron 模型
    print("\n2. 初始化 Neuron 模型...")
    model = NeuronOlmo2ForCausalLM(
        model_path=args.model_path,
        config=config,
    )
    print("   ✓ 模型已初始化")

    # 3. 编译模型
    print("\n3. 编译模型到 Neuron...")
    print("   这可能需要 10-30 分钟...")

    model.compile(
        compiled_model_path=args.compiled_model_path
    )

    print(f"   ✓ 编译完成！")
    print(f"   编译结果保存在: {args.compiled_model_path}")

    return model, config


def test_inference(model, config, args):
    """测试推理"""

    print("\n" + "=" * 60)
    print("测试推理")
    print("=" * 60)

    # 加载 tokenizer
    print("\n1. 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 设置 pad_token（如果没有）
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("   ✓ Tokenizer 已加载")
    print(f"   Pad token ID: {tokenizer.pad_token_id}")

    # 准备输入
    print("\n2. 准备输入...")
    prompt = args.prompt if args.prompt else "Language modeling is"
    print(f"   Prompt: '{prompt}'")

    # Tokenize 输入
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
    input_ids = inputs['input_ids']
    print(f"   输入 token 数: {input_ids.shape[1]}")
    print(f"   输入 IDs: {input_ids[0].tolist()}")

    # 推理 - 使用 model.generate() 方法
    print("\n3. 运行推理...")
    print(f"   生成 {args.max_new_tokens} 个 tokens...")

    try:
        # 使用模型的 generate() 方法（HuggingFace 风格）
        generated_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,  # 贪婪解码
        )

        # 解码
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        print("\n" + "=" * 60)
        print("推理结果:")
        print("-" * 60)
        print(generated_text)
        print("=" * 60)
        print("\n✓ 推理完成！")

    except Exception as e:
        print(f"\n✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示: neuronx-distributed-inference 的推理需要模型已加载到 Neuron 设备")
        print("      请确保已运行编译并且模型已通过 model.load() 加载")


def main():
    parser = argparse.ArgumentParser(description="OLMo2 Neuron 编译")

    # 模型参数
    parser.add_argument(
        "--model-path",
        type=str,
        default="allenai/OLMo-2-0425-1B",
        help="HuggingFace 模型路径"
    )
    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default="/tmp/olmo2_neuron_compiled",
        help="编译后模型保存路径"
    )

    # Neuron 配置
    parser.add_argument("--tp-degree", type=int, default=2, help="张量并行度")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--n-positions", type=int, default=2048, help="最大序列长度")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="数据类型"
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default=None,
        help="序列长度 buckets，如 '[128, 256, 512]'"
    )
    parser.add_argument(
        "--qk-norm-strategy",
        type=str,
        default="weighted_mean",
        choices=["weighted_mean", "mean", "rms", "first", "median"],
        help="QK norm 权重转换策略：weighted_mean(加权平均,推荐), mean(简单平均), rms(均方根), first(第一个头), median(中位数头)"
    )

    # 推理测试参数
    parser.add_argument("--test-inference", action="store_true", help="编译后测试推理")
    parser.add_argument("--prompt", type=str, default=None, help="测试 prompt")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="生成 token 数")

    # 仅推理模式
    parser.add_argument("--inference-only", action="store_true", help="只推理，不编译")

    args = parser.parse_args()

    try:
        if args.inference_only:
            # 只推理模式：加载已编译模型
            print("加载已编译模型...")
            config = create_config(args)
            model = NeuronOlmo2ForCausalLM(
                model_path=args.model_path,
                config=config,
            )
            model.load(args.compiled_model_path)
            print("✓ 模型已加载")
            test_inference(model, config, args)
        else:
            # 编译模式
            model, config = compile_model(args)

            if args.test_inference:
                print("\n注意: 推理测试需要模型加载到 Neuron 设备")
                print("      当前跳过推理测试")
                print("      要测试推理，请使用: --inference-only 标志")
                # test_inference(model, config, args)

        print("\n✓ 完成！")
        return 0

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
