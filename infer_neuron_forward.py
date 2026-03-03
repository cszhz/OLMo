#!/usr/bin/env python3
"""
OLMo Neuron 推理脚本 - 简单前向传播
避免使用 generate() 方法的复杂逻辑
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 60)
    print("OLMo Neuron 推理测试 - 前向传播")
    print("=" * 60)
    print()

    # 使用小模型
    model_name = "allenai/OLMo-2-0425-1B"
    print(f"模型: {model_name}")
    print()

    # 加载 tokenizer
    print("1. 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✓ Tokenizer 已加载")

    # 加载模型
    print()
    print("2. 加载模型到 CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("   ✓ 模型已加载")

    # 准备输入
    print()
    print("3. 准备输入...")
    text = "Language modeling is"
    print(f"   输入: '{text}'")

    inputs = tokenizer(text, return_tensors='pt', return_token_type_ids=False)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    print(f"   输入形状: {input_ids.shape}")

    # 使用 Neuron
    print()
    print("4. 初始化 Neuron 设备...")
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"   ✓ Neuron 设备: {device}")

        # 移动模型到 Neuron
        print()
        print("5. 编译模型到 Neuron (约 1-2 分钟)...")
        model = model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 预热
        print("   执行预热编译...")
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        xm.mark_step()
        print("   ✓ 编译完成")

        # 简单的自回归生成 - 逐个生成 token
        print()
        print("6. 运行推理 (生成 10 个 tokens)...")

        generated_ids = input_ids.clone()

        for i in range(10):
            with torch.no_grad():
                outputs = model(input_ids=generated_ids, attention_mask=None)
                logits = outputs.logits

                # 获取最后一个位置的 logits
                next_token_logits = logits[:, -1, :]

                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 添加到序列
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            xm.mark_step()

            # 打印进度
            if (i + 1) % 5 == 0:
                print(f"   已生成 {i + 1} 个 tokens...")

        # 将结果移回 CPU 并解码
        generated_ids_cpu = generated_ids.cpu()
        generated_text = tokenizer.decode(generated_ids_cpu[0], skip_special_tokens=True)

        print()
        print("=" * 60)
        print("推理结果:")
        print("-" * 60)
        print(f"输入: {text}")
        print(f"输出: {generated_text}")
        print("=" * 60)
        print()
        print("✓ 推理完成！")

    except Exception as e:
        print(f"   ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
