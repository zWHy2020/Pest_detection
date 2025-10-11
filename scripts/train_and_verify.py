# scripts/train_and_verify.py
"""
训练并验证脚本
用于训练少量epoch后，立即检测HSI模态是否被有效忽略
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
from PIL import Image
import numpy as np
import warnings
import gc # 导入gc模块

# 导入您项目中的现有模块
from train_qwen_fixed import MemoryEfficientQwenTrainer  # 复用您已经完善的训练器
from models.main_model_qwen import MultiModalPestDetectionWithQwen
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

def run_verification_test(checkpoint_path: str, qwen_path: str, verify_rgb_image: str, num_classes: int, device: str):
    """
    执行敏感性测试以验证HSI是否被忽略
    """
    print("\n" + "="*60)
    print(" 🚀 开始执行HSI模态敏感性验证 ")
    print("="*60)

    # 1. 加载刚刚训练好的模型
    print(f"[验证步骤 1] 正在加载模型: {checkpoint_path}")
    model = MultiModalPestDetectionWithQwen(
        num_classes=num_classes,
        qwen_path=qwen_path,
        use_hsi=False,  # 必须以无HSI模式加载
        use_lora=True
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("模型加载成功！")

    # 2. 准备输入数据
    print("\n[验证步骤 2] 正在准备输入数据...")
    rgb_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    rgb_image = Image.open(verify_rgb_image).convert('RGB')
    rgb_tensor = rgb_transform(image=np.array(rgb_image))['image'].unsqueeze(0).to(device)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    text = "一片有病斑的植物叶子"
    text_encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    text_ids = text_encoded['input_ids'].to(device)
    text_mask = text_encoded['attention_mask'].to(device)

    hsi_zeros = torch.zeros(1, 224, 64, 64).to(device)
    hsi_random = torch.randn(1, 224, 64, 64).to(device)
    print("输入数据准备完毕。")

    # 3. 执行两次推理并比较结果
    print("\n[验证步骤 3] 正在执行两次推理并比较...")
    with torch.no_grad():
        logits_with_zeros = model(rgb_tensor, hsi_zeros, text_ids, text_mask)['logits']
        print("完成第一次推理 (使用全零HSI)。")
        
        logits_with_random = model(rgb_tensor, hsi_random, text_ids, text_mask)['logits']
        print("完成第二次推理 (使用随机HSI)。")

    abs_diff = torch.abs(logits_with_zeros - logits_with_random).mean().item()
    is_close = torch.allclose(logits_with_zeros, logits_with_random, atol=1e-5)

    print(f"\n两次推理输出的平均绝对差异: {abs_diff:.8f}")
    
    print("\n" + "="*60)
    print(" 最终验证结论 ")
    print("="*60)
    if is_close:
        print("✅ 验证通过：HSI模态已被成功忽略。")
        print("   解释：改变HSI输入对模型最终预测结果几乎没有影响，这完全符合预期。")
    else:
        print("❌ 验证失败：HSI模态仍然在影响模型。")
        print("   解释：改变HSI输入对模型最终预测结果产生了明显影响，请检查代码修改。")
    print("="*60)


def parse_args():
    # 复用并扩展 train_qwen_fixed.py 的参数解析器
    parser = argparse.ArgumentParser(description='训练并验证Qwen模型')
    
    # 训练参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数 (建议设置为1或2用于快速验证)')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度（推荐启用）')
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--qwen_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/train_and_verify')
    
    # 验证专用参数
    parser.add_argument('--verify_rgb_image', type=str, required=True, help='用于验证的一张RGB图像路径')
    parser.add_argument('--num_classes', type=int, default=38, help='数据集的类别总数 (例如PlantVillage为38)')

    # HSI开关
    parser.add_argument('--use_hsi', action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()

def main():
    args = parse_args()

    # 强制启用混合精度以保证能运行
    if not args.use_amp:
        print("警告: 自动启用混合精度以节省显存。")
        args.use_amp = True

    # 检查是否处于“无HSI”模式
    if args.use_hsi:
        print("="*60)
        print("⚠️ 警告: 此脚本的验证部分仅在'无HSI'模式下有意义。")
        print("请使用 '--no-use_hsi' 参数来运行此脚本以进行有效验证。")
        print("="*60)

    # --- 训练阶段 ---
    print("\n" + "#"*60)
    print("### 开始训练阶段 ###")
    print("#"*60)
    
    trainer = MemoryEfficientQwenTrainer(args)
    trainer.train()

    # --- 验证阶段 ---
    print("\n" + "#"*60)
    print("### 训练结束, 开始自动验证阶段 ###")
    print("#"*60)

    checkpoint_path = os.path.join(trainer.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到训练好的模型 {checkpoint_path}。验证无法进行。")
        return

    ### 关键修复：在这里释放训练器和它占用的GPU显存 ###
    print("\n释放训练阶段占用的GPU显存...")
    del trainer  # 删除trainer对象，释放对第一个模型的引用
    gc.collect() # 触发Python的垃圾回收
    torch.cuda.empty_cache() # 指示PyTorch清空显存缓存
    print("显存已成功释放。")
    ### 修复结束 ###
    
    run_verification_test(
        checkpoint_path=checkpoint_path,
        qwen_path=args.qwen_path,
        verify_rgb_image=args.verify_rgb_image,
        num_classes=args.num_classes,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == '__main__':
    main()