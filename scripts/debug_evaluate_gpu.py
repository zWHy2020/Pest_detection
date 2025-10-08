#!/usr/bin/env python
# scripts/evaluate_gpu_fixed.py
"""
修复版GPU评估脚本
确保所有操作都在GPU上进行
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import json
from typing import Dict, List, Optional

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.main_model import MultiModalPestDetection
from data.dataset import create_dataloaders
from utils.metrics import calculate_metrics, calculate_per_class_metrics, topk_accuracy
from utils.visualization import plot_confusion_matrix, plot_per_class_metrics


class GPUEvaluator:
    """GPU评估器 - 修复版"""
    
    def __init__(self, args):
        self.args = args
        
        print("="*60)
        print("GPU评估器 (修复版)")
        print("="*60)
        
        # 检查GPU
        if not torch.cuda.is_available():
            raise RuntimeError("需要GPU来运行此脚本")
        
        # 设置设备
        self.device = torch.device('cuda:0')
        self.num_gpus = torch.cuda.device_count()
        
        print(f"使用设备: {self.device}")
        print(f"可用GPU数量: {self.num_gpus}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 加载检查点
        self.load_checkpoint()
        
        # 创建数据加载器
        self.setup_dataloader()
        
        # 创建模型
        self.setup_model()
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
    
    def load_checkpoint(self):
        """安全加载检查点"""
        print(f"\n加载检查点: {self.args.checkpoint}")
        
        # PyTorch 2.6+ 兼容性
        try:
            self.checkpoint = torch.load(
                self.args.checkpoint, 
                map_location='cpu',
                weights_only=False
            )
        except:
            # 添加numpy安全全局变量
            import numpy
            torch.serialization.add_safe_globals([
                numpy.core.multiarray._reconstruct,
                numpy.ndarray,
                numpy.dtype
            ])
            self.checkpoint = torch.load(
                self.args.checkpoint, 
                map_location='cpu'
            )
        
        self.model_args = self.checkpoint.get('args', {})
        print("✓ 检查点加载成功")
        
        # 打印检查点信息
        if 'epoch' in self.checkpoint:
            print(f"  训练轮数: {self.checkpoint['epoch']}")
        if 'metrics' in self.checkpoint:
            metrics = self.checkpoint['metrics']
            if isinstance(metrics, dict):
                if 'val_acc' in metrics:
                    print(f"  验证准确率: {metrics['val_acc']:.4f}")
    
    def setup_dataloader(self):
        """创建数据加载器"""
        print(f"\n创建数据加载器...")
        
        # 创建数据加载器
        _, _, self.test_loader = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            text_model_name=self.model_args.get('text_model_name', 'bert-base-chinese'),
            use_augmentation=False
        )
        
        self.num_classes = self.test_loader.dataset.num_classes
        self.class_names = list(self.test_loader.dataset.class_to_idx.keys())
        
        print(f"  测试集大小: {len(self.test_loader.dataset)}")
        print(f"  批次大小: {self.args.batch_size}")
        print(f"  类别数: {self.num_classes}")
    
    def setup_model(self):
        """创建并设置模型"""
        print(f"\n创建模型...")
        
        # 创建模型
        self.model = MultiModalPestDetection(
            num_classes=self.num_classes,
            rgb_image_size=self.model_args.get('rgb_size', 224),
            hsi_image_size=self.model_args.get('hsi_size', 64),
            hsi_channels=self.model_args.get('hsi_channels', 224),
            text_model_name=self.model_args.get('text_model_name', 'bert-base-chinese'),
            embed_dim=self.model_args.get('embed_dim', 768),
            num_heads=self.model_args.get('num_heads', 12),
            dropout=0.0,
            fusion_layers=self.model_args.get('fusion_layers', 4),
            fusion_strategy=self.model_args.get('fusion_strategy', 'hierarchical'),
            llm_model_name=None,  # 不加载LLM
            use_lora=False,
            freeze_encoders=False
        )
        
        # 加载权重
        print("  加载模型权重...")
        state_dict = self.checkpoint['model_state_dict']
        
        # 处理权重键名（移除module.前缀和LLM相关）
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除DataParallel的module.前缀
            new_key = key[7:] if key.startswith('module.') else key
            
            # 跳过LLM相关权重
            if new_key.startswith(('llm.', 'llm_adapter.')):
                continue
            
            new_state_dict[new_key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.load_state_dict(
            new_state_dict, strict=False
        )
        
        print(f"  加载了 {len(new_state_dict)} 个权重")
        if len(missing_keys) > 0:
            print(f"  缺失的键（前5个）: {missing_keys[:5]}")
        
        # 移动模型到GPU
        print(f"  将模型移到GPU: {self.device}")
        self.model = self.model.to(self.device)
        
        # 设置评估模式
        self.model.eval()
        
        # 如果有多个GPU，使用DataParallel
        if self.num_gpus > 1 and self.args.use_multi_gpu:
            print(f"  使用DataParallel (GPU: 0-{self.num_gpus-1})")
            self.model = DataParallel(self.model)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  模型参数总量: {total_params / 1e6:.2f}M")
    
    @torch.no_grad()
    def evaluate(self):
        """执行评估（仅用于推理，不计算损失）"""
        print("\n" + "="*60)
        print("开始评估")
        print("="*60)
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        # 进度条
        pbar = tqdm(self.test_loader, desc='评估进度', ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到GPU
            rgb_images = batch['rgb_images'].to(self.device, non_blocking=True)
            hsi_images = batch['hsi_images'].to(self.device, non_blocking=True)
            text_input_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # 前向传播（不传入labels，避免计算损失）
            outputs = self.model(
                rgb_images, hsi_images,
                text_input_ids, text_attention_mask,
                labels=None,  # 不传入labels，只做推理
                return_features=self.args.extract_features
            )
            
            # 获取logits
            logits = outputs['logits']
            
            # 确保logits在GPU上
            if logits.device.type == 'cpu':
                print(f"警告: logits在CPU上，移到GPU")
                logits = logits.to(self.device)
            
            # 计算概率和预测
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # 收集结果（转到CPU）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 收集特征（如果需要）
            if self.args.extract_features and 'fused_features' in outputs:
                features = outputs['fused_features']
                if features.device.type == 'cpu':
                    features = features.to(self.device)
                all_features.extend(features.cpu().numpy())
            
            # 更新进度条
            current_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix({'acc': f'{current_acc:.3f}'})
            
            # 定期清理GPU缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print("\n计算评估指标...")
        
        # 计算指标
        metrics = calculate_metrics(
            all_labels, all_preds, all_probs,
            num_classes=self.num_classes
        )
        
        # 打印结果
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        
        # 计算Top-k准确率
        if self.args.compute_topk:
            probs_tensor = torch.from_numpy(all_probs)
            labels_tensor = torch.from_numpy(all_labels)
            top1, top5 = topk_accuracy(probs_tensor, labels_tensor, topk=(1, 5))
            print(f"Top-1 准确率: {top1:.2f}%")
            print(f"Top-5 准确率: {top5:.2f}%")
            metrics['top1_acc'] = top1
            metrics['top5_acc'] = top5
        
        # 计算每个类别的指标
        if self.args.compute_per_class:
            per_class_report = calculate_per_class_metrics(
                all_labels, all_preds, 
                class_names=self.class_names
            )
            
            print("\n各类别性能（F1分数）:")
            for class_name in self.class_names[:10]:  # 只显示前10个
                if class_name in per_class_report:
                    f1 = per_class_report[class_name]['f1-score']
                    support = per_class_report[class_name]['support']
                    print(f"  {class_name:15s}: {f1:.3f} (n={int(support)})")
        
        # 可视化
        if self.args.plot_confusion_matrix:
            print("\n生成混淆矩阵...")
            cm_path = os.path.join(self.args.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(
                all_labels, all_preds,
                class_names=self.class_names,
                save_path=cm_path,
                normalize=True
            )
            print(f"  保存到: {cm_path}")
        
        # 保存结果
        self.save_results(metrics, all_preds, all_labels, all_features)
        
        return metrics
    
    def save_results(self, metrics, preds, labels, features):
        """保存评估结果"""
        # 准备可序列化的结果
        results = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        }
        
        if 'top1_acc' in metrics:
            results['top1_accuracy'] = float(metrics['top1_acc'])
            results['top5_accuracy'] = float(metrics['top5_acc'])
        
        # 保存JSON结果
        json_path = os.path.join(self.args.output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果保存到: {json_path}")
        
        # 保存预测结果
        np.save(os.path.join(self.args.output_dir, 'predictions.npy'), preds)
        np.save(os.path.join(self.args.output_dir, 'labels.npy'), labels)
        
        # 保存特征（如果有）
        if len(features) > 0:
            np.save(os.path.join(self.args.output_dir, 'features.npy'), np.array(features))
            print(f"特征保存到: features.npy")


def parse_args():
    parser = argparse.ArgumentParser(description='GPU评估脚本（修复版）')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='输出目录')
    
    # 功能开关
    parser.add_argument('--extract_features', action='store_true',
                        help='是否提取特征')
    parser.add_argument('--compute_topk', action='store_true',
                        help='是否计算Top-k准确率')
    parser.add_argument('--compute_per_class', action='store_true',
                        help='是否计算每个类别的指标')
    parser.add_argument('--plot_confusion_matrix', action='store_true',
                        help='是否绘制混淆矩阵')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='是否使用多GPU')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    try:
        # 创建评估器
        evaluator = GPUEvaluator(args)
        
        # 执行评估
        results = evaluator.evaluate()
        
        print("\n" + "="*60)
        print("✓ 评估完成！")
        print("="*60)
        print(f"所有结果已保存到: {args.output_dir}")
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == '__main__':
    main()