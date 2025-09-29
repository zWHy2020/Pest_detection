# scripts/evaluate.py
"""
模型评估脚本
评估训练好的模型在测试集上的性能
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.main_model import MultiModalPestDetection
from data.dataset import create_dataloaders
from utils.metrics import calculate_metrics, calculate_per_class_metrics, topk_accuracy
from utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_feature_distribution
)


class Evaluator:
    """评估器类"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 加载检查点
        print(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        
        # 获取配置
        if 'args' in checkpoint:
            model_args = checkpoint['args']
        else:
            print("警告: 检查点中没有保存配置，使用默认配置")
            model_args = {}
        
        # 创建数据加载器
        print("创建数据加载器...")
        _, _, self.test_loader = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text_model_name=model_args.get('text_model_name', 'bert-base-chinese'),
            use_augmentation=False
        )
        
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        
        # 创建模型
        print("创建模型...")
        self.model = MultiModalPestDetection(
            num_classes=self.test_loader.dataset.num_classes,
            rgb_image_size=model_args.get('rgb_size', 224),
            hsi_image_size=model_args.get('hsi_size', 64),
            hsi_channels=model_args.get('hsi_channels', 224),
            text_model_name=model_args.get('text_model_name', 'bert-base-chinese'),
            embed_dim=model_args.get('embed_dim', 768),
            num_heads=model_args.get('num_heads', 12),
            dropout=0.0,
            fusion_layers=model_args.get('fusion_layers', 4),
            fusion_strategy=model_args.get('fusion_strategy', 'hierarchical'),
            llm_model_name=None,
            use_lora=False
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成！\n")
        
        # 类别名称
        self.class_names = list(self.test_loader.dataset.class_to_idx.keys())
        
        # 创建输出目录
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        print("开始评估...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        extract_features = self.args.extract_features
        
        pbar = tqdm(self.test_loader, desc='评估进度')
        
        for batch in pbar:
            rgb_images = batch['rgb_images'].to(self.device)
            hsi_images = batch['hsi_images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                rgb_images, hsi_images,
                text_input_ids, text_attention_mask,
                return_features=extract_features
            )
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if extract_features:
                all_features.extend(outputs['fused_features'].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print("\n" + "="*50)
        print("评估完成！")
        print("="*50)
        
        # 计算指标
        metrics = calculate_metrics(
            all_labels, all_preds, all_probs,
            num_classes=self.model.num_classes
        )
        
        # 打印总体指标
        print(f"\n总体性能指标:")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        if 'auc_macro' in metrics:
            print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
        
        # 计算Top-k准确率
        if self.args.compute_topk:
            logits_tensor = torch.from_numpy(all_probs)
            labels_tensor = torch.from_numpy(all_labels)
            top1, top5 = topk_accuracy(logits_tensor, labels_tensor, topk=(1, 5))
            print(f"  Top-1准确率: {top1:.2f}%")
            print(f"  Top-5准确率: {top5:.2f}%")
        
        # 计算每个类别的指标
        per_class_report = calculate_per_class_metrics(
            all_labels, all_preds, class_names=self.class_names
        )
        
        print(f"\n各类别性能:")
        print("-" * 70)
        print(f"{'类别':<20} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'支持数':<10}")
        print("-" * 70)
        
        for class_name in self.class_names:
            if class_name in per_class_report:
                metrics_class = per_class_report[class_name]
                print(f"{class_name:<20} "
                      f"{metrics_class['precision']:<12.4f} "
                      f"{metrics_class['recall']:<12.4f} "
                      f"{metrics_class['f1-score']:<12.4f} "
                      f"{int(metrics_class['support']):<10}")
        
        # 保存结果
        results = {
            'overall_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1'])
            },
            'per_class_metrics': per_class_report,
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
        
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存到: {results_path}")
        
        # 绘制混淆矩阵
        if self.args.plot_confusion_matrix:
            cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(
                all_labels, all_preds,
                class_names=self.class_names,
                save_path=cm_path,
                normalize=True
            )
        
        # 绘制每个类别的指标
        if self.args.plot_per_class:
            per_class_metrics = {
                'precision': metrics['precision_per_class'],
                'recall': metrics['recall_per_class'],
                'f1': metrics['f1_per_class']
            }
            per_class_path = os.path.join(self.output_dir, 'per_class_metrics.png')
            plot_per_class_metrics(
                per_class_metrics,
                self.class_names,
                save_path=per_class_path
            )
        
        # 可视化特征分布
        if extract_features and len(all_features) > 0:
            features_array = np.array(all_features)
            
            for method in ['tsne', 'pca']:
                feature_path = os.path.join(
                    self.output_dir, f'feature_distribution_{method}.png'
                )
                plot_feature_distribution(
                    features_array, all_labels,
                    method=method,
                    save_path=feature_path
                )
        
        return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态病虫害识别模型评估')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='结果输出目录')
    
    parser.add_argument('--plot_confusion_matrix', action='store_true',
                        help='是否绘制混淆矩阵')
    parser.add_argument('--plot_per_class', action='store_true',
                        help='是否绘制各类别指标')
    parser.add_argument('--extract_features', action='store_true',
                        help='是否提取并可视化特征')
    parser.add_argument('--compute_topk', action='store_true',
                        help='是否计算Top-k准确率')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    evaluator = Evaluator(args)
    results = evaluator.evaluate()
    print("\n评估完成！所有结果已保存。")


if __name__ == '__main__':
    main()