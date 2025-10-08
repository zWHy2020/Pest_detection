# scripts/evaluate_multi_gpu.py
"""
多GPU模型评估脚本
支持DataParallel和模型权重过滤，避免GPU内存溢出
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
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.main_model import MultiModalPestDetection
from data.dataset import create_dataloaders
from utils.metrics import calculate_metrics, calculate_per_class_metrics, topk_accuracy
from utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_feature_distribution
)


class MultiGPUEvaluator:
    """多GPU评估器类"""
    def __init__(self, args):
        self.args = args
        
        # 设置GPU设备
        self.setup_devices()
        
        print("="*60)
        print("多GPU评估器初始化")
        print("="*60)
        print(f"检查点路径: {args.checkpoint}")
        print(f"数据根目录: {args.data_root}")
        
        # 加载检查点（先加载到CPU）
        self.load_checkpoint()
        
        # 创建数据加载器
        self.setup_dataloader()
        
        # 创建模型
        self.setup_model()
        
        # 创建输出目录
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("="*60)
        print("初始化完成！")
        print("="*60 + "\n")
    
    def setup_devices(self):
        """设置GPU设备"""
        if torch.cuda.is_available():
            # 获取所有可用GPU
            self.num_gpus = torch.cuda.device_count()
            self.gpu_ids = list(range(self.num_gpus))
            self.device = torch.device('cuda:0')  # 主设备
            
            print(f"检测到 {self.num_gpus} 个GPU:")
            for i in self.gpu_ids:
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f} GB")
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            print("已清理GPU缓存")
        else:
            self.num_gpus = 0
            self.gpu_ids = []
            self.device = torch.device('cpu')
            print("未检测到GPU，使用CPU进行评估")
    
    def load_checkpoint(self):
        """加载模型检查点"""
        print(f"\n加载检查点: {self.args.checkpoint}")
        
        try:
            # PyTorch 2.6+ 兼容性处理
            # 方法1: 使用weights_only=False（如果信任检查点来源）
            checkpoint = torch.load(self.args.checkpoint, map_location='cpu', weights_only=False)
            print("✓ 检查点加载成功")
            
        except Exception as e:
            print(f"✗ 使用weights_only=False加载失败: {e}")
            
            # 方法2: 添加安全的全局变量
            try:
                import numpy
                torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
                checkpoint = torch.load(self.args.checkpoint, map_location='cpu')
                print("✓ 使用安全全局变量加载成功")
                
            except Exception as e2:
                print(f"✗ 添加安全全局变量失败: {e2}")
                
                # 方法3: 尝试兼容旧版本PyTorch的方式
                try:
                    import pickle
                    with open(self.args.checkpoint, 'rb') as f:
                        checkpoint = pickle.load(f)
                    print("✓ 使用pickle加载成功")
                    
                except Exception as e3:
                    print(f"✗ 检查点加载失败: {e3}")
                    raise RuntimeError(f"无法加载检查点: {self.args.checkpoint}")
        
        # 获取模型配置
        self.model_args = checkpoint.get('args', {})
        self.checkpoint = checkpoint
        
        # 打印检查点信息
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            if isinstance(metrics, dict):
                if 'val_acc' in metrics:
                    print(f"  验证准确率: {metrics['val_acc']:.4f}")
                if 'val_f1' in metrics:
                    print(f"  验证F1: {metrics['val_f1']:.4f}")
        
        # 分析模型权重
        self.analyze_checkpoint_weights()
    
    def analyze_checkpoint_weights(self):
        """分析检查点中的权重"""
        model_state_dict = self.checkpoint['model_state_dict']
        
        # 统计各模块的参数
        module_params = {}
        total_params = 0
        llm_params = 0
        
        for key, value in model_state_dict.items():
            module_name = key.split('.')[0]
            param_count = value.numel()
            total_params += param_count
            
            if module_name not in module_params:
                module_params[module_name] = 0
            module_params[module_name] += param_count
            
            if key.startswith('llm.') or key.startswith('llm_adapter.'):
                llm_params += param_count
        
        print(f"\n检查点权重分析:")
        print(f"  总参数量: {total_params / 1e6:.2f}M")
        print(f"  LLM相关参数: {llm_params / 1e6:.2f}M")
        print(f"  非LLM参数: {(total_params - llm_params) / 1e6:.2f}M")
        
        print(f"\n各模块参数量:")
        for module, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            print(f"    {module}: {params / 1e6:.2f}M")
    
    def setup_dataloader(self):
        """创建数据加载器"""
        print(f"\n创建数据加载器...")
        
        # 根据GPU数量调整batch size
        effective_batch_size = self.args.batch_size
        if self.num_gpus > 1:
            effective_batch_size = self.args.batch_size * self.num_gpus
            print(f"  使用{self.num_gpus}个GPU，有效batch size: {effective_batch_size}")
        
        # 创建数据加载器
        _, _, self.test_loader = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=effective_batch_size,
            num_workers=self.args.num_workers,
            text_model_name=self.model_args.get('text_model_name', 'bert-base-chinese'),
            use_augmentation=False
        )
        
        self.num_classes = self.test_loader.dataset.num_classes
        self.class_names = list(self.test_loader.dataset.class_to_idx.keys())
        
        print(f"  测试集大小: {len(self.test_loader.dataset)}")
        print(f"  类别数: {self.num_classes}")
        print(f"  批次数: {len(self.test_loader)}")
    
    def setup_model(self):
        """创建并配置模型"""
        print(f"\n创建模型...")
        
        # 创建模型（不加载LLM以节省内存）
        self.model = MultiModalPestDetection(
            num_classes=self.num_classes,
            rgb_image_size=self.model_args.get('rgb_size', 224),
            hsi_image_size=self.model_args.get('hsi_size', 64),
            hsi_channels=self.model_args.get('hsi_channels', 224),
            text_model_name=self.model_args.get('text_model_name', 'bert-base-chinese'),
            embed_dim=self.model_args.get('embed_dim', 768),
            num_heads=self.model_args.get('num_heads', 12),
            dropout=0.0,  # 评估时不使用dropout
            fusion_layers=self.model_args.get('fusion_layers', 4),
            fusion_strategy=self.model_args.get('fusion_strategy', 'hierarchical'),
            llm_model_name=None,  # 不加载LLM
            use_lora=False,
            freeze_encoders=False
        )
        
        # 加载模型权重（过滤LLM相关权重）
        self.load_model_weights()
        
        # 配置多GPU
        if self.num_gpus > 1:
            print(f"\n配置DataParallel (使用GPU: {self.gpu_ids})")
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 打印模型信息
        self.print_model_info()
    
    def load_model_weights(self):
        """加载模型权重（过滤不需要的权重）"""
        print(f"  加载模型权重...")
        
        model_state_dict = self.checkpoint['model_state_dict']
        
        # 过滤掉LLM相关的权重
        filtered_state_dict = {}
        skipped_keys = []
        loaded_keys = []
        
        for key, value in model_state_dict.items():
            # 跳过LLM相关权重
            if key.startswith('llm.') or key.startswith('llm_adapter.'):
                skipped_keys.append(key)
                continue
            
            # 检查权重是否与模型匹配
            if key in self.model.state_dict():
                if value.shape == self.model.state_dict()[key].shape:
                    filtered_state_dict[key] = value
                    loaded_keys.append(key)
                else:
                    print(f"    警告: 权重形状不匹配 {key}")
                    print(f"      期望: {self.model.state_dict()[key].shape}")
                    print(f"      实际: {value.shape}")
            else:
                # 可能是因为模型结构略有不同
                pass
        
        # 加载过滤后的权重
        missing_keys, unexpected_keys = self.model.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        print(f"  加载了 {len(loaded_keys)} 个权重")
        print(f"  跳过了 {len(skipped_keys)} 个LLM相关权重")
        
        if missing_keys:
            print(f"  缺失的键 (前10个): {missing_keys[:10]}")
        if unexpected_keys:
            print(f"  意外的键 (前10个): {unexpected_keys[:10]}")
    
    def print_model_info(self):
        """打印模型信息"""
        # 获取实际的模型（如果使用了DataParallel）
        actual_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        
        # 统计参数
        total_params = sum(p.numel() for p in actual_model.parameters())
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        
        print(f"\n模型信息:")
        print(f"  总参数: {total_params / 1e6:.2f}M")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
        
        # 打印各模块参数
        module_params = {}
        for name, module in actual_model.named_children():
            params = sum(p.numel() for p in module.parameters())
            module_params[name] = params
        
        print(f"\n  各模块参数:")
        for name, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name}: {params / 1e6:.2f}M")
    
    @torch.no_grad()
    def evaluate(self):
        """执行模型评估"""
        print("\n" + "="*60)
        print("开始评估")
        print("="*60)
        
        # 初始化结果容器
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        extract_features = self.args.extract_features
        
        # 创建进度条
        pbar = tqdm(self.test_loader, desc='评估进度', ncols=100)
        
        # 评估循环
        for batch_idx, batch in enumerate(pbar):
            try:
                # 移动数据到设备
                rgb_images = batch['rgb_images'].to(self.device, non_blocking=True)
                hsi_images = batch['hsi_images'].to(self.device, non_blocking=True)
                text_input_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
                text_attention_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = self.model(
                    rgb_images, hsi_images,
                    text_input_ids, text_attention_mask,
                    return_features=extract_features
                )
                
                # 获取预测结果
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # 收集结果
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # 收集特征（如果需要）
                if extract_features and 'fused_features' in outputs:
                    features = outputs['fused_features']
                    all_features.extend(features.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'batch': f'{batch_idx + 1}/{len(self.test_loader)}',
                    'acc': f'{(np.array(all_preds) == np.array(all_labels)).mean():.3f}'
                })
                
                # 定期清理GPU缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\nGPU内存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算评估指标
        results = self.compute_metrics(all_preds, all_labels, all_probs)
        
        # 可视化结果
        if self.args.plot_confusion_matrix or self.args.plot_per_class:
            self.visualize_results(all_preds, all_labels, results)
        
        # 特征可视化
        if extract_features and len(all_features) > 0:
            self.visualize_features(np.array(all_features), all_labels)
        
        # 保存结果
        self.save_results(results)
        
        return results
    
    def compute_metrics(self, all_preds, all_labels, all_probs):
        """计算评估指标"""
        print("\n" + "="*60)
        print("计算评估指标")
        print("="*60)
        
        # 基础指标
        metrics = calculate_metrics(
            all_labels, all_preds, all_probs,
            num_classes=self.num_classes
        )
        
        # 打印总体指标
        print(f"\n总体性能指标:")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        
        if 'auc_macro' in metrics:
            print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
            print(f"  AUC (weighted): {metrics['auc_weighted']:.4f}")
        
        # 计算Top-k准确率
        if self.args.compute_topk:
            logits_tensor = torch.from_numpy(all_probs)
            labels_tensor = torch.from_numpy(all_labels)
            top1, top5 = topk_accuracy(logits_tensor, labels_tensor, topk=(1, 5))
            print(f"  Top-1 准确率: {top1:.2f}%")
            print(f"  Top-5 准确率: {top5:.2f}%")
            metrics['top1_accuracy'] = top1
            metrics['top5_accuracy'] = top5
        
        # 计算每个类别的指标
        per_class_report = calculate_per_class_metrics(
            all_labels, all_preds, class_names=self.class_names
        )
        
        # 打印每个类别的性能
        print(f"\n各类别性能:")
        print("-" * 80)
        print(f"{'类别':<20} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'支持数':<10}")
        print("-" * 80)
        
        for class_name in self.class_names:
            if class_name in per_class_report:
                m = per_class_report[class_name]
                print(f"{class_name:<20} "
                      f"{m['precision']:<12.4f} "
                      f"{m['recall']:<12.4f} "
                      f"{m['f1-score']:<12.4f} "
                      f"{int(m['support']):<10}")
        
        # 找出表现最好和最差的类别
        f1_scores = [(name, m['f1-score']) for name, m in per_class_report.items() 
                     if name in self.class_names]
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n表现最好的5个类别:")
        for name, f1 in f1_scores[:5]:
            print(f"  {name}: F1 = {f1:.4f}")
        
        print(f"\n表现最差的5个类别:")
        for name, f1 in f1_scores[-5:]:
            print(f"  {name}: F1 = {f1:.4f}")
        
        # 合并结果
        results = {
            'overall_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
            },
            'per_class_metrics': per_class_report,
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
        
        if 'auc_macro' in metrics:
            results['overall_metrics']['auc_macro'] = float(metrics['auc_macro'])
            results['overall_metrics']['auc_weighted'] = float(metrics['auc_weighted'])
        
        if self.args.compute_topk:
            results['overall_metrics']['top1_accuracy'] = float(top1)
            results['overall_metrics']['top5_accuracy'] = float(top5)
        
        return results
    
    def visualize_results(self, all_preds, all_labels, results):
        """可视化评估结果"""
        print("\n生成可视化图表...")
        
        # 绘制混淆矩阵
        if self.args.plot_confusion_matrix:
            cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(
                all_labels, all_preds,
                class_names=self.class_names,
                save_path=cm_path,
                normalize=True
            )
            print(f"  混淆矩阵已保存: {cm_path}")
        
        # 绘制每个类别的指标
        if self.args.plot_per_class:
            # 从结果中提取每个类别的指标
            per_class_metrics = results['per_class_metrics']
            
            precision_list = []
            recall_list = []
            f1_list = []
            
            for class_name in self.class_names:
                if class_name in per_class_metrics:
                    precision_list.append(per_class_metrics[class_name]['precision'])
                    recall_list.append(per_class_metrics[class_name]['recall'])
                    f1_list.append(per_class_metrics[class_name]['f1-score'])
                else:
                    precision_list.append(0)
                    recall_list.append(0)
                    f1_list.append(0)
            
            metrics_dict = {
                'precision': np.array(precision_list),
                'recall': np.array(recall_list),
                'f1': np.array(f1_list)
            }
            
            per_class_path = os.path.join(self.output_dir, 'per_class_metrics.png')
            plot_per_class_metrics(
                metrics_dict,
                self.class_names,
                save_path=per_class_path
            )
            print(f"  类别指标图已保存: {per_class_path}")
    
    def visualize_features(self, features_array, labels):
        """可视化特征分布"""
        print("\n生成特征分布图...")
        
        # t-SNE可视化
        tsne_path = os.path.join(self.output_dir, 'feature_distribution_tsne.png')
        plot_feature_distribution(
            features_array, labels,
            method='tsne',
            save_path=tsne_path
        )
        print(f"  t-SNE特征分布已保存: {tsne_path}")
        
        # PCA可视化
        pca_path = os.path.join(self.output_dir, 'feature_distribution_pca.png')
        plot_feature_distribution(
            features_array, labels,
            method='pca',
            save_path=pca_path
        )
        print(f"  PCA特征分布已保存: {pca_path}")
    
    def save_results(self, results):
        """保存评估结果"""
        # 保存JSON结果
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {results_path}")
        
        # 生成Markdown报告
        self.generate_report(results)
    
    def generate_report(self, results):
        """生成评估报告"""
        report_path = os.path.join(self.output_dir, 'evaluation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 多模态病虫害识别模型评估报告\n\n")
            
            # 基本信息
            f.write("## 基本信息\n\n")
            f.write(f"- **检查点**: `{self.args.checkpoint}`\n")
            f.write(f"- **数据集**: `{self.args.data_root}`\n")
            f.write(f"- **测试样本数**: {len(self.test_loader.dataset)}\n")
            f.write(f"- **类别数**: {self.num_classes}\n")
            f.write(f"- **GPU数量**: {self.num_gpus}\n\n")
            
            # 总体性能
            f.write("## 总体性能\n\n")
            metrics = results['overall_metrics']
            f.write("| 指标 | 值 |\n")
            f.write("|------|----|\n")
            f.write(f"| 准确率 | {metrics['accuracy']:.4f} |\n")
            f.write(f"| 精确率 | {metrics['precision']:.4f} |\n")
            f.write(f"| 召回率 | {metrics['recall']:.4f} |\n")
            f.write(f"| F1分数 | {metrics['f1']:.4f} |\n")
            
            if 'auc_macro' in metrics:
                f.write(f"| AUC (macro) | {metrics['auc_macro']:.4f} |\n")
            if 'top1_accuracy' in metrics:
                f.write(f"| Top-1 准确率 | {metrics['top1_accuracy']:.2f}% |\n")
                f.write(f"| Top-5 准确率 | {metrics['top5_accuracy']:.2f}% |\n")
            f.write("\n")
            
            # 各类别性能
            f.write("## 各类别性能\n\n")
            f.write("| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            
            per_class = results['per_class_metrics']
            for class_name in self.class_names:
                if class_name in per_class:
                    m = per_class[class_name]
                    f.write(f"| {class_name} | "
                           f"{m['precision']:.4f} | "
                           f"{m['recall']:.4f} | "
                           f"{m['f1-score']:.4f} | "
                           f"{int(m['support'])} |\n")
            
            f.write("\n")
            
            # 图表链接
            f.write("## 可视化结果\n\n")
            if self.args.plot_confusion_matrix:
                f.write("- [混淆矩阵](confusion_matrix.png)\n")
            if self.args.plot_per_class:
                f.write("- [类别指标](per_class_metrics.png)\n")
            if self.args.extract_features:
                f.write("- [t-SNE特征分布](feature_distribution_tsne.png)\n")
                f.write("- [PCA特征分布](feature_distribution_pca.png)\n")
        
        print(f"评估报告已生成: {report_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多GPU病虫害识别模型评估')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    
    # 评估配置
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每个GPU的批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='结果输出目录')
    
    # 可视化选项
    parser.add_argument('--plot_confusion_matrix', action='store_true',
                        help='是否绘制混淆矩阵')
    parser.add_argument('--plot_per_class', action='store_true',
                        help='是否绘制各类别指标')
    parser.add_argument('--extract_features', action='store_true',
                        help='是否提取并可视化特征')
    parser.add_argument('--compute_topk', action='store_true',
                        help='是否计算Top-k准确率')
    
    # GPU配置
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='指定使用的GPU ID，如"0,1"')
    
    args = parser.parse_args()
    
    # 设置CUDA设备
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    return args


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置PyTorch配置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 创建评估器
    evaluator = MultiGPUEvaluator(args)
    
    # 执行评估
    results = evaluator.evaluate()
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print(f"所有结果已保存到: {evaluator.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()






