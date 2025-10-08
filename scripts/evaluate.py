# scripts/evaluate.py
"""
多GPU模型评估脚本 - 优化版
支持完整的评估指标和可视化
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
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 统一导入
from models import MultiModalPestDetection
from data import create_dataloaders
from utils import (calculate_metrics, calculate_per_class_metrics, 
                   topk_accuracy, plot_confusion_matrix, 
                   plot_per_class_metrics, plot_feature_distribution)


class Evaluator:
    """优化的评估器"""
    
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.load_checkpoint()
        self.setup_dataloader()
        self.setup_model()
        os.makedirs(args.output_dir, exist_ok=True)
    
    def setup_device(self):
        """设置设备"""
        if not torch.cuda.is_available():
            raise RuntimeError("需要GPU进行评估")
        
        self.device = torch.device('cuda:0')
        self.num_gpus = torch.cuda.device_count()
        
        print("\n" + "="*60)
        print("评估器初始化")
        print("="*60)
        print(f"使用设备: {self.device}")
        print(f"可用GPU: {self.num_gpus}")
        
        torch.cuda.empty_cache()
    
    def load_checkpoint(self):
        """安全加载检查点"""
        print(f"\n加载检查点: {self.args.checkpoint}")
        
        try:
            self.checkpoint = torch.load(
                self.args.checkpoint,
                map_location='cpu',
                weights_only=False
            )
        except TypeError:
            # PyTorch < 2.6
            self.checkpoint = torch.load(
                self.args.checkpoint,
                map_location='cpu'
            )
        except Exception as e:
            # 添加numpy安全类
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
        if 'epoch' in self.checkpoint:
            print(f"  训练轮数: {self.checkpoint['epoch']}")
        if 'metrics' in self.checkpoint:
            metrics = self.checkpoint['metrics']
            if isinstance(metrics, dict) and 'val_acc' in metrics:
                print(f"  验证准确率: {metrics['val_acc']:.4f}")
    
    def setup_dataloader(self):
        """创建数据加载器"""
        print(f"\n创建数据加载器...")
        
        _, _, self.test_loader = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size * self.num_gpus,
            num_workers=self.args.num_workers,
            text_model_name=self.model_args.get('text_model_name', 'bert-base-chinese'),
            use_augmentation=False
        )
        
        self.num_classes = self.test_loader.dataset.num_classes
        self.class_names = list(self.test_loader.dataset.class_to_idx.keys())
        
        print(f"✓ 测试集: {len(self.test_loader.dataset)} 样本")
        print(f"✓ 类别数: {self.num_classes}")
    
    def setup_model(self):
        """创建并加载模型"""
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
            llm_model_name=None,
            use_lora=False,
            freeze_encoders=False
        )
        
        # 加载权重
        state_dict = self.checkpoint['model_state_dict']
        
        # 处理权重键名
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除module.前缀
            new_key = key.replace('module.', '')
            
            # 跳过LLM相关
            if not new_key.startswith(('llm.', 'llm_adapter.')):
                new_state_dict[new_key] = value
        
        # 加载
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ 加载了 {len(new_state_dict)} 个权重")
        
        # 移到GPU
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 多GPU
        if self.num_gpus > 1:
            print(f"✓ 使用DataParallel ({self.num_gpus} GPUs)")
            self.model = DataParallel(self.model)
        
        # 参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ 模型参数: {total_params/1e6:.2f}M")
    
    @torch.no_grad()
    def evaluate(self):
        """执行评估"""
        print("\n" + "="*60)
        print("开始评估")
        print("="*60)
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        pbar = tqdm(self.test_loader, desc='评估进度', ncols=100)
        
        for batch in pbar:
            # 移动数据
            rgb = batch['rgb_images'].to(self.device, non_blocking=True)
            hsi = batch['hsi_images'].to(self.device, non_blocking=True)
            text_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            text_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # 前向传播
            outputs = self.model(
                rgb, hsi, text_ids, text_mask,
                labels=None,
                return_features=self.args.extract_features
            )
            
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if self.args.extract_features and 'fused_features' in outputs:
                all_features.extend(outputs['fused_features'].cpu().numpy())
            
            # 更新进度
            acc = np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix({'acc': f'{acc:.3f}'})
        
        # 转换
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        results = self.compute_metrics(all_preds, all_labels, all_probs)
        
        # 可视化
        self.visualize_results(all_preds, all_labels, results)
        
        # 特征可视化
        if len(all_features) > 0:
            self.visualize_features(np.array(all_features), all_labels)
        
        # 保存结果
        self.save_results(results, all_preds, all_labels)
        
        return results
    
    def compute_metrics(self, preds, labels, probs):
        """计算评估指标"""
        print("\n" + "="*60)
        print("计算评估指标")
        print("="*60)
        
        # 基础指标
        metrics = calculate_metrics(labels, preds, probs, self.num_classes)
        
        print(f"\n总体性能:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        
        # Top-k
        if self.args.compute_topk:
            top1, top5 = topk_accuracy(
                torch.from_numpy(probs),
                torch.from_numpy(labels),
                topk=(1, 5)
            )
            print(f"  Top-1: {top1:.2f}%")
            print(f"  Top-5: {top5:.2f}%")
            metrics['top1'] = top1
            metrics['top5'] = top5
        
        # 每类指标
        per_class = calculate_per_class_metrics(labels, preds, self.class_names)
        
        print(f"\n各类别性能 (Top 10):")
        print("-" * 60)
        print(f"{'类别':<20} {'精确率':<10} {'召回率':<10} {'F1':<10}")
        print("-" * 60)
        
        for i, name in enumerate(self.class_names[:10]):
            if name in per_class:
                m = per_class[name]
                print(f"{name:<20} {m['precision']:<10.3f} "
                      f"{m['recall']:<10.3f} {m['f1-score']:<10.3f}")
        
        return {
            'overall': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1'])
            },
            'per_class': per_class,
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
    
    def visualize_results(self, preds, labels, results):
        """可视化结果"""
        print("\n生成可视化图表...")
        
        # 混淆矩阵
        if self.args.plot_confusion_matrix:
            cm_path = os.path.join(self.args.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(
                labels, preds,
                class_names=self.class_names,
                save_path=cm_path,
                normalize=True,
                figsize=(14, 12)
            )
            print(f"  ✓ 混淆矩阵: {cm_path}")
        
        # 每类指标
        if self.args.plot_per_class:
            per_class = results['per_class']
            
            precision = [per_class[name]['precision'] for name in self.class_names if name in per_class]
            recall = [per_class[name]['recall'] for name in self.class_names if name in per_class]
            f1 = [per_class[name]['f1-score'] for name in self.class_names if name in per_class]
            
            metrics_dict = {
                'precision': np.array(precision),
                'recall': np.array(recall),
                'f1': np.array(f1)
            }
            
            pc_path = os.path.join(self.args.output_dir, 'per_class_metrics.png')
            plot_per_class_metrics(
                metrics_dict,
                [name for name in self.class_names if name in per_class],
                save_path=pc_path,
                figsize=(16, 8)
            )
            print(f"  ✓ 类别指标: {pc_path}")
    
    def visualize_features(self, features, labels):
        """特征可视化"""
        print("\n生成特征分布图...")
        
        # t-SNE
        tsne_path = os.path.join(self.args.output_dir, 'features_tsne.png')
        plot_feature_distribution(
            features, labels,
            method='tsne',
            save_path=tsne_path,
            title='t-SNE特征分布'
        )
        print(f"  ✓ t-SNE: {tsne_path}")
        
        # PCA
        pca_path = os.path.join(self.args.output_dir, 'features_pca.png')
        plot_feature_distribution(
            features, labels,
            method='pca',
            save_path=pca_path,
            title='PCA特征分布'
        )
        print(f"  ✓ PCA: {pca_path}")
    
    def save_results(self, results, preds, labels):
        """保存结果"""
        # JSON结果
        json_path = os.path.join(self.args.output_dir, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 预测结果
        np.save(os.path.join(self.args.output_dir, 'predictions.npy'), preds)
        np.save(os.path.join(self.args.output_dir, 'labels.npy'), labels)
        
        print(f"\n✓ 结果保存到: {self.args.output_dir}")
        
        # 生成报告
        self.generate_report(results)
    
    def generate_report(self, results):
        """生成Markdown报告"""
        report_path = os.path.join(self.args.output_dir, 'report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 多模态病虫害识别评估报告\n\n")
            
            # 基本信息
            f.write("## 基本信息\n\n")
            f.write(f"- 检查点: `{self.args.checkpoint}`\n")
            f.write(f"- 数据集: `{self.args.data_root}`\n")
            f.write(f"- 测试样本: {len(self.test_loader.dataset)}\n")
            f.write(f"- 类别数: {self.num_classes}\n")
            f.write(f"- GPU数量: {self.num_gpus}\n\n")
            
            # 总体性能
            f.write("## 总体性能\n\n")
            overall = results['overall']
            f.write("| 指标 | 值 |\n")
            f.write("|------|----|\n")
            f.write(f"| 准确率 | {overall['accuracy']:.4f} |\n")
            f.write(f"| 精确率 | {overall['precision']:.4f} |\n")
            f.write(f"| 召回率 | {overall['recall']:.4f} |\n")
            f.write(f"| F1分数 | {overall['f1']:.4f} |\n\n")
            
            # 各类别性能
            f.write("## 各类别性能\n\n")
            f.write("| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            
            per_class = results['per_class']
            for name in self.class_names:
                if name in per_class:
                    m = per_class[name]
                    f.write(f"| {name} | {m['precision']:.3f} | "
                           f"{m['recall']:.3f} | {m['f1-score']:.3f} | "
                           f"{int(m['support'])} |\n")
            
            f.write("\n## 可视化\n\n")
            if self.args.plot_confusion_matrix:
                f.write("- [混淆矩阵](confusion_matrix.png)\n")
            if self.args.plot_per_class:
                f.write("- [类别指标](per_class_metrics.png)\n")
            if self.args.extract_features:
                f.write("- [t-SNE分布](features_tsne.png)\n")
                f.write("- [PCA分布](features_pca.png)\n")
        
        print(f"✓ 报告生成: {report_path}")


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(
        description='多模态病虫害识别评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
                       help='提取特征')
    parser.add_argument('--compute_topk', action='store_true',
                       help='计算Top-k准确率')
    parser.add_argument('--plot_confusion_matrix', action='store_true',
                       help='绘制混淆矩阵')
    parser.add_argument('--plot_per_class', action='store_true',
                       help='绘制类别指标')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 配置
    torch.backends.cudnn.benchmark = True
    
    try:
        # 创建评估器
        evaluator = Evaluator(args)
        
        # 执行评估
        results = evaluator.evaluate()
        
        print("\n" + "="*60)
        print("✓ 评估完成！")
        print("="*60)
        print(f"准确率: {results['overall']['accuracy']:.4f}")
        print(f"F1分数: {results['overall']['f1']:.4f}")
        print(f"输出目录: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()