# utils/visualization.py
"""
可视化工具
包含混淆矩阵、训练曲线、注意力图等可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Dict
import torch
from sklearn.metrics import confusion_matrix
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    normalize: bool = True,
    title: str = '混淆矩阵'
):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': '比例' if normalize else '数量'}
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_xlabel('预测标签', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0].set_title('损失曲线', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('损失', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    axes[1].set_title('准确率曲线', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('准确率', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1分数曲线
    if 'val_f1' in history:
        axes[2].plot(epochs, history['val_f1'], 'g-', label='验证F1', linewidth=2)
    if 'train_f1' in history:
        axes[2].plot(epochs, history['train_f1'], 'b-', label='训练F1', linewidth=2)
    axes[2].set_title('F1分数曲线', fontsize=14)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1分数', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.close()


def plot_attention_map(
    attention: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    title: str = '注意力图'
):
    """可视化注意力权重"""
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    
    num_heads = attention.shape[0]
    
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_heads):
        im = axes[i].imshow(attention[i], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Head {i+1}', fontsize=10)
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])
    
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力图已保存到: {save_path}")
    
    plt.close()


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    title: str = '特征分布'
):
    """使用降维方法可视化特征分布"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    features_2d = reducer.fit_transform(features)
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab20',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label='类别')
    plt.title(f'{title} ({method.upper()})', fontsize=16)
    plt.xlabel(f'{method.upper()} 1', fontsize=12)
    plt.ylabel(f'{method.upper()} 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存到: {save_path}")
    
    plt.close()


def plot_per_class_metrics(
    metrics: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """绘制每个类别的性能指标"""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'precision' in metrics:
        ax.bar(x - width, metrics['precision'], width, label='Precision', alpha=0.8)
    if 'recall' in metrics:
        ax.bar(x, metrics['recall'], width, label='Recall', alpha=0.8)
    if 'f1' in metrics:
        ax.bar(x + width, metrics['f1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('各类别性能指标', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"类别指标图已保存到: {save_path}")
    
    plt.close()


def plot_learning_rate_schedule(
    lr_history: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """绘制学习率变化曲线"""
    plt.figure(figsize=figsize)
    plt.plot(lr_history, linewidth=2)
    plt.title('学习率调度', fontsize=14)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习率曲线已保存到: {save_path}")
    
    plt.close()


def visualize_predictions(
    images: np.ndarray,
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10),
    num_samples: int = 12
):
    """可视化预测结果"""
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f'真实: {true_labels[i]}\n预测: {pred_labels[i]}\n置信度: {confidences[i]:.2f}'
        axes[i].set_title(title, fontsize=10, color=color)
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('预测结果可视化', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果已保存到: {save_path}")
    
    plt.close()