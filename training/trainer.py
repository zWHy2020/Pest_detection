# training/trainer.py
"""
训练器类
封装完整的训练、验证和测试流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, List
from tqdm import tqdm
import os
import json
from datetime import datetime
import numpy as np

from utils.metrics import AverageMeter, calculate_metrics, MetricsTracker
from utils.visualization import plot_training_curves


class Trainer:
    """
    通用训练器类
    支持混合精度训练、梯度累积、学习率调度等
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device: str = 'cuda',
        output_dir: str = './outputs',
        exp_name: str = 'experiment',
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
        log_interval: int = 50,
        save_interval: int = 10,
        early_stopping_patience: int = 20
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            output_dir: 输出目录
            exp_name: 实验名称
            use_amp: 是否使用混合精度
            gradient_accumulation_steps: 梯度累积步数
            gradient_clip: 梯度裁剪阈值
            log_interval: 日志记录间隔
            save_interval: 模型保存间隔
            early_stopping_patience: 早停耐心值
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 输出配置
        self.output_dir = os.path.join(output_dir, exp_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 训练配置
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.early_stopping_patience = early_stopping_patience
        
        # 混合精度
        self.scaler = GradScaler() if use_amp else None
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'logs'))
        
        # 指标追踪
        self.metrics_tracker = MetricsTracker()
        
        # 最佳指标
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            rgb_images = batch['rgb_images'].to(self.device)
            hsi_images = batch['hsi_images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播（混合精度）
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        rgb_images, hsi_images,
                        text_input_ids, text_attention_mask,
                        labels=labels
                    )
                    loss = outputs['total_loss']
                    loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    rgb_images, hsi_images,
                    text_input_ids, text_attention_mask,
                    labels=labels
                )
                loss = outputs['total_loss']
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # 梯度裁剪
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 梯度裁剪
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    
                    # 更新参数
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 更新统计
            losses.update(loss.item() * self.gradient_accumulation_steps, rgb_images.size(0))
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
            
            # 记录到TensorBoard
            if batch_idx % self.log_interval == 0:
                self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                self.writer.add_scalar('Train/LR', 
                                      self.optimizer.param_groups[0]['lr'], 
                                      self.global_step)
        
        return {'train_loss': losses.avg}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        for batch in pbar:
            rgb_images = batch['rgb_images'].to(self.device)
            hsi_images = batch['hsi_images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                rgb_images, hsi_images,
                text_input_ids, text_attention_mask,
                labels=labels
            )
            
            loss = outputs['total_loss']
            logits = outputs['logits']
            
            losses.update(loss.item(), rgb_images.size(0))
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # 计算指标
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', losses.avg, epoch)
        self.writer.add_scalar('Val/Accuracy', metrics['accuracy'], epoch)
        self.writer.add_scalar('Val/F1', metrics['f1'], epoch)
        
        return {
            'val_loss': losses.avg,
            'val_acc': metrics['accuracy'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'val_f1': metrics['f1']
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存周期性检查点
        if epoch % self.save_interval == 0:
            epoch_path = os.path.join(
                self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save(checkpoint, epoch_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型 (Acc: {metrics['val_acc']:.4f})")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """完整训练流程"""
        start_epoch = 1
        
        # 恢复训练
        if resume_from:
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"从epoch {start_epoch-1}恢复训练")
        
        print(f"\n{'='*60}")
        print(f"开始训练")
        print(f"{'='*60}")
        print(f"输出目录: {self.output_dir}")
        print(f"总epochs: {num_epochs}")
        print(f"设备: {self.device}")
        print(f"混合精度: {self.use_amp}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 合并指标
            all_metrics = {**train_metrics, **val_metrics}
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 更新指标追踪
            self.metrics_tracker.update(all_metrics)
            
            # 打印结果
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  Val F1: {val_metrics['val_f1']:.4f}")
            
            # 检查是否为最佳模型
            is_best = val_metrics['val_acc'] > self.best_val_acc
            
            if is_best:
                self.best_val_acc = val_metrics['val_acc']
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, all_metrics, is_best)
            
            # 早停
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n早停触发！{self.early_stopping_patience}个epoch无改进")
                break
        
        # 保存训练历史
        history_path = os.path.join(self.output_dir, 'training_history.json')
        self.metrics_tracker.save(history_path)
        
        # 绘制训练曲线
        curves_path = os.path.join(self.output_dir, 'training_curves.png')
        plot_training_curves(self.metrics_tracker.history, save_path=curves_path)
        
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"{'='*60}")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"最佳模型保存在: {os.path.join(self.checkpoint_dir, 'best_model.pth')}")
        print(f"{'='*60}\n")
        
        self.writer.close()