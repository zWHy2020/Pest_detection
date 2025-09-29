# scripts/train.py
"""
多模态病虫害识别模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.main_model import MultiModalPestDetection
from data.dataset import create_dataloaders
from utils.metrics import calculate_metrics, AverageMeter
from utils.visualization import plot_confusion_matrix, plot_training_curves


class Trainer:
    """训练器类"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'logs'))
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # 创建数据加载器
        print("创建数据加载器...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text_model_name=args.text_model_name,
            use_augmentation=args.use_augmentation
        )
        
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        
        # 创建模型
        print("创建模型...")
        self.model = MultiModalPestDetection(
            num_classes=self.train_loader.dataset.num_classes,
            rgb_image_size=args.rgb_size,
            hsi_image_size=args.hsi_size,
            hsi_channels=args.hsi_channels,
            text_model_name=args.text_model_name,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            fusion_layers=args.fusion_layers,
            fusion_strategy=args.fusion_strategy,
            llm_model_name=args.llm_model_name if args.use_llm else None,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_encoders=args.freeze_encoders
        )
        
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"总参数: {total_params / 1e6:.2f}M")
        print(f"可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"可训练比例: {100 * trainable_params / total_params:.2f}%\n")
        
        # 优化器
        if args.use_lora:
            # 只优化LoRA参数
            from models.adapters.lora import get_lora_parameters
            lora_params = get_lora_parameters(self.model)
            other_params = [p for p in self.model.parameters() 
                          if p.requires_grad and p not in lora_params]
            
            self.optimizer = optim.AdamW([
                {'params': lora_params, 'lr': args.lr * 10},  # LoRA参数使用更高学习率
                {'params': other_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        
        # 最佳指标
        self.best_acc = 0.0
        self.best_f1 = 0.0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        cls_losses = AverageMeter()
        align_losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            rgb_images = batch['rgb_images'].to(self.device)
            hsi_images = batch['hsi_images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 混合精度训练
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        rgb_images, hsi_images,
                        text_input_ids, text_attention_mask,
                        labels=labels
                    )
                    loss = outputs['total_loss']
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                if self.args.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    rgb_images, hsi_images,
                    text_input_ids, text_attention_mask,
                    labels=labels
                )
                loss = outputs['total_loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad
                    )
                
                self.optimizer.step()
            
            # 更新统计
            losses.update(loss.item(), rgb_images.size(0))
            cls_losses.update(outputs['cls_loss'].item(), rgb_images.size(0))
            align_losses.update(outputs['alignment_loss'].item(), rgb_images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'cls': f'{cls_losses.avg:.4f}',
                'align': f'{align_losses.avg:.4f}'
            })
            
            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('Train/Loss', losses.avg, global_step)
                self.writer.add_scalar('Train/ClsLoss', cls_losses.avg, global_step)
                self.writer.add_scalar('Train/AlignLoss', align_losses.avg, global_step)
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
        
        for batch in pbar:
            rgb_images = batch['rgb_images'].to(self.device)
            hsi_images = batch['hsi_images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
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
            np.array(all_preds),
            num_classes=self.model.num_classes
        )
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', losses.avg, epoch)
        self.writer.add_scalar('Val/Accuracy', metrics['accuracy'], epoch)
        self.writer.add_scalar('Val/Precision', metrics['precision'], epoch)
        self.writer.add_scalar('Val/Recall', metrics['recall'], epoch)
        self.writer.add_scalar('Val/F1', metrics['f1'], epoch)
        
        print(f"\n验证结果:")
        print(f"Loss: {losses.avg:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}\n")
        
        return metrics, all_preds, all_labels
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'args': vars(self.args)
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(
                self.output_dir, 'checkpoints', 'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到 {best_path}")
    
    def train(self):
        """完整训练流程"""
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"实验名称: {self.args.exp_name}")
        print(f"输出目录: {self.output_dir}\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            metrics, preds, labels = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LR', current_lr, epoch)
            
            # 保存检查点
            is_best = metrics['accuracy'] > self.best_acc
            if is_best:
                self.best_acc = metrics['accuracy']
                self.best_f1 = metrics['f1']
            
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best)
            
            # 保存混淆矩阵
            if epoch % self.args.plot_interval == 0:
                cm_path = os.path.join(
                    self.output_dir, f'confusion_matrix_epoch_{epoch}.png'
                )
                plot_confusion_matrix(
                    labels, preds,
                    class_names=list(self.train_loader.dataset.class_to_idx.keys()),
                    save_path=cm_path
                )
        
        print(f"\n训练完成！")
        print(f"最佳准确率: {self.best_acc:.4f}")
        print(f"最佳F1分数: {self.best_f1:.4f}")
        
        self.writer.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态病虫害识别训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='是否使用数据增强')
    
    # 模型参数
    parser.add_argument('--rgb_size', type=int, default=224,
                        help='RGB图像大小')
    parser.add_argument('--hsi_size', type=int, default=64,
                        help='HSI图像大小')
    parser.add_argument('--hsi_channels', type=int, default=224,
                        help='HSI波段数')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='嵌入维度')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')
    parser.add_argument('--fusion_layers', type=int, default=4,
                        help='融合层数')
    parser.add_argument('--fusion_strategy', type=str, default='hierarchical',
                        choices=['concat', 'gated', 'hierarchical'],
                        help='融合策略')
    
    # 文本编码器参数
    parser.add_argument('--text_model_name', type=str, default='bert-base-chinese',
                        help='文本编码器模型名称')
    
    # LLM参数
    parser.add_argument('--use_llm', action='store_true',
                        help='是否使用大语言模型')
    parser.add_argument('--llm_model_name', type=str, default=None,
                        help='大语言模型名称')
    parser.add_argument('--use_lora', action='store_true',
                        help='是否使用LoRA')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=float, default=16,
                        help='LoRA alpha')
    parser.add_argument('--freeze_encoders', action='store_true',
                        help='是否冻结编码器')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='标签平滑')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--use_amp', action='store_true',
                        help='是否使用混合精度训练')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存间隔(epoch)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='日志间隔(iter)')
    parser.add_argument('--plot_interval', type=int, default=10,
                        help='绘图间隔(epoch)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()