# scripts/train_qwen.py
"""
Qwen 增强模型训练脚本
优化显存使用和训练效率
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.main_model_qwen import MultiModalPestDetectionWithQwen
from data import create_dataloaders
from utils import calculate_metrics, AverageMeter, plot_training_curves


class QwenTrainer:
    """Qwen 模型训练器"""
    
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.setup_dataloader()
        self.setup_model()
        self.setup_optimizer()
        self.setup_training()
        
    def setup_device(self):
        """设置设备"""
        if not torch.cuda.is_available():
            raise RuntimeError("需要 GPU 进行训练")
        
        self.device = torch.device('cuda:0')
        self.num_gpus = torch.cuda.device_count()
        
        print("\n" + "="*60)
        print("训练器初始化")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"GPU 数量: {self.num_gpus}")
        
        # 清理显存
        torch.cuda.empty_cache()
        
    def setup_dataloader(self):
        """创建数据加载器"""
        print("\n创建数据加载器...")
        
        self.train_loader, self.val_loader, _ = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            text_model_name=self.args.text_model_name,
            use_augmentation=True
        )
        
        self.num_classes = self.train_loader.dataset.num_classes
        
        print(f"✓ 训练集: {len(self.train_loader.dataset)} 样本")
        print(f"✓ 验证集: {len(self.val_loader.dataset)} 样本")
        print(f"✓ 类别数: {self.num_classes}")
        
    def setup_model(self):
        """创建模型"""
        print("\n创建模型...")
        
        self.model = MultiModalPestDetectionWithQwen(
            num_classes=self.num_classes,
            rgb_image_size=224,
            hsi_image_size=64,
            hsi_channels=224,
            text_model_name=self.args.text_model_name,
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            fusion_layers=4,
            fusion_strategy='hierarchical',
            qwen_path=self.args.qwen_path,
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            num_query_tokens=32,
            freeze_encoders=False
        )
        
        # 移到设备
        self.model = self.model.to(self.device)
        
        # 多GPU
        if self.num_gpus > 1:
            print(f"✓ 使用 DataParallel ({self.num_gpus} GPUs)")
            self.model = nn.DataParallel(self.model)
        
        print("✓ 模型创建完成")
    
    def setup_optimizer(self):
        """设置优化器"""
        print("\n设置优化器...")
        
        # 分组参数
        encoder_params = []
        fusion_params = []
        adapter_params = []
        qwen_params = []
        classifier_params = []
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'fusion' in name or 'alignment' in name:
                fusion_params.append(param)
            elif 'embedding_adapter' in name:
                adapter_params.append(param)
            elif 'qwen' in name:
                qwen_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
        
        # 不同学习率
        param_groups = [
            {'params': encoder_params, 'lr': self.args.lr * 0.1, 'name': 'encoders'},
            {'params': fusion_params, 'lr': self.args.lr * 0.5, 'name': 'fusion'},
            {'params': adapter_params, 'lr': self.args.lr, 'name': 'adapter'},
            {'params': qwen_params, 'lr': self.args.lr * 0.1, 'name': 'qwen'},
            {'params': classifier_params, 'lr': self.args.lr, 'name': 'classifier'}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.args.lr * 0.01
        )
        
        print("✓ 优化器设置完成")
        
        # 打印参数组
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            print(f"  {group['name']}: {num_params/1e6:.2f}M 参数, lr={group['lr']:.2e}")
    
    def setup_training(self):
        """设置训练组件"""
        # 混合精度
        self.scaler = GradScaler() if self.args.use_amp else None
        
        # TensorBoard
        log_dir = os.path.join(self.args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # 检查点目录
        self.checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 最佳指标
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.global_step = 0
        
        # 历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        
        losses = AverageMeter()
        cls_losses = AverageMeter()
        align_losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            rgb = batch['rgb_images'].to(self.device, non_blocking=True)
            hsi = batch['hsi_images'].to(self.device, non_blocking=True)
            text_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            text_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # 前向传播
            if self.args.use_amp:
                with autocast():
                    outputs = self.model(rgb, hsi, text_ids, text_mask, labels=labels)
                    loss = outputs['total_loss']
                    if isinstance(loss, tuple):
                        loss = loss[0]
                    loss = loss.mean()
                
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
                outputs = self.model(rgb, hsi, text_ids, text_mask, labels=labels)
                loss = outputs['total_loss']
                if isinstance(loss, tuple):
                    loss = loss[0]
                loss = loss.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad
                    )
                
                self.optimizer.step()
            
            # 统计
            losses.update(loss.item(), rgb.size(0))
            
            cls_loss = outputs['cls_loss']
            if isinstance(cls_loss, tuple):
                cls_loss = cls_loss[0]
            cls_losses.update(cls_loss.mean().item(), rgb.size(0))
            
            align_loss = outputs['alignment_loss']
            if isinstance(align_loss, tuple):
                align_loss = align_loss[0]
            align_losses.update(align_loss.mean().item(), rgb.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.3f}',
                'cls': f'{cls_losses.avg:.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # TensorBoard
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                self.writer.add_scalar('Train/ClsLoss', cls_losses.avg, self.global_step)
                self.writer.add_scalar('Train/AlignLoss', align_losses.avg, self.global_step)
            
            self.global_step += 1
            
            # 清理显存
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Val {epoch}', ncols=100)
        
        for batch in pbar:
            rgb = batch['rgb_images'].to(self.device, non_blocking=True)
            hsi = batch['hsi_images'].to(self.device, non_blocking=True)
            text_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            text_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            outputs = self.model(rgb, hsi, text_ids, text_mask, labels=labels)
            
            loss = outputs['total_loss']
            if isinstance(loss, tuple):
                loss = loss[0]
            loss = loss.mean()
            
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1)
            
            losses.update(loss.item(), rgb.size(0))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{losses.avg:.3f}'})
        
        # 计算指标
        import numpy as np
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            num_classes=self.num_classes
        )
        
        return losses.avg, metrics['accuracy'], metrics['f1']
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc
        }
        
        # 保存最新
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"   ✓ 保存最佳模型 (Acc: {val_acc:.4f})")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print("-" * 60)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_f1 = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # TensorBoard
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Val/F1', val_f1, epoch)
            
            # 打印
            print(f"\n结果:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
            print(f"  Val F1: {val_f1:.4f}")
            
            # 保存
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # 早停
            if epoch - self.best_epoch >= self.args.early_stopping:
                print(f"\n早停触发 (最佳 epoch: {self.best_epoch})")
                break
        
        # 保存历史
        history_path = os.path.join(self.args.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 绘制曲线
        curves_path = os.path.join(self.args.output_dir, 'curves.png')
        plot_training_curves(self.history, save_path=curves_path)
        
        print("\n" + "="*60)
        print("✓ 训练完成!")
        print("="*60)
        print(f"最佳准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"输出目录: {self.args.output_dir}")
        print("="*60)
        
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 模型
    parser.add_argument('--qwen_path', type=str, default='./models/qwen2.5-7b')
    parser.add_argument('--text_model_name', type=str, default='bert-base-chinese')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--early_stopping', type=int, default=10)
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs/qwen')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 训练
    trainer = QwenTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()