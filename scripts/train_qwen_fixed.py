# scripts/train_qwen_fixed.py
"""
修复版Qwen训练脚本 - 解决显存不足问题
关键优化：
1. 梯度累积
2. 更激进的显存管理
3. 梯度检查点
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
import numpy as np
import warnings
import gc
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.main_model_qwen import MultiModalPestDetectionWithQwen
from data import create_dataloaders
from utils import calculate_metrics, AverageMeter, plot_training_curves


class MemoryEfficientQwenTrainer:
    """显存优化的Qwen训练器"""
    
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
            raise RuntimeError("需要GPU")
        
        self.device = torch.device('cuda:0')
        
        print("\n" + "="*60)
        print("显存优化的Qwen训练器")
        print("="*60)
        print(f"设备: {self.device}")
        
        # 🔧 清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 🔧 显存优化设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # 显示显存状态
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"总显存: {total_memory:.2f} GB")
    
    def setup_dataloader(self):
        """创建数据加载器"""
        print("\n创建数据加载器...")
        
        # 🔧 减少workers以节省内存
        self.train_loader, self.val_loader, _ = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=min(self.args.num_workers, 2),  # 最多2个worker
            text_model_name='bert-base-chinese',
            use_hsi=self.args.use_hsi,
            use_augmentation=False  # Qwen训练时不用增强
        )
        
        self.num_classes = self.train_loader.dataset.num_classes
        
        print(f"✓ 训练集: {len(self.train_loader.dataset)} 样本")
        print(f"✓ 验证集: {len(self.val_loader.dataset)} 样本")
        print(f"✓ 类别数: {self.num_classes}")
        print(f"✓ Batch size: {self.args.batch_size}")
        print(f"✓ 梯度累积步数: {self.args.accumulation_steps}")
        print(f"✓ 有效batch size: {self.args.batch_size * self.args.accumulation_steps}")
    
    def setup_model(self):
        """创建模型"""
        print("\n创建Qwen增强模型...")
        
        self.model = MultiModalPestDetectionWithQwen(
            num_classes=self.num_classes,
            rgb_image_size=224,
            hsi_image_size=64,
            hsi_channels=224,
            text_model_name='bert-base-chinese',
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            fusion_layers=4,
            fusion_strategy='hierarchical',
            qwen_path=self.args.qwen_path,
            use_hsi=self.args.use_hsi,
            freeze_encoders=False,
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            num_query_tokens=32
        )

        # 移到设备
        self.model = self.model.to(self.device)
        
        print("✓ 模型创建完成")
        
        # 显示显存使用
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"模型加载后显存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
    
    def setup_optimizer(self):
        """设置优化器"""
        print("\n设置优化器...")
        
        # 分组参数 - 不同模块不同学习率
        encoder_params = []
        fusion_params = []
        adapter_params = []
        qwen_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
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
        
        param_groups = [
            {'params': encoder_params, 'lr': self.args.lr * 0.1},
            {'params': fusion_params, 'lr': self.args.lr * 0.5},
            {'params': adapter_params, 'lr': self.args.lr},
            {'params': qwen_params, 'lr': self.args.lr * 0.1},
            {'params': classifier_params, 'lr': self.args.lr}
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
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.global_step = 0
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch - 带梯度累积和显存优化"""
        self.model.train()
        
        losses = AverageMeter()
        cls_losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', ncols=100)
        
        # 🔧 梯度累积
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 移动数据
                rgb = batch['rgb_images'].to(self.device, non_blocking=True)
                hsi = batch['hsi_images'].to(self.device, non_blocking=True)
                text_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
                text_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # 🔧 使用混合精度
                with autocast(dtype=torch.float16):
                    outputs = self.model(rgb, hsi, text_ids, text_mask, labels=labels)
                    loss = outputs['total_loss']
                    
                    if isinstance(loss, tuple):
                        loss = loss[0]
                    loss = loss.mean()
                    
                    # 🔧 梯度累积：损失除以累积步数
                    loss = loss / self.args.accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 🔧 每accumulation_steps步更新一次
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    # 梯度裁剪
                    if self.args.clip_grad > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad
                        )
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # 统计（记录真实loss，不是除以accumulation_steps后的）
                real_loss = loss.item() * self.args.accumulation_steps
                losses.update(real_loss, rgb.size(0))
                
                cls_loss = outputs.get('cls_loss', loss)
                if isinstance(cls_loss, tuple):
                    cls_loss = cls_loss[0]
                cls_losses.update(cls_loss.mean().item(), rgb.size(0))
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{losses.avg:.3f}',
                    'cls': f'{cls_losses.avg:.3f}'
                })
                
                # 🔧 定期清理显存
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    
                    # TensorBoard
                    self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                    self.writer.add_scalar('Train/ClsLoss', cls_losses.avg, self.global_step)
                
                # 🔧 显示显存使用（前几个batch）
                if epoch == 1 and batch_idx < 5:
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"\nBatch {batch_idx}: 显存 {allocated:.2f}/{reserved:.2f}GB")
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f'\n❌ Batch {batch_idx} 显存不足!')
                    print(f'当前batch_size: {self.args.batch_size}')
                    print(f'建议减小到: {self.args.batch_size // 2}')
                    
                    # 清理显存
                    torch.cuda.empty_cache()
                    gc.collect()
                    raise e
                else:
                    raise e
        
        # 🔧 epoch结束清理
        torch.cuda.empty_cache()
        gc.collect()
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证 - 显存优化"""
        self.model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Val {epoch}', ncols=100)
        
        for batch in pbar:
            try:
                rgb = batch['rgb_images'].to(self.device, non_blocking=True)
                hsi = batch['hsi_images'].to(self.device, non_blocking=True)
                text_ids = batch['text_input_ids'].to(self.device, non_blocking=True)
                text_mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # 🔧 验证时也用混合精度
                with autocast(dtype=torch.float16):
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
                
            except Exception as e:
                print(f'\n❌ 验证错误: {e}')
                continue
        
        # 🔧 验证结束清理
        torch.cuda.empty_cache()
        gc.collect()
        
        # 计算指标
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            num_classes=self.num_classes
        )
        
        return losses.avg, metrics['accuracy'], metrics['f1']
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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
        print("开始Qwen训练")
        print("="*60)
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print("-" * 60)
            
            try:
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
                
                # 显存状态
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"  显存: {allocated:.2f}/{reserved:.2f}GB")
                
                # 保存
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                
                self.save_checkpoint(epoch, val_acc, is_best)
                
                # 早停
                if epoch - self.best_epoch >= self.args.early_stopping:
                    print(f"\n早停触发 (最佳epoch: {self.best_epoch})")
                    break
                
            except Exception as e:
                print(f"\n❌ Epoch {epoch} 失败: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 保存历史
        history_path = os.path.join(self.args.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 绘制曲线
        try:
            curves_path = os.path.join(self.args.output_dir, 'curves.png')
            plot_training_curves(self.history, save_path=curves_path)
        except:
            pass
        
        print("\n" + "="*60)
        print("✓ 训练完成!")
        print("="*60)
        print(f"最佳准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"输出目录: {self.args.output_dir}")
        print("="*60)
        
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='显存优化的Qwen训练脚本')
    
    # 数据
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2,
                       help='每个GPU的batch size（默认2）')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载线程数（默认2）')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度（必须启用）')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='梯度累积步数（默认4，有效batch=2*4=8）')
    parser.add_argument('--early_stopping', type=int, default=10)
    
    # Qwen
    parser.add_argument('--qwen_path', type=str, required=True)
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs/qwen_fixed')
    parser.add_argument('--use_hsi', action=argparse.BooleanOptionalAction, default=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 🔧 强制启用混合精度
    if not args.use_amp:
        print("警告: 自动启用混合精度以节省显存")
        args.use_amp = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("  显存优化的Qwen训练脚本")
    print("="*60)
    print(f"Batch size: {args.batch_size}")
    print(f"梯度累积: {args.accumulation_steps}")
    print(f"有效batch: {args.batch_size * args.accumulation_steps}")
    print("="*60)
    
    try:
        trainer = MemoryEfficientQwenTrainer(args)
        trainer.train()
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()