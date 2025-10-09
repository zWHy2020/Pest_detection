# scripts/train_fixed.py
"""
修复版训练脚本 - 解决原有问题
适用于不使用Qwen的基线模型训练
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
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.main_model import MultiModalPestDetection
from data import create_dataloaders
from utils import calculate_metrics, AverageMeter, plot_training_curves


class FixedTrainer:
    """修复版训练器 - 解决原有代码问题"""
    
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
            raise RuntimeError("需要GPU进行训练")
        
        self.device = torch.device('cuda:0')
        self.num_gpus = torch.cuda.device_count()
        
        print("\n" + "="*60)
        print("训练器初始化")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"GPU数量: {self.num_gpus}")
        
        # 清理显存
        torch.cuda.empty_cache()
    
    def setup_dataloader(self):
        """创建数据加载器"""
        print("\n创建数据加载器...")
        
        self.train_loader, self.val_loader, _ = create_dataloaders(
            data_root=self.args.data_root,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            text_model_name='bert-base-chinese',
            use_augmentation=True
        )
        
        self.num_classes = self.train_loader.dataset.num_classes
        
        print(f"✓ 训练集: {len(self.train_loader.dataset)} 样本")
        print(f"✓ 验证集: {len(self.val_loader.dataset)} 样本")
        print(f"✓ 类别数: {self.num_classes}")
    
    def setup_model(self):
        """创建模型 - 修复版"""
        print("\n创建模型...")
        
        try:
            self.model = MultiModalPestDetection(
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
                llm_model_name=None,  # 不使用LLM
                use_lora=False,
                freeze_encoders=False
            )
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            print("\n尝试使用简化配置...")
            
            # 🔧 修复：使用更保守的配置
            self.model = MultiModalPestDetection(
                num_classes=self.num_classes,
                rgb_image_size=224,
                hsi_image_size=64,
                hsi_channels=224,
                text_model_name='bert-base-chinese',
                embed_dim=768,
                num_heads=8,  # 减少注意力头数
                dropout=0.1,
                fusion_layers=2,  # 减少融合层
                fusion_strategy='concat',  # 使用简单融合
                llm_model_name=None,
                use_lora=False
            )
        
        # 移到设备
        self.model = self.model.to(self.device)
        
        # 🔧 修复：保存原始模型引用（解决DDP问题）
        self.model_raw = self.model
        
        # 多GPU
        if self.num_gpus > 1:
            print(f"✓ 使用DataParallel ({self.num_gpus} GPUs)")
            self.model = nn.DataParallel(self.model)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model_raw.parameters())
        trainable_params = sum(p.numel() for p in self.model_raw.parameters() if p.requires_grad)
        
        print(f"✓ 总参数: {total_params/1e6:.2f}M")
        print(f"✓ 可训练: {trainable_params/1e6:.2f}M")
    
    def setup_optimizer(self):
        """设置优化器"""
        print("\n设置优化器...")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器 - 使用余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
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
        
        # 最佳指标
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.global_step = 0
        self.patience_counter = 0
        
        # 历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch - 修复版"""
        self.model.train()
        
        losses = AverageMeter()
        cls_losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 🔧 修复：更安全的数据移动
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
                        
                        # 🔧 修复：处理DataParallel返回的tuple
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
                
                cls_loss = outputs.get('cls_loss', loss)
                if isinstance(cls_loss, tuple):
                    cls_loss = cls_loss[0]
                cls_losses.update(cls_loss.mean().item(), rgb.size(0))
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{losses.avg:.3f}',
                    'cls': f'{cls_losses.avg:.3f}'
                })
                
                # TensorBoard
                if batch_idx % 50 == 0:
                    self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                    self.writer.add_scalar('Train/ClsLoss', cls_losses.avg, self.global_step)
                
                self.global_step += 1
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f'\n❌ GPU显存不足! 尝试减小batch_size')
                    print(f'当前batch_size: {self.args.batch_size}')
                    torch.cuda.empty_cache()
                    raise e
                else:
                    print(f'\n❌ 训练错误: {e}')
                    raise e
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证 - 修复版"""
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
        
        # 计算指标
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            num_classes=self.num_classes
        )
        
        return losses.avg, metrics['accuracy'], metrics['f1']
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        # 🔧 修复：使用原始模型
        model_to_save = self.model_raw
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'args': vars(self.args)
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
                
                # 保存
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                self.save_checkpoint(epoch, val_acc, is_best)
                
                # 早停
                if self.patience_counter >= self.args.early_stopping:
                    print(f"\n早停触发 (最佳epoch: {self.best_epoch})")
                    break
                
            except Exception as e:
                print(f"\n❌ Epoch {epoch} 失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 保存当前状态
                self.save_checkpoint(epoch, 0.0, False)
                
                if 'out of memory' in str(e):
                    print("\n建议:")
                    print("  1. 减小 --batch_size")
                    print("  2. 减少 --num_workers")
                    print("  3. 使用 --use_amp (混合精度)")
                    break
                else:
                    continue
        
        # 保存历史
        history_path = os.path.join(self.args.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 绘制曲线
        try:
            curves_path = os.path.join(self.args.output_dir, 'curves.png')
            plot_training_curves(self.history, save_path=curves_path)
        except Exception as e:
            print(f"警告: 绘图失败 - {e}")
        
        print("\n" + "="*60)
        print("✓ 训练完成!")
        print("="*60)
        print(f"最佳准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"输出目录: {self.args.output_dir}")
        print("="*60)
        
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='修复版训练脚本')
    
    # 数据
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='早停轮数')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs/baseline_fixed',
                       help='输出目录')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    
    print("="*60)
    print("  修复版训练脚本")
    print("  解决原有代码的问题")
    print("="*60)
    
    try:
        # 训练
        trainer = FixedTrainer(args)
        trainer.train()
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n请检查:")
        print("  1. 数据集是否准备好")
        print("  2. GPU显存是否充足")
        print("  3. Qwen模型是否需要加载")


if __name__ == '__main__':
    main()