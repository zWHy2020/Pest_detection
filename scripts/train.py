# scripts/train.py
"""
多模态病虫害识别模型训练脚本 - 多GPU版本
支持DataParallel和DistributedDataParallel两种模式
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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


def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


class MultiGPUTrainer:
    """多GPU训练器类"""
    def __init__(self, args, rank=0, world_size=1):
        self.args = args
        self.rank = rank  # 当前GPU的rank
        self.world_size = world_size  # 总GPU数量
        self.is_main_process = (rank == 0)  # 只有rank 0进行日志记录
        
        # 设置设备
        if args.distributed:
            torch.cuda.set_device(rank)
            self.device = torch.device(f'cuda:{rank}')
        
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"多GPU训练配置")
            print(f"{'='*60}")
            print(f"训练模式: {'DistributedDataParallel' if args.distributed else 'DataParallel'}")
            print(f"GPU数量: {world_size}")
            print(f"总batch size: {args.batch_size * world_size}")
            print(f"每GPU batch size: {args.batch_size}")
            print(f"{'='*60}\n")
        
        # 创建输出目录（只在主进程）
        if self.is_main_process:
            self.output_dir = os.path.join(args.output_dir, args.exp_name)
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
            
            # TensorBoard
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'logs'))
            
            # 保存配置
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(vars(args), f, indent=2)
        
        # 创建数据加载器
        if self.is_main_process:
            print("创建数据加载器...")
        
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()
        
        if self.is_main_process:
            print(f"训练集: {len(self.train_loader.dataset)} 样本")
            print(f"验证集: {len(self.val_loader.dataset)} 样本")
            print(f"测试集: {len(self.test_loader.dataset)} 样本")
            print(f"类别数: {self.train_loader.dataset.num_classes}")
        
        # 创建模型
        if self.is_main_process:
            print("\n创建模型...")
        
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
        
        # 移动模型到GPU并包装
        if args.distributed:
            # 使用DistributedDataParallel
            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model, 
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True  # 处理可能的未使用参数
            )
        else:
            # 使用DataParallel（更简单但效率略低）
            self.model = self.model.to(self.device)
            if torch.cuda.device_count() > 1:
                if self.is_main_process:
                    print(f"使用 DataParallel，{torch.cuda.device_count()} 个GPU")
                self.model = nn.DataParallel(self.model)
        
        # 打印模型信息（只在主进程）
        if self.is_main_process:
            model_for_count = self.model.module if hasattr(self.model, 'module') else self.model
            total_params = sum(p.numel() for p in model_for_count.parameters())
            trainable_params = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
            print(f"\n模型参数统计:")
            print(f"总参数: {total_params / 1e6:.2f}M")
            print(f"可训练参数: {trainable_params / 1e6:.2f}M")
            print(f"可训练比例: {100 * trainable_params / total_params:.2f}%\n")
        
        # 创建优化器
        self._create_optimizer()
        
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
        self.global_step = 0
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        # 根据是否使用分布式采样器
        if self.args.distributed:
            # 使用DistributedSampler
            from data.dataset import PestDataset
            
            train_dataset = PestDataset(
                data_root=self.args.data_root,
                split='train',
                text_model_name=self.args.text_model_name,
                use_augmentation=self.args.use_augmentation
            )
            
            val_dataset = PestDataset(
                data_root=self.args.data_root,
                split='val',
                text_model_name=self.args.text_model_name,
                use_augmentation=False
            )
            
            test_dataset = PestDataset(
                data_root=self.args.data_root,
                split='test',
                text_model_name=self.args.text_model_name,
                use_augmentation=False
            )
            
            # 创建分布式采样器
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            
            from data.dataset import collate_fn
            from torch.utils.data import DataLoader
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=train_sampler,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                sampler=val_sampler,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                sampler=test_sampler,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
            
            return train_loader, val_loader, test_loader
        else:
            # 使用普通的create_dataloaders
            return create_dataloaders(
                data_root=self.args.data_root,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                text_model_name=self.args.text_model_name,
                use_augmentation=self.args.use_augmentation
            )
    
    def _create_optimizer(self):
        """创建优化器"""
        # 获取实际的模型（去除DDP/DP包装）
        model_for_opt = self.model.module if hasattr(self.model, 'module') else self.model
        
        if self.args.use_lora:
            from models.adapters.lora import get_lora_parameters
            #lora_params = get_lora_parameters(model_for_opt)
            lora_params = list(get_lora_parameters(model_for_opt))
            lora_param_ids = {id(p) for p in lora_params}
            other_params = [p for p in model_for_opt.parameters()
                            if p.requires_grad and id(p) not in lora_param_ids]
            
            #other_params = [p for p in model_for_opt.parameters() 
                          #if p.requires_grad and p not in lora_params]
            param_groups = []
            if lora_params:
                param_groups.append({'params': lora_params, 'lr': self.args.lr * 10})
            if other_params:
                param_groups.append({'params': other_params, 'lr': self.args.lr})
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
            #self.optimizer = optim.AdamW([
                #{'params': lora_params, 'lr': self.args.lr * 10},
                #{'params': other_params, 'lr': self.args.lr}
            #], weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model_for_opt.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        cls_losses = AverageMeter()
        align_losses = AverageMeter()
        
        # 设置分布式采样器的epoch
        if self.args.distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        # 只在主进程显示进度条
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        else:
            pbar = self.train_loader
        
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
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
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
                outputs = self.model(
                    rgb_images, hsi_images,
                    text_input_ids, text_attention_mask,
                    labels=labels
                )
                loss = outputs['total_loss']
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad
                    )
                
                self.optimizer.step()
            
            # 更新统计
            losses.update(loss.item(), rgb_images.size(0))
            cls_losses.update(outputs['cls_loss'].mean().item() if outputs['cls_loss'].dim() > 0 else outputs['cls_loss'].item(), rgb_images.size(0))
            align_losses.update(outputs['alignment_loss'].mean().item() if outputs['alignment_loss'].dim() > 0 else outputs['alignment_loss'].item(), rgb_images.size(0))
            
            # 更新进度条（只在主进程）
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'cls': f'{cls_losses.avg:.4f}',
                    'align': f'{align_losses.avg:.4f}'
                })
                
                # 记录到TensorBoard
                if batch_idx % self.args.log_interval == 0:
                    self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                    self.writer.add_scalar('Train/ClsLoss', cls_losses.avg, self.global_step)
                    self.writer.add_scalar('Train/AlignLoss', align_losses.avg, self.global_step)
                
                self.global_step += 1
        
        return losses.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        # 只在主进程显示进度条
        if self.is_main_process:
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
        else:
            pbar = self.val_loader
        
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
            
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            logits = outputs['logits']
            
            losses.update(loss.item(), rgb_images.size(0))
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if self.is_main_process:
                pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # 在分布式训练中，需要收集所有进程的预测结果
        if self.args.distributed:
            # 将列表转换为tensor
            preds_tensor = torch.tensor(all_preds, device=self.device)
            labels_tensor = torch.tensor(all_labels, device=self.device)
            
            # 收集所有GPU的结果
            all_preds_list = [torch.zeros_like(preds_tensor) for _ in range(self.world_size)]
            all_labels_list = [torch.zeros_like(labels_tensor) for _ in range(self.world_size)]
            
            dist.all_gather(all_preds_list, preds_tensor)
            dist.all_gather(all_labels_list, labels_tensor)
            
            # 只在主进程计算指标
            if self.is_main_process:
                all_preds = torch.cat(all_preds_list).cpu().numpy()
                all_labels = torch.cat(all_labels_list).cpu().numpy()
        
        # 计算指标（只在主进程）
        if self.is_main_process:
            metrics = calculate_metrics(
                np.array(all_labels),
                np.array(all_preds),
                num_classes=self.train_loader.dataset.num_classes
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
        else:
            return None, None, None
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点（只在主进程）"""
        if not self.is_main_process:
            return
        
        # 获取实际的模型state_dict
        model_for_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_for_save.state_dict(),
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
            print(f"✓ 保存最佳模型 (Acc: {metrics['accuracy']:.4f})")
    
    def train(self):
        """完整训练流程"""
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"开始训练")
            print(f"{'='*60}")
            print(f"输出目录: {self.output_dir}")
            print(f"总epochs: {self.args.epochs}")
            print(f"设备: {self.device}")
            print(f"混合精度: {self.args.use_amp}")
            print(f"{'='*60}\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            metrics, preds, labels = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 只在主进程保存检查点
            if self.is_main_process and metrics is not None:
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
        
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"训练完成！")
            print(f"{'='*60}")
            print(f"最佳准确率: {self.best_acc:.4f}")
            print(f"最佳F1分数: {self.best_f1:.4f}")
            print(f"{'='*60}\n")
            
            self.writer.close()


def main_worker(rank, world_size, args):
    """每个GPU的工作进程"""
    if args.distributed:
        setup_distributed(rank, world_size)
    
    try:
        trainer = MultiGPUTrainer(args, rank, world_size)
        trainer.train()
    finally:
        if args.distributed:
            cleanup_distributed()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态病虫害识别训练 - 多GPU版本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='每个GPU的batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_augmentation', action='store_true')
    
    # 模型参数
    parser.add_argument('--rgb_size', type=int, default=224)
    parser.add_argument('--hsi_size', type=int, default=64)
    parser.add_argument('--hsi_channels', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fusion_layers', type=int, default=4)
    parser.add_argument('--fusion_strategy', type=str, default='hierarchical')
    parser.add_argument('--text_model_name', type=str, default='bert-base-chinese')
    
    # LLM参数
    parser.add_argument('--use_llm', action='store_true')
    parser.add_argument('--llm_model_name', type=str, default=None)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16)
    parser.add_argument('--freeze_encoders', action='store_true')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adamw')
    
    # 多GPU参数
    parser.add_argument('--distributed', action='store_true',
                        help='使用DistributedDataParallel (推荐)')
    parser.add_argument('--world_size', type=int, default=2,
                        help='GPU数量')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default='multi_gpu_training')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--plot_interval', type=int, default=10)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    
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
    args = parse_args()
    set_seed(args.seed)
    
    # 检测可用的GPU数量
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"\n检测到 {available_gpus} 个GPU:")
        for i in range(available_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if args.world_size > available_gpus:
            print(f"警告: 指定了{args.world_size}个GPU，但只有{available_gpus}个可用")
            args.world_size = available_gpus
    else:
        print("错误: 没有检测到CUDA设备")
        return
    
    if args.distributed and args.world_size > 1:
        # 使用DistributedDataParallel
        print(f"\n使用 DistributedDataParallel 训练 ({args.world_size} GPUs)")
        torch.multiprocessing.spawn(
            main_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # 使用DataParallel或单GPU
        if args.world_size > 1:
            print(f"\n使用 DataParallel 训练 ({args.world_size} GPUs)")
        else:
            print("\n使用单GPU训练")
        
        args.distributed = False
        main_worker(0, args.world_size, args)


if __name__ == '__main__':
    main()

