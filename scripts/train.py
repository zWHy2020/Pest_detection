# scripts/train.py
"""
多模态病虫害识别模型训练脚本 - 优化版
支持DataParallel和DistributedDataParallel
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
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 统一导入
from models import MultiModalPestDetection
from data import create_dataloaders
from utils import calculate_metrics, AverageMeter
from utils import plot_confusion_matrix, plot_training_curves
from training import Trainer


def setup_distributed(rank, world_size):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='多模态病虫害识别训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ============ 数据参数 ============
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data_root', type=str, required=True,
                           help='数据集根目录')
    data_group.add_argument('--batch_size', type=int, default=16,
                           help='每个GPU的batch size')
    data_group.add_argument('--num_workers', type=int, default=4,
                           help='数据加载线程数')
    data_group.add_argument('--use_augmentation', action='store_true',
                           help='使用数据增强')
    
    # ============ 模型参数 ============
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--rgb_size', type=int, default=224)
    model_group.add_argument('--hsi_size', type=int, default=64)
    model_group.add_argument('--hsi_channels', type=int, default=224)
    model_group.add_argument('--embed_dim', type=int, default=768)
    model_group.add_argument('--num_heads', type=int, default=12)
    model_group.add_argument('--dropout', type=float, default=0.1)
    model_group.add_argument('--fusion_layers', type=int, default=4)
    model_group.add_argument('--fusion_strategy', type=str, default='hierarchical',
                            choices=['concat', 'gated', 'hierarchical'])
    model_group.add_argument('--text_model_name', type=str, 
                            default='bert-base-chinese')
    
    # ============ LLM参数 ============
    llm_group = parser.add_argument_group('LLM')
    llm_group.add_argument('--use_llm', action='store_true',
                          help='使用大语言模型')
    llm_group.add_argument('--llm_model_name', type=str, default=None)
    llm_group.add_argument('--use_lora', action='store_true',
                          help='使用LoRA微调')
    llm_group.add_argument('--lora_rank', type=int, default=8)
    llm_group.add_argument('--lora_alpha', type=float, default=16)
    llm_group.add_argument('--freeze_encoders', action='store_true',
                          help='冻结编码器')
    
    # ============ 训练参数 ============
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=100)
    train_group.add_argument('--lr', type=float, default=1e-4)
    train_group.add_argument('--weight_decay', type=float, default=0.01)
    train_group.add_argument('--label_smoothing', type=float, default=0.1)
    train_group.add_argument('--clip_grad', type=float, default=1.0)
    train_group.add_argument('--use_amp', action='store_true',
                            help='使用混合精度训练')
    train_group.add_argument('--early_stopping_patience', type=int, default=20)
    
    # ============ 多GPU参数 ============
    gpu_group = parser.add_argument_group('Multi-GPU')
    gpu_group.add_argument('--distributed', action='store_true',
                          help='使用DistributedDataParallel')
    gpu_group.add_argument('--world_size', type=int, default=None,
                          help='GPU数量（自动检测）')
    gpu_group.add_argument('--gpu_ids', type=str, default=None,
                          help='指定GPU，如"0,1,2,3"')
    
    # ============ 输出参数 ============
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=str, default='./outputs')
    output_group.add_argument('--exp_name', type=str, 
                             default=f'exp_{torch.randint(0, 10000, (1,)).item()}')
    output_group.add_argument('--save_interval', type=int, default=10)
    output_group.add_argument('--log_interval', type=int, default=50)
    output_group.add_argument('--plot_interval', type=int, default=10)
    
    # ============ 其他 ============
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 设置GPU
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    # 自动检测GPU数量
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    
    return args


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_worker(rank, world_size, args):
    """每个GPU的工作进程"""
    is_main = (rank == 0)
    
    # 分布式初始化
    if args.distributed:
        setup_distributed(rank, world_size)
    
    try:
        # 设置设备
        if args.distributed:
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if is_main:
            print("\n" + "="*60)
            print("训练配置")
            print("="*60)
            print(f"实验名称: {args.exp_name}")
            print(f"数据路径: {args.data_root}")
            print(f"训练模式: {'DistributedDataParallel' if args.distributed else 'DataParallel'}")
            print(f"GPU数量: {world_size}")
            print(f"Batch Size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")
            print(f"训练轮数: {args.epochs}")
            print(f"学习率: {args.lr}")
            print(f"混合精度: {args.use_amp}")
            print("="*60 + "\n")
        
        # 创建数据加载器
        if is_main:
            print("创建数据加载器...")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text_model_name=args.text_model_name,
            use_augmentation=args.use_augmentation
        )
        
        num_classes = train_loader.dataset.num_classes
        
        if is_main:
            print(f"✓ 训练集: {len(train_loader.dataset)} 样本")
            print(f"✓ 验证集: {len(val_loader.dataset)} 样本")
            print(f"✓ 测试集: {len(test_loader.dataset)} 样本")
            print(f"✓ 类别数: {num_classes}\n")
        
        # 创建模型
        if is_main:
            print("创建模型...")
        
        model = MultiModalPestDetection(
            num_classes=num_classes,
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
        
        # 移动到设备
        model = model.to(device)
        
        # 包装模型
        if args.distributed:
            model = DDP(model, device_ids=[rank], output_device=rank,
                       find_unused_parameters=True)
        elif world_size > 1:
            model = nn.DataParallel(model)
        
        if is_main:
            model_for_count = model.module if hasattr(model, 'module') else model
            total_params = sum(p.numel() for p in model_for_count.parameters())
            trainable_params = sum(p.numel() for p in model_for_count.parameters() 
                                  if p.requires_grad)
            print(f"✓ 总参数: {total_params/1e6:.2f}M")
            print(f"✓ 可训练参数: {trainable_params/1e6:.2f}M")
            print(f"✓ 可训练比例: {100*trainable_params/total_params:.1f}%\n")
        
        # 创建优化器
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        
        # 使用Trainer类（只在主进程创建输出目录）
        if is_main:
            output_dir = os.path.join(args.output_dir, args.exp_name)
            os.makedirs(output_dir, exist_ok=True)
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                output_dir=args.output_dir,
                exp_name=args.exp_name,
                use_amp=args.use_amp,
                gradient_clip=args.clip_grad,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                early_stopping_patience=args.early_stopping_patience
            )
            
            # 开始训练
            trainer.train(
                num_epochs=args.epochs,
                resume_from=args.resume
            )
            
            print("\n" + "="*60)
            print("✓ 训练完成！")
            print("="*60)
            print(f"最佳准确率: {trainer.best_val_acc:.4f}")
            print(f"输出目录: {output_dir}")
            print("="*60)
        else:
            # 非主进程只做训练
            for epoch in range(1, args.epochs + 1):
                model.train()
                for batch in train_loader:
                    # 简化的训练循环
                    rgb = batch['rgb_images'].to(device)
                    hsi = batch['hsi_images'].to(device)
                    text_ids = batch['text_input_ids'].to(device)
                    text_mask = batch['text_attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(rgb, hsi, text_ids, text_mask, labels=labels)
                    loss = outputs['total_loss'].mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
    
    finally:
        if args.distributed:
            cleanup_distributed()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到CUDA设备")
        return
    
    available_gpus = torch.cuda.device_count()
    print(f"\n检测到 {available_gpus} 个GPU:")
    for i in range(available_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if args.world_size > available_gpus:
        print(f"警告: 指定了{args.world_size}个GPU，但只有{available_gpus}个可用")
        args.world_size = available_gpus
    
    # 启动训练
    if args.distributed and args.world_size > 1:
        print(f"\n使用 DistributedDataParallel ({args.world_size} GPUs)")
        torch.multiprocessing.spawn(
            main_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        if args.world_size > 1:
            print(f"\n使用 DataParallel ({args.world_size} GPUs)")
        else:
            print("\n使用单GPU训练")
        args.distributed = False
        main_worker(0, args.world_size, args)


if __name__ == '__main__':
    main()