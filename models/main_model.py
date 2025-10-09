# models/main_model.py
"""
完整的多模态病虫害识别模型 - 修复版
"""
from training.loss import FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import sys
import os

# 确保能够导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.encoders.rgb_encoder import RGBEncoder
from models.encoders.hsi_encoder import HSIEncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion.cross_modal_alignment import CrossModalAlignmentModule
from models.fusion.multi_modal_fusion import MultiModalFusionModule

class MultiModalPestDetection(nn.Module):
    """
    完整的多模态病虫害识别系统
    """
    def __init__(
        self,
        # 任务配置
        num_classes: int = 50,
        
        # 编码器配置
        rgb_image_size: int = 224,
        rgb_patch_size: int = 16,
        hsi_image_size: int = 64,
        hsi_channels: int = 224,
        text_model_name: str = "bert-base-chinese",
        
        # 通用配置
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        
        # 融合配置
        fusion_layers: int = 4,
        fusion_strategy: str = 'hierarchical',
        
        # LLM配置 (可选)
        llm_model_name: Optional[str] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        
        # 其他配置
        freeze_encoders: bool = False
    ):
        super().__init__()
        self.cls_criterion = FocalLoss()
        self.num_classes = num_classes
        self.use_llm = llm_model_name is not None
        
        # ============ 编码器模块 ============
        self.rgb_encoder = RGBEncoder(
            image_size=rgb_image_size,
            patch_size=rgb_patch_size,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.hsi_encoder = HSIEncoder(
            hsi_channels=hsi_channels,
            spatial_size=hsi_image_size,
            embed_dim=embed_dim,
            depth=6,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_encoder=freeze_encoders,
            dropout=dropout
        )
        
        # 冻结编码器（如果需要）
        if freeze_encoders:
            for param in self.rgb_encoder.parameters():
                param.requires_grad = False
            for param in self.hsi_encoder.parameters():
                param.requires_grad = False
        
        # ============ 跨模态对齐 ============
        self.alignment = CrossModalAlignmentModule(
            rgb_dim=embed_dim,
            hsi_dim=embed_dim,
            text_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ============ 多模态融合 ============
        self.fusion = MultiModalFusionModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=fusion_layers,
            dropout=dropout,
            fusion_strategy=fusion_strategy
        )
        
        # ============ LLM适配器（可选）============
        if self.use_llm:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from models.adapters.lora import apply_lora_to_model
                from models.adapters.llm_adapter import LLMAdapter
                
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                
                # 冻结LLM
                for param in self.llm.parameters():
                    param.requires_grad = False
                
                # 应用LoRA
                if use_lora:
                    self.llm = apply_lora_to_model(
                        self.llm,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=dropout
                    )
                
                # LLM适配器
                self.llm_adapter = LLMAdapter(
                    multimodal_dim=embed_dim,
                    llm_hidden_size=self.llm.config.hidden_size,
                    num_query_tokens=32,
                    dropout=dropout
                )
            except Exception as e:
                print(f"警告: LLM初始化失败: {e}")
                self.use_llm = False
        
        # ============ 分类头 ============
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(
        self,
        rgb_images: torch.Tensor,
        hsi_images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb_images: [B, 3, H, W]
            hsi_images: [B, C_hsi, H, W]
            text_input_ids: [B, L]
            text_attention_mask: [B, L]
            labels: [B] 标签
            return_features: 是否返回中间特征
            
        Returns:
            包含输出和损失的字典
        """
        # ============ 特征编码 ============
        # RGB编码
        rgb_cls, rgb_patches = self.rgb_encoder(rgb_images)
        rgb_features = torch.cat([rgb_cls.unsqueeze(1), rgb_patches], dim=1)
        
        # HSI编码
        hsi_cls, hsi_patches = self.hsi_encoder(hsi_images)
        hsi_features = torch.cat([hsi_cls.unsqueeze(1), hsi_patches], dim=1)
        
        # 文本编码
        text_outputs = self.text_encoder(text_input_ids, text_attention_mask)
        text_features = text_outputs['sequence_features']
        
        # ============ 跨模态对齐 ============
        alignment_outputs = self.alignment(
            rgb_features, hsi_features, text_features, text_attention_mask
        )
        
        rgb_aligned = alignment_outputs['rgb_aligned']
        hsi_aligned = alignment_outputs['hsi_aligned']
        text_aligned = alignment_outputs['text_aligned']
        alignment_loss = alignment_outputs['alignment_loss']
        
        # ============ 多模态融合 ============
        fusion_outputs = self.fusion(
            rgb_aligned, hsi_aligned, text_aligned, return_intermediate=False
        )
        
        fused_features = fusion_outputs['fused_features']
        
        # ============ 分类 ============
        logits = self.classifier(fused_features)
        
        # ============ 计算损失 ============
        total_loss = None
        cls_loss = None
        
        if labels is not None:
            cls_loss = self.cls_criterion(logits, labels)
            alignment_loss = alignment_outputs['alignment_loss']
            alignment_weight = 1.0
            total_loss = cls_loss + alignment_weight * alignment_loss
        
        # ============ 返回结果 ============
        device = logits.device
        outputs = {
            'logits': logits,
            'fused_features': fused_features,
            'alignment_loss': alignment_loss,
            'cls_loss': cls_loss if cls_loss is not None else torch.tensor(0.0, device=device),
            'total_loss': total_loss if total_loss is not None else torch.tensor(0.0, device=device)
        }
        
        if return_features:
            outputs.update({
                'rgb_features': rgb_features,
                'hsi_features': hsi_features,
                'text_features': text_features,
                'rgb_aligned': rgb_aligned,
                'hsi_aligned': hsi_aligned,
                'text_aligned': text_aligned
            })
        
        return outputs

# 测试代码
if __name__ == "__main__":
    model = MultiModalPestDetection(
        num_classes=50,
        llm_model_name=None
    )
    
    B = 2
    rgb = torch.randn(B, 3, 224, 224)
    hsi = torch.randn(B, 224, 64, 64)
    text_ids = torch.randint(0, 1000, (B, 50))
    text_mask = torch.ones(B, 50)
    labels = torch.randint(0, 50, (B,))
    
    outputs = model(rgb, hsi, text_ids, text_mask, labels=labels)
    
    print("=" * 50)
    print("模型测试结果")
    print("=" * 50)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Total loss: {outputs['total_loss'].item():.4f}")
    print(f"Classification loss: {outputs['cls_loss'].item():.4f}")
    print(f"Alignment loss: {outputs['alignment_loss'].item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 50)
    print("模型参数统计")
    print("=" * 50)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"可训练比例: {100 * trainable_params / total_params:.2f}%")