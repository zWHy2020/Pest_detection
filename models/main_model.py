# models/main_model.py
"""
完整的多模态病虫害识别模型
集成所有编码器、对齐、融合和LLM适配器模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入自定义模块（需要确保路径正确）
import sys
sys.path.append('..')

from encoders.rgb_encoder import RGBEncoder
from encoders.hsi_encoder import HSIEncoder
from encoders.text_encoder import TextEncoder
from fusion.cross_modal_alignment import CrossModalAlignmentModule
from fusion.multi_modal_fusion import MultiModalFusionModule
from adapters.lora import apply_lora_to_model, LoRALinear


class LLMAdapter(nn.Module):
    """大语言模型适配器"""
    def __init__(
        self,
        multimodal_dim: int = 768,
        llm_hidden_size: int = 4096,
        num_query_tokens: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 可学习的查询token
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, multimodal_dim))
        
        # 跨注意力：查询多模态特征
        self.cross_attn = nn.MultiheadAttention(
            multimodal_dim, num_heads=12, dropout=dropout, batch_first=True
        )
        
        # 投影到LLM维度
        self.projector = nn.Sequential(
            nn.Linear(multimodal_dim, multimodal_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(multimodal_dim * 2, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size)
        )
        
    def forward(self, multimodal_features: torch.Tensor) -> torch.Tensor:
        """
        将多模态特征转换为LLM可用的token
        
        Args:
            multimodal_features: [B, N, D] 融合后的多模态特征
        Returns:
            llm_tokens: [B, num_query_tokens, llm_hidden_size]
        """
        B = multimodal_features.shape[0]
        
        # 扩展查询token
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # 跨注意力提取特征
        attended_features, _ = self.cross_attn(
            query_tokens, multimodal_features, multimodal_features
        )
        
        # 投影到LLM维度
        llm_tokens = self.projector(attended_features)
        
        return llm_tokens


class MultiModalPestDetection(nn.Module):
    """
    完整的多模态病虫害识别系统
    """
    def __init__(
        self,
        # 编码器配置
        rgb_image_size: int = 224,
        rgb_patch_size: int = 16,
        hsi_image_size: int = 64,
        hsi_channels: int = 224,
        hsi_patch_size: int = 8,
        text_model_name: str = "bert-base-chinese",
        
        # 通用配置
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        
        # 融合配置
        fusion_layers: int = 4,
        fusion_strategy: str = 'hierarchical',
        
        # LLM配置
        llm_model_name: Optional[str] = None,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        
        # 任务配置
        num_classes: int = 50,
        freeze_encoders: bool = False
    ):
        super().__init__()
        
        self.use_llm = llm_model_name is not None
        self.num_classes = num_classes
        
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
            image_size=hsi_image_size,
            in_channels=hsi_channels,
            patch_size=hsi_patch_size,
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
            # 加载LLM
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16,
                device_map='auto'
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
        instruction_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            rgb_images: [B, 3, H, W]
            hsi_images: [B, C_hsi, D, H, W]
            text_input_ids: [B, L]
            text_attention_mask: [B, L]
            instruction_ids: [B, L_inst] 任务指令（用于LLM）
            labels: [B] 标签
            return_features: 是否返回中间特征
            
        Returns:
            包含输出和损失的字典
        """
        # ============ 特征编码 ============
        # RGB编码
        rgb_cls, rgb_patches = self.rgb_encoder(rgb_images)  # [B, D], [B, N_rgb, D]
        rgb_features = torch.cat([rgb_cls.unsqueeze(1), rgb_patches], dim=1)
        
        # HSI编码
        hsi_cls, hsi_patches = self.hsi_encoder(hsi_images)  # [B, D], [B, N_hsi, D]
        hsi_features = torch.cat([hsi_cls.unsqueeze(1), hsi_patches], dim=1)
        
        # 文本编码
        text_outputs = self.text_encoder(text_input_ids, text_attention_mask)
        text_features = text_outputs['sequence_features']  # [B, L, D]
        
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
        
        fused_features = fusion_outputs['fused_features']  # [B, D]
        
        # ============ 分类 ============
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        # ============ LLM生成（可选）============
        llm_output = None
        llm_loss = None
        
        if self.use_llm and instruction_ids is not None:
            # 获取完整的融合特征序列
            all_features = fusion_outputs['all_features']  # [B, N_total, D]
            
            # 转换为LLM tokens
            llm_tokens = self.llm_adapter(all_features)  # [B, num_query, llm_hidden]
            
            # 嵌入指令
            instruction_embeds = self.llm.get_input_embeddings()(instruction_ids)
            
            # 拼接多模态tokens和指令
            inputs_embeds = torch.cat([llm_tokens, instruction_embeds], dim=1)
            
            # LLM前向传播
            llm_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                labels=labels if labels is not None else None
            )
            
            llm_output = llm_outputs.logits
            if labels is not None:
                llm_loss = llm_outputs.loss
        
        # ============ 计算总损失 ============
        total_loss = None
        cls_loss = None
        
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
            total_loss = cls_loss + 0.1 * alignment_loss
            
            if llm_loss is not None:
                total_loss = total_loss + 0.5 * llm_loss
        
        # ============ 返回结果 ============
        outputs = {
            'logits': logits,
            'fused_features': fused_features,
            'alignment_loss': alignment_loss,
            'cls_loss': cls_loss,
            'total_loss': total_loss
        }
        
        if llm_output is not None:
            outputs['llm_output'] = llm_output
            outputs['llm_loss'] = llm_loss
        
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
    
    def generate_description(
        self,
        rgb_images: torch.Tensor,
        hsi_images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        instruction: str = "请描述这个病虫害的症状和可能的原因：",
        max_length: int = 128
    ) -> List[str]:
        """
        使用LLM生成病虫害描述
        """
        if not self.use_llm:
            raise ValueError("Model was not initialized with LLM")
        
        self.eval()
        with torch.no_grad():
            # 获取融合特征
            outputs = self.forward(
                rgb_images, hsi_images, 
                text_input_ids, text_attention_mask
            )
            
            all_features = self.fusion(
                outputs['rgb_aligned'],
                outputs['hsi_aligned'],
                outputs['text_aligned']
            )['all_features']
            
            # 转换为LLM tokens
            llm_tokens = self.llm_adapter(all_features)
            
            # Tokenize指令
            instruction_ids = self.llm_tokenizer(
                [instruction] * rgb_images.shape[0],
                return_tensors='pt',
                padding=True
            ).input_ids.to(rgb_images.device)
            
            instruction_embeds = self.llm.get_input_embeddings()(instruction_ids)
            
            # 拼接
            inputs_embeds = torch.cat([llm_tokens, instruction_embeds], dim=1)
            
            # 生成
            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            # 解码
            generated_texts = self.llm_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            return generated_texts


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = MultiModalPestDetection(
        num_classes=50,
        use_lora=True,
        llm_model_name=None  # 不使用LLM进行快速测试
    )
    
    # 模拟输入
    B = 2
    rgb = torch.randn(B, 3, 224, 224)
    hsi = torch.randn(B, 224, 8, 64, 64)
    text_ids = torch.randint(0, 1000, (B, 50))
    text_mask = torch.ones(B, 50)
    labels = torch.randint(0, 50, (B,))
    
    # 前向传播
    outputs = model(rgb, hsi, text_ids, text_mask, labels=labels)
    
    print("=" * 50)
    print("模型测试结果")
    print("=" * 50)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Total loss: {outputs['total_loss'].item():.4f}")
    print(f"Classification loss: {outputs['cls_loss'].item():.4f}")
    print(f"Alignment loss: {outputs['alignment_loss'].item():.4f}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 50)
    print("模型参数统计")
    print("=" * 50)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"可训练比例: {100 * trainable_params / total_params:.2f}%")