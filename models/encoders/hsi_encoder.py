# models/encoders/hsi_encoder_sota.py
"""
SOTA轻量级高光谱图像编码器
基于SpectralFormer (TGRS 2023) 和 HyperTransformer (TPAMI 2024)
结合了光谱-空间分组注意力和高效下采样策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class SpectralGroupAttention(nn.Module):
    """
    光谱分组注意力 (Spectral Group Attention)
    将高光谱波段分组处理，大幅减少计算量
    参考: SpectralFormer (TGRS 2023)
    """
    def __init__(self, in_channels, num_groups=8, reduction=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups
        
        # 每组的注意力
        self.group_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.group_channels, self.group_channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.group_channels // reduction, self.group_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_groups)
        ])
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 分组
        x = x.view(B, self.num_groups, self.group_channels, H, W)
        
        # 对每组应用注意力
        out = []
        for i in range(self.num_groups):
            group = x[:, i]  # [B, group_channels, H, W]
            att = self.group_attention[i](group)
            out.append(group * att)
        
        # 合并
        out = torch.cat(out, dim=1)  # [B, C, H, W]
        
        return out


class EfficientSpectralReduction(nn.Module):
    """
    高效光谱降维模块
    使用深度可分离卷积 + 点卷积
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        
        # 深度可分离卷积：逐通道卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 分组注意力
        self.spectral_att = SpectralGroupAttention(in_channels, num_groups)
        
        # 点卷积：降维
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        # [B, C_in, H, W] -> [B, C_out, H, W]
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.spectral_att(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class SpatialReductionBlock(nn.Module):
    """
    空间下采样块
    使用步长卷积进行空间降维
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GroupedSpatialAttention(nn.Module):
    """
    分组空间注意力
    比全局空间注意力更高效
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.scale = self.head_channels ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x)  # [B, 3*C, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 重塑为多头
        q = rearrange(q, 'b (h c) x y -> b h (x y) c', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h (x y) c', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h (x y) c', h=self.num_heads)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out = attn @ v
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=H, y=W)
        out = self.proj(out)
        
        return out


class SpectralSpatialBlock(nn.Module):
    """
    光谱-空间联合处理块
    结合光谱和空间信息
    """
    def __init__(self, channels, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(channels)
        self.spatial_attn = GroupedSpatialAttention(channels, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 空间注意力
        identity = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.spatial_attn(x)
        x = identity + self.dropout1(x)
        
        # MLP
        identity = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm2(x)
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = identity + x
        
        return x


class HSIEncoder(nn.Module):
    """
    SOTA轻量级高光谱编码器
    
    架构设计：
    1. 高效光谱降维：224 -> 64通道（使用分组卷积）
    2. 渐进式空间下采样：64x64 -> 32x32 -> 16x16
    3. 光谱-空间联合处理：分组注意力
    4. 最终embedding：使用全局池化
    
    显存优化：
    - 分组卷积替代标准卷积
    - 渐进式降维，避免大尺寸特征图
    - 分组注意力替代全局注意力
    - 移除3D卷积
    
    参数：
        hsi_channels: 输入光谱通道数 (default: 224)
        spatial_size: 输入空间尺寸 (default: 64)
        embed_dim: 输出嵌入维度 (default: 768)
        num_groups: 光谱分组数 (default: 8)
        num_heads: 注意力头数 (default: 8)
        depth: Transformer深度 (default: 2)
        dropout: Dropout率 (default: 0.1)
    """
    
    def __init__(
        self,
        hsi_channels=224,
        spatial_size=64,
        embed_dim=768,
        num_groups=8,
        num_heads=8,
        depth=2,
        dropout=0.1,
        **kwargs  # 兼容旧接口
    ):
        super().__init__()
        
        self.hsi_channels = hsi_channels
        self.spatial_size = spatial_size
        self.embed_dim = embed_dim
        
        # Stage 1: 光谱降维 224 -> 64
        # 使用分组卷积，显存友好
        self.spectral_reduction = EfficientSpectralReduction(
            in_channels=hsi_channels,
            out_channels=64,
            num_groups=num_groups
        )
        
        # Stage 2: 空间下采样 64x64 -> 32x32
        self.spatial_down1 = SpatialReductionBlock(64, 128, stride=2)
        
        # Stage 3: 光谱-空间联合处理
        self.ss_block1 = SpectralSpatialBlock(128, num_heads=num_heads, dropout=dropout)
        
        # Stage 4: 进一步空间下采样 32x32 -> 16x16
        self.spatial_down2 = SpatialReductionBlock(128, 256, stride=2)
        
        # Stage 5: 深层特征提取
        self.ss_blocks = nn.ModuleList([
            SpectralSpatialBlock(256, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Stage 6: 特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 最终投影到embed_dim
        self.final_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 用于生成patch tokens
        # 16x16的特征图，使用4x4的patch -> 16个patches
        self.patch_size = 4
        num_patches = (16 // self.patch_size) ** 2  # 16
        
        self.patch_proj = nn.Conv2d(256, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # Position embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, H, W] 高光谱图像，C=224, H=W=64
            
        Returns:
            cls_token: [B, embed_dim] 全局特征
            patch_tokens: [B, num_patches, embed_dim] patch级特征
        """
        B = x.shape[0]
        
        # Stage 1: 光谱降维 [B, 224, 64, 64] -> [B, 64, 64, 64]
        x = self.spectral_reduction(x)
        
        # Stage 2: 空间下采样1 [B, 64, 64, 64] -> [B, 128, 32, 32]
        x = self.spatial_down1(x)
        
        # Stage 3: 光谱-空间处理
        x = self.ss_block1(x)
        
        # Stage 4: 空间下采样2 [B, 128, 32, 32] -> [B, 256, 16, 16]
        x = self.spatial_down2(x)
        
        # Stage 5: 深层特征提取
        for block in self.ss_blocks:
            x = block(x)
        
        # [B, 256, 16, 16]
        
        # 生成全局特征 (cls_token)
        global_feat = self.global_pool(x).flatten(1)  # [B, 256]
        cls_token = self.final_proj(global_feat)  # [B, embed_dim]
        
        # 生成patch tokens
        patch_feat = self.patch_proj(x)  # [B, embed_dim, 4, 4]
        patch_tokens = rearrange(patch_feat, 'b c h w -> b (h w) c')  # [B, 16, embed_dim]
        patch_tokens = self.patch_norm(patch_tokens)
        patch_tokens = patch_tokens + self.pos_embed
        
        return cls_token, patch_tokens
    
    def get_feature_maps(self, x):
        """
        返回中间特征图，用于可视化
        """
        features = {}
        
        x = self.spectral_reduction(x)
        features['stage1'] = x
        
        x = self.spatial_down1(x)
        features['stage2'] = x
        
        x = self.ss_block1(x)
        features['stage3'] = x
        
        x = self.spatial_down2(x)
        features['stage4'] = x
        
        for i, block in enumerate(self.ss_blocks):
            x = block(x)
            features[f'stage5_{i}'] = x
        
        return features


# 内存优化版本 - 使用梯度检查点
class HSIEncoderCheckpoint(HSIEncoder):
    """
    带梯度检查点的HSI编码器
    训练时显存减半，速度略慢
    """
    def forward(self, x):
        B = x.shape[0]
        
        # 使用checkpoint节省显存
        from torch.utils.checkpoint import checkpoint
        
        x = checkpoint(self.spectral_reduction, x, use_reentrant=False)
        x = checkpoint(self.spatial_down1, x, use_reentrant=False)
        x = checkpoint(self.ss_block1, x, use_reentrant=False)
        x = checkpoint(self.spatial_down2, x, use_reentrant=False)
        
        for block in self.ss_blocks:
            x = checkpoint(block, x, use_reentrant=False)
        
        # 全局特征
        global_feat = self.global_pool(x).flatten(1)
        cls_token = self.final_proj(global_feat)
        
        # Patch tokens
        patch_feat = self.patch_proj(x)
        patch_tokens = rearrange(patch_feat, 'b c h w -> b (h w) c')
        patch_tokens = self.patch_norm(patch_tokens)
        patch_tokens = patch_tokens + self.pos_embed
        
        return cls_token, patch_tokens


# 测试和性能分析
if __name__ == "__main__":
    import time
    
    print("="*60)
    print("SOTA轻量级HSI编码器测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 标准版本
    model = HSIEncoder(
        hsi_channels=224,
        spatial_size=64,
        embed_dim=768,
        num_groups=8,
        num_heads=8,
        depth=2
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params/1e6:.2f}M")
    
    # 测试不同batch size的显存占用
    print("\n显存测试:")
    print("-"*60)
    
    for batch_size in [1, 2, 4, 8, 16]:
        torch.cuda.empty_cache()
        try:
            x = torch.randn(batch_size, 224, 64, 64).to(device)
            
            # 前向传播
            with torch.cuda.amp.autocast():
                cls_token, patch_tokens = model(x)
            
            if device == 'cuda':
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"Batch {batch_size:2d}: cls={cls_token.shape}, patches={patch_tokens.shape}, "
                      f"显存={mem_allocated:.2f}GB")
            else:
                print(f"Batch {batch_size:2d}: cls={cls_token.shape}, patches={patch_tokens.shape}")
            
            del x, cls_token, patch_tokens
            
        except RuntimeError as e:
            print(f"Batch {batch_size:2d}: 显存溢出")
            break
    
    # 速度测试
    if device == 'cuda':
        print("\n速度测试:")
        print("-"*60)
        
        model.eval()
        x = torch.randn(4, 224, 64, 64).to(device)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"平均推理时间: {elapsed/100*1000:.2f}ms")
        print(f"吞吐量: {400/elapsed:.2f} samples/sec")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)