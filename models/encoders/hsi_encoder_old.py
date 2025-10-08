# models/encoders/hsi_encoder.py
"""
高光谱图像编码器模块 - 修复版
结合3D-CNN和Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SpectralAttention(nn.Module):
    """光谱注意力机制"""
    def __init__(self, num_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        reduction = max(num_channels // 8, 1)
        self.channel_attention = nn.Sequential(
            nn.Conv3d(num_channels, reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduction, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if len(x.shape) == 5:  # [B, C, D, H, W]
            x_2d = torch.mean(x, dim=2)
        else:
            x_2d = x
            
        avg_out = torch.mean(x_2d, dim=1, keepdim=True)
        max_out, _ = torch.max(x_2d, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        
        if len(x.shape) == 5:
            attention = attention.unsqueeze(2)
            return x * attention
        else:
            return x_2d * attention

class Conv3DBlock(nn.Module):
    """3D卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SpectralSpatialFeatureExtractor(nn.Module):
    """光谱-空间特征提取器"""
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        self.conv3d_1 = Conv3DBlock(in_channels, base_channels,
                                   kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.conv3d_2 = Conv3DBlock(base_channels, base_channels*2,
                                   kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.conv3d_3 = Conv3DBlock(base_channels*2, base_channels*4,
                                   kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.spectral_attention = SpectralAttention(base_channels*4)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.spectral_attention(x)
        x = self.spatial_attention(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x

class HSIEncoder(nn.Module):
    """
    高光谱图像编码器 - 修复版
    输入: [B, C, H, W] 其中C是光谱波段数
    输出: [B, embed_dim] 分类token 和 [B, N, embed_dim] patch tokens
    """
    def __init__(
        self,
        hsi_channels=224,      # 光谱波段数
        spatial_size=64,       # 空间尺寸
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
        patch_size=8
    ):
        super().__init__()
        
        self.hsi_channels = hsi_channels
        self.spatial_size = spatial_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # 3D卷积特征提取
        self.feature_extractor = SpectralSpatialFeatureExtractor(
            in_channels=1,
            base_channels=32
        )
        
        # 计算3D卷积后的维度
        conv_out_channels = 128  # 32 * 4
        
        # 维度压缩：将光谱维度压缩
        self.spectral_compress = nn.Sequential(
            nn.Conv3d(conv_out_channels, embed_dim, kernel_size=(hsi_channels, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Patch embedding
        num_patches = (spatial_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # CLS token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 高光谱图像
        Returns:
            cls_token: [B, embed_dim]
            patch_tokens: [B, N, embed_dim]
        """
        B, C, H, W = x.shape
        
        # 重塑为3D: [B, 1, C, H, W]
        x = x.unsqueeze(1)
        
        # 3D卷积特征提取
        x = self.feature_extractor(x)  # [B, 128, C, H, W]
        
        # 光谱维度压缩
        x = self.spectral_compress(x)  # [B, embed_dim, 1, H, W]
        x = x.squeeze(2)  # [B, embed_dim, H, W]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/p, W/p]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, N, embed_dim]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x[:, 0], x[:, 1:]  # CLS token, patch tokens

# 测试代码
if __name__ == "__main__":
    model = HSIEncoder(
        hsi_channels=224,
        spatial_size=64,
        embed_dim=768,
        depth=6,
        num_heads=12
    )
    
    x = torch.randn(2, 224, 64, 64)
    cls_token, patch_tokens = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"CLS token形状: {cls_token.shape}")
    print(f"Patch tokens形状: {patch_tokens.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")