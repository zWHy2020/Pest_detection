# models/encoders/hsi_encoder.py
"""
高光谱图像编码器模块
结合3D-CNN和Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SpectralAttention(nn.Module):
    """光谱注意力机制 - 关注重要的光谱波段"""
    def __init__(self, num_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv3d(num_channels, num_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_channels // 8, num_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化和最大池化
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        
        # 融合并应用注意力
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    """空间注意力机制 - 关注重要的空间位置"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 对于3D输入，先在光谱维度上进行池化
        if len(x.shape) == 5:  # [B, C, D, H, W]
            # 在深度维度上取平均
            x_2d = torch.mean(x, dim=2)
        else:
            x_2d = x
            
        # 计算空间注意力
        avg_out = torch.mean(x_2d, dim=1, keepdim=True)
        max_out, _ = torch.max(x_2d, dim=1, keepdim=True)
        
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        
        # 将注意力扩展回原始维度
        if len(x.shape) == 5:
            attention = attention.unsqueeze(2)
            return x * attention
        else:
            return x_2d * attention

class Conv3DBlock(nn.Module):
    """3D卷积块 - 用于提取光谱-空间特征"""
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
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpectralSpatialFeatureExtractor(nn.Module):
    """光谱-空间特征提取器"""
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        # 3D卷积层 - 逐步提取特征
        self.conv3d_1 = Conv3DBlock(in_channels, base_channels,
                                   kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.conv3d_2 = Conv3DBlock(base_channels, base_channels*2,
                                   kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.conv3d_3 = Conv3DBlock(base_channels*2, base_channels*4,
                                   kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        # 注意力机制
        self.spectral_attention = SpectralAttention(base_channels*4)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        # 3D卷积特征提取
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        
        # 应用注意力机制
        x = self.spectral_attention(x)
        x = self.spatial_attention(x)
        
        return x

class HSITransformerEncoder(nn.Module):
    """高光谱图像的Transformer编码器"""
    def __init__(self, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
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
        # 自注意力
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class HSIEncoder(nn.Module):
    """
    高光谱图像编码器 - 3D-CNN + Transformer混合架构
    
    Args:
        hsi_channels: 高光谱波段数 (如224个波段)
        spatial_size: 空间维度大小 (如64x64)
        embed_dim: 嵌入维度
        depth: Transformer深度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        dropout: dropout率
    """
    def __init__(
        self,
        hsi_channels=224,
        spatial_size=64,
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
        
        # 3D卷积特征提取器
        self.feature_extractor = SpectralSpatialFeatureExtractor(
            in_channels=1,
            base_channels=32
        )
        
        # 计算3D卷积后的特征维度
        # 假设经过3D卷积后，光谱维度减半，空间维度不变
        conv_out_channels = 128
        reduced_spectral_dim = hsi_channels // 4
        
        # 将3D特征展平并投影到嵌入维度
        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                conv_out_channels * reduced_spectral_dim,
                embed_dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Patch embedding for transformer
        self.patch_embed = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Transformer编码器
        self.transformer = HSITransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入高光谱图像 [B, C, H, W]
               其中C是光谱波段数，H和W是空间维度
               
        Returns:
            features: 编码后的特征 [B, embed_dim]
        """
        B, C, H, W = x.shape
        
        # 重塑为3D卷积的输入格式 [B, 1, C, H, W]
        x = x.unsqueeze(1)
        
        # 3D卷积特征提取
        x = self.feature_extractor(x)
        
        # 重塑为2D格式进行投影
        # x shape: [B, channels, depth, H, W]
        B, ch, d, h, w = x.shape
        x = rearrange(x, 'b c d h w -> b (c d) h w')
        
        # 特征投影
        x = self.feature_projection(x)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # 展平为序列
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [B, embed_dim, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [B, embed_dim]
        
        return x

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = HSIEncoder(
        hsi_channels=224,
        spatial_size=64,
        embed_dim=768,
        depth=6,
        num_heads=12
    )
    
    # 测试输入 - 高光谱图像
    x = torch.randn(2, 224, 64, 64)
    
    # 前向传播
    features = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出特征形状: {features.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")