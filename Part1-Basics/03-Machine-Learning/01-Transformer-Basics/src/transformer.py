#!/usr/bin/env python3
"""
完整的Transformer模型实现
Complete Transformer Model Implementation

作者 | Author: First-Principles ML Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 导入我们实现的组件
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    前馈神经网络 | Feed-Forward Network

    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: 模型维度
        d_ff: 前馈网络隐藏层维度（通常是d_model的4倍）
        dropout: Dropout概率
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer编码器层 | Transformer Encoder Layer

    包含:
    1. 多头自注意力
    2. Add & Norm
    3. 前馈网络
    4. Add & Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()

        # 多头注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        前向传播

        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len]

        Returns:
            [batch_size, seq_len, d_model]
        """
        # 多头自注意力 + 残差连接 + Layer Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Transformer解码器层 | Transformer Decoder Layer

    包含:
    1. Masked多头自注意力
    2. Add & Norm
    3. 编码器-解码器注意力（交叉注意力）
    4. Add & Norm
    5. 前馈网络
    6. Add & Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        # Masked自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 交叉注意力（编码器-解码器注意力）
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ):
        """
        前向传播

        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列mask [batch_size, 1, src_seq_len]
            tgt_mask: 目标序列mask [batch_size, tgt_seq_len, tgt_seq_len]

        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        # Masked自注意力
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力（Query来自解码器，Key和Value来自编码器）
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型

    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 编码器和解码器的层数
        d_ff: 前馈网络隐藏层维度
        max_seq_len: 最大序列长度
        dropout: Dropout概率
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 解码器层堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """
        编码器前向传播

        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列mask [batch_size, 1, src_seq_len]

        Returns:
            编码器输出 [batch_size, src_seq_len, d_model]
        """
        # 词嵌入 + 缩放 + 位置编码
        x = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)

        # 通过所有编码器层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ):
        """
        解码器前向传播

        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列mask
            tgt_mask: 目标序列mask

        Returns:
            解码器输出 [batch_size, tgt_seq_len, d_model]
        """
        # 词嵌入 + 缩放 + 位置编码
        x = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)

        # 通过所有解码器层
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ):
        """
        完整前向传播

        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
            src_mask: 源序列mask
            tgt_mask: 目标序列mask

        Returns:
            logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 编码
        encoder_output = self.encode(src, src_mask)

        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 输出投影
        logits = self.output_projection(decoder_output)

        return logits


def create_masks(src, tgt, pad_idx=0):
    """
    创建源序列和目标序列的mask

    Args:
        src: 源序列 [batch_size, src_seq_len]
        tgt: 目标序列 [batch_size, tgt_seq_len]
        pad_idx: padding的索引

    Returns:
        src_mask: [batch_size, 1, src_seq_len]
        tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
    """
    # 源序列mask（padding mask）
    src_mask = (src != pad_idx).unsqueeze(1)  # [batch_size, 1, src_seq_len]

    # 目标序列mask（padding + future mask）
    tgt_seq_len = tgt.size(1)

    # Padding mask
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_seq_len]

    # Future mask（下三角矩阵）
    tgt_future_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).unsqueeze(0).unsqueeze(0)
    # [1, 1, tgt_seq_len, tgt_seq_len]

    # 组合两个mask
    tgt_mask = tgt_padding_mask & tgt_future_mask.to(tgt.device)
    tgt_mask = tgt_mask.squeeze(1)  # [batch_size, tgt_seq_len, tgt_seq_len]

    return src_mask, tgt_mask


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Transformer模型测试 | Transformer Model Test")
    print("=" * 70)
    print()

    # 设置参数
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048

    # 创建模型
    print("创建Transformer模型...")
    print("Creating Transformer model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"Total parameters: {total_params:,}")
    print()

    # 创建随机输入
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    print(f"源序列形状: {src.shape}")
    print(f"Source sequence shape: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"Target sequence shape: {tgt.shape}")
    print()

    # 创建masks
    src_mask, tgt_mask = create_masks(src, tgt)
    print(f"源mask形状: {src_mask.shape}")
    print(f"目标mask形状: {tgt_mask.shape}")
    print()

    # 前向传播
    print("执行前向传播...")
    print("Performing forward pass...")
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)

    print(f"输出形状: {output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"预期形状: [{batch_size}, {tgt_seq_len}, {tgt_vocab_size}]")
    print(f"Expected shape: [{batch_size}, {tgt_seq_len}, {tgt_vocab_size}]")
    print()

    print("测试完成！")
    print("Test completed!")
