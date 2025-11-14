#!/usr/bin/env python3
"""
多头注意力机制 | Multi-Head Attention Mechanism

多头注意力是Transformer的核心组件，允许模型同时关注来自不同表示子空间的信息。

Multi-head attention is the core component of Transformer, allowing the model
to jointly attend to information from different representation subspaces.

公式 | Formulas:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

作者 | Author: First-Principles ML Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力 | Scaled Dot-Product Attention

    计算公式:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        dropout: Dropout概率
    """

    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        前向传播

        Args:
            query: [batch_size, n_heads, seq_len_q, d_k]
            key:   [batch_size, n_heads, seq_len_k, d_k]
            value: [batch_size, n_heads, seq_len_v, d_v]
            mask:  [batch_size, 1, 1, seq_len_k] 或 [batch_size, 1, seq_len_q, seq_len_k]

        Returns:
            output: [batch_size, n_heads, seq_len_q, d_v]
            attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)

        # 计算注意力分数: QK^T / √d_k
        # Compute attention scores: QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用mask（用于padding和future masking）
        # Apply mask (for padding and future masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 应用softmax得到注意力权重
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和value
        # Weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 | Multi-Head Attention

    将输入投影到多个头，每个头独立计算注意力，然后拼接。

    Args:
        d_model: 模型维度
        n_heads: 头的数量
        dropout: Dropout概率
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q, K, V的线性投影层
        # Linear projection layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影层
        # Output projection layer
        self.W_o = nn.Linear(d_model, d_model)

        # 缩放点积注意力
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将最后一个维度分割成(n_heads, d_k)

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, n_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        合并多头

        Args:
            x: [batch_size, n_heads, seq_len, d_k]

        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        前向传播

        Args:
            query: [batch_size, seq_len_q, d_model]
            key:   [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask:  [batch_size, 1, seq_len] 或 [batch_size, seq_len_q, seq_len_k]

        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # 线性投影 Linear projections
        Q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.W_v(value)  # [batch_size, seq_len_v, d_model]

        # 分割成多头 Split into multiple heads
        Q = self.split_heads(Q)  # [batch_size, n_heads, seq_len_q, d_k]
        K = self.split_heads(K)  # [batch_size, n_heads, seq_len_k, d_k]
        V = self.split_heads(V)  # [batch_size, n_heads, seq_len_v, d_k]

        # 调整mask的形状（如果提供）
        # Adjust mask shape if provided
        if mask is not None:
            if mask.dim() == 3:
                # [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1)

        # 应用注意力 Apply attention
        x, attention_weights = self.attention(Q, K, V, mask)
        # x: [batch_size, n_heads, seq_len_q, d_k]
        # attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]

        # 合并多头 Combine heads
        x = self.combine_heads(x)  # [batch_size, seq_len_q, d_model]

        # 最后的线性投影 Final linear projection
        output = self.W_o(x)
        output = self.dropout(output)

        return output, attention_weights


def visualize_attention_weights(attention_weights: torch.Tensor, save_path: str = None):
    """
    可视化注意力权重

    Args:
        attention_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 取第一个样本
    attn = attention_weights[0].detach().cpu().numpy()
    n_heads = attn.shape[0]

    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(min(n_heads, 8)):
        im = axes[i].imshow(attn[i], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])

    # 隐藏多余的子图
    for i in range(n_heads, 8):
        axes[i].axis('off')

    plt.suptitle('Multi-Head Attention Weights', fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力权重可视化已保存到: {save_path}")
        print(f"Attention weights visualization saved to: {save_path}")

    plt.show()


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("多头注意力测试 | Multi-Head Attention Test")
    print("=" * 70)
    print()

    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    # 创建随机输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    print(f"Query形状: {query.shape}")
    print(f"Key形状: {key.shape}")
    print(f"Value形状: {value.shape}")
    print()

    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, n_heads)

    # 前向传播
    print("执行前向传播...")
    print("Performing forward pass...")
    output, attention_weights = mha(query, key, value)

    print(f"输出形状: {output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print()

    # 测试带mask的情况
    print("测试带mask的情况...")
    print("Testing with mask...")
    mask = torch.ones(batch_size, 1, seq_len)
    mask[:, :, 5:] = 0  # Mask掉后半部分
    output_masked, attention_weights_masked = mha(query, key, value, mask)
    print(f"Masked输出形状: {output_masked.shape}")
    print(f"Masked output shape: {output_masked.shape}")
    print()

    # 可视化注意力权重
    print("生成注意力权重可视化...")
    print("Generating attention weights visualization...")
    try:
        import os
        os.makedirs('../results', exist_ok=True)
        visualize_attention_weights(
            attention_weights,
            save_path='../results/attention_weights.png'
        )
    except ImportError:
        print("需要matplotlib来可视化")
        print("matplotlib required for visualization")

    print()
    print("测试完成！")
    print("Test completed!")

    # 打印参数数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"Total parameters: {total_params:,}")
