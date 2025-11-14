#!/usr/bin/env python3
"""
位置编码 | Positional Encoding

Transformer模型没有循环结构，因此需要位置编码来注入位置信息。
本模块实现了原始论文中的正弦/余弦位置编码。

The Transformer has no recurrent structure, so positional encoding is needed
to inject position information. This module implements the sine/cosine
positional encoding from the original paper.

公式 | Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

作者 | Author: First-Principles ML Course
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块

    Args:
        d_model: 模型维度（嵌入维度）
        max_len: 最大序列长度
        dropout: Dropout概率

    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 位置索引 [max_len, 1]
        # Position indices [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 除数项: 10000^(2i/d_model)
        # Divisor term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 应用正弦和余弦函数
        # Apply sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos

        # 添加batch维度 [1, max_len, d_model]
        # Add batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为buffer（不是参数，不需要梯度）
        # Register as buffer (not a parameter, doesn't need gradients)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        # 将位置编码添加到输入
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    可学习的位置编码

    与固定的正弦/余弦编码不同，这个版本的位置编码是可训练的。
    某些任务中，可学习的位置编码可能效果更好。

    Unlike fixed sine/cosine encoding, this version uses trainable
    position embeddings. May work better for certain tasks.

    Args:
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout概率
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 可学习的位置嵌入
        # Learnable position embeddings
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()

        # 生成位置索引
        # Generate position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # 获取位置嵌入并添加到输入
        # Get position embeddings and add to input
        x = x + self.pe(positions)
        return self.dropout(x)


def visualize_positional_encoding(d_model: int = 512, max_len: int = 100):
    """
    可视化位置编码

    Args:
        d_model: 模型维度
        max_len: 最大序列长度
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建位置编码
    pe = PositionalEncoding(d_model, max_len)

    # 获取位置编码矩阵（去掉batch维度）
    encoding = pe.pe.squeeze(0).numpy()

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制热力图
    plt.subplot(2, 1, 1)
    plt.imshow(encoding.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Heatmap')
    plt.colorbar()

    # 绘制几个维度的曲线
    plt.subplot(2, 1, 2)
    positions = np.arange(max_len)
    for i in [0, 1, d_model//2, d_model//2+1]:
        plt.plot(positions, encoding[:, i], label=f'Dim {i}')

    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title('Positional Encoding for Selected Dimensions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/positional_encoding.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("位置编码可视化已保存")
    print("Positional encoding visualization saved")


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("位置编码测试 | Positional Encoding Test")
    print("=" * 70)
    print()

    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    print(f"Input shape: {x.shape}")
    print()

    # 测试固定位置编码
    print("测试固定位置编码...")
    print("Testing fixed positional encoding...")
    pe_fixed = PositionalEncoding(d_model)
    output_fixed = pe_fixed(x)
    print(f"输出形状: {output_fixed.shape}")
    print(f"Output shape: {output_fixed.shape}")
    print()

    # 测试可学习位置编码
    print("测试可学习位置编码...")
    print("Testing learned positional encoding...")
    pe_learned = LearnedPositionalEncoding(d_model)
    output_learned = pe_learned(x)
    print(f"输出形状: {output_learned.shape}")
    print(f"Output shape: {output_learned.shape}")
    print()

    # 可视化
    print("生成可视化...")
    print("Generating visualization...")
    try:
        import os
        os.makedirs('../results', exist_ok=True)
        visualize_positional_encoding(d_model=256, max_len=100)
    except ImportError:
        print("需要matplotlib来可视化")
        print("matplotlib required for visualization")

    print()
    print("测试完成！")
    print("Test completed!")
