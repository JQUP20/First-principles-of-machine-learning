#!/usr/bin/env python3
"""
使用Transformer进行时间序列预测
Time Series Forecasting with Transformer

本示例展示如何使用Transformer模型预测时间序列数据（如股票价格、温度等）

This example demonstrates how to use Transformer model for time series
forecasting (e.g., stock prices, temperature, etc.)

作者 | Author: First-Principles ML Course
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention


class TimeSeriesTransformer(nn.Module):
    """
    用于时间序列预测的Transformer模型

    简化版本：只使用编码器部分
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # 输出层
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]

        Returns:
            [batch_size, seq_len, input_dim]
        """
        # 输入投影
        x = self.input_projection(x)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer编码器
        x = self.transformer_encoder(x)

        # 输出投影
        x = self.output_projection(x)

        return x


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集

    Args:
        data: 时间序列数据 [total_length, num_features]
        seq_len: 输入序列长度
        pred_len: 预测长度
    """

    def __init__(self, data: np.ndarray, seq_len: int = 50, pred_len: int = 10):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 输入：过去seq_len个时间步
        x = self.data[idx:idx + self.seq_len]

        # 目标：接下来pred_len个时间步
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]

        return x, y


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 1
) -> np.ndarray:
    """
    生成合成时间序列数据（正弦波 + 噪声）

    Args:
        n_samples: 样本数量
        n_features: 特征数量

    Returns:
        时间序列数据 [n_samples, n_features]
    """
    t = np.linspace(0, 100, n_samples)
    data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # 多个正弦波的叠加
        freq1 = 0.1 + i * 0.05
        freq2 = 0.3 + i * 0.03
        wave = (np.sin(2 * np.pi * freq1 * t) +
                0.5 * np.sin(2 * np.pi * freq2 * t) +
                0.2 * np.random.randn(n_samples))
        data[:, i] = wave

    return data


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    标准化数据

    Args:
        data: 原始数据

    Returns:
        normalized_data, mean, std
    """
    mean = data.mean()
    std = data.std()
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_data(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """反标准化"""
    return data * std + mean


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练模型

    Args:
        model: Transformer模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        n_epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses = []
    val_losses = []

    print("=" * 70)
    print("开始训练 | Starting Training")
    print("=" * 70)
    print()

    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            outputs = model(batch_x)

            # 只计算最后pred_len步的损失
            pred_len = batch_y.size(1)
            loss = criterion(outputs[:, -pred_len:, :], batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                pred_len = batch_y.size(1)
                loss = criterion(outputs[:, -pred_len:, :], batch_y)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")

    print()
    print("训练完成！| Training completed!")
    print()

    return train_losses, val_losses


def evaluate_and_visualize(
    model: nn.Module,
    test_data: np.ndarray,
    seq_len: int,
    pred_len: int,
    mean: float,
    std: float,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    评估模型并可视化结果
    """
    model.eval()
    model = model.to(device)

    # 选择一个测试样本
    idx = 100
    x = test_data[idx:idx + seq_len]
    y_true = test_data[idx + seq_len:idx + seq_len + pred_len]

    # 预测
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        y_pred = model(x_tensor)
        y_pred = y_pred[0, -pred_len:, :].cpu().numpy()

    # 反标准化
    x = denormalize_data(x, mean, std)
    y_true = denormalize_data(y_true, mean, std)
    y_pred = denormalize_data(y_pred, mean, std)

    # 计算误差
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    print("评估结果 | Evaluation Results")
    print("-" * 70)
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print()

    # 可视化
    plt.figure(figsize=(14, 6))

    # 绘制历史数据
    hist_len = min(seq_len, 100)
    hist_x = np.arange(0, hist_len)
    plt.plot(hist_x, x[-hist_len:, 0], 'b-', label='Historical Data', linewidth=2)

    # 绘制真实值和预测值
    pred_x = np.arange(hist_len, hist_len + pred_len)
    plt.plot(pred_x, y_true[:, 0], 'g-', label='Ground Truth', linewidth=2, marker='o')
    plt.plot(pred_x, y_pred[:, 0], 'r--', label='Prediction', linewidth=2, marker='x')

    # 添加垂直线分隔历史和预测
    plt.axvline(x=hist_len, color='gray', linestyle=':', alpha=0.7)

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Time Series Forecasting with Transformer', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 添加文本框显示误差
    textstr = f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()

    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/time_series_prediction.png', dpi=300, bbox_inches='tight')
    print("预测结果已保存到: ../results/time_series_prediction.png")
    print("Prediction results saved to: ../results/time_series_prediction.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("Transformer时间序列预测示例")
    print("Time Series Forecasting with Transformer")
    print("=" * 70)
    print()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 参数设置
    n_samples = 1000
    n_features = 1
    seq_len = 50
    pred_len = 10
    batch_size = 32
    n_epochs = 100
    learning_rate = 0.001

    # 生成数据
    print("生成合成时间序列数据...")
    print("Generating synthetic time series data...")
    data = generate_synthetic_data(n_samples, n_features)

    # 标准化
    data_normalized, mean, std = normalize_data(data)

    # 划分训练集、验证集、测试集
    train_size = int(0.7 * len(data_normalized))
    val_size = int(0.15 * len(data_normalized))

    train_data = data_normalized[:train_size]
    val_data = data_normalized[train_size:train_size + val_size]
    test_data = data_normalized[train_size + val_size:]

    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    print()

    # 创建数据加载器
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    print("创建Transformer模型...")
    print("Creating Transformer model...")
    model = TimeSeriesTransformer(
        input_dim=n_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"Model parameters: {total_params:,}")
    print()

    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/training_history.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存到: ../results/training_history.png")
    print("Training history saved to: ../results/training_history.png")
    print()

    # 评估和可视化
    evaluate_and_visualize(model, test_data, seq_len, pred_len, mean, std)

    # 保存模型
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/time_series_transformer.pth')
    print()
    print("模型已保存到: ../models/time_series_transformer.pth")
    print("Model saved to: ../models/time_series_transformer.pth")
    print()

    print("=" * 70)
    print("示例完成！| Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
