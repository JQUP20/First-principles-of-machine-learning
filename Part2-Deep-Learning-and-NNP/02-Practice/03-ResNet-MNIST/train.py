#!/usr/bin/env python3
"""
ResNet MNIST训练脚本
ResNet Training Script for MNIST
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# 导入模型（如果文件存在）
try:
    from models.resnet import ResNet18
except ImportError:
    print("Warning: models.resnet not found, using inline definition")

    # 内联定义ResNet（简化版）
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet18(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.in_channels = 64

            self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)  # MNIST是单通道
            self.bn1 = nn.BatchNorm2d(64)

            self.layer1 = self._make_layer(64, 2, stride=1)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)

            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, out_channels, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(BasicBlock(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out


def get_data_loaders(batch_size, data_dir='./data'):
    """
    获取MNIST数据加载器
    Get MNIST data loaders
    """
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # ResNet期望32x32输入
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    Train one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def test(model, test_loader, criterion, device):
    """
    测试模型
    Test the model
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def plot_curves(train_losses, train_accs, test_losses, test_accs, save_dir):
    """
    绘制训练曲线
    Plot training curves
    """
    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, test_accs, 'r-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    print(f"Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train ResNet on MNIST')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='device (cuda or cpu)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='save directory')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    args = parser.parse_args()

    print("=" * 60)
    print("ResNet MNIST训练")
    print("ResNet MNIST Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("\nLoading data...")
    train_loader, test_loader = get_data_loaders(args.batch_size, args.data_dir)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # 创建模型
    print("\nCreating model...")
    model = ResNet18(num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print("\nTraining...")
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        # 训练
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 学习率调度
        scheduler.step()

        # 打印统计
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ★ Best model saved! (Acc: {best_acc:.2f}%)")

    # 绘制训练曲线
    print("\nPlotting training curves...")
    plot_curves(train_losses, train_accs, test_losses, test_accs, 'results')

    # 最终结果
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Model saved in: {args.save_dir}/best_model.pth")
    print(f"Results saved in: results/")


if __name__ == '__main__':
    main()
