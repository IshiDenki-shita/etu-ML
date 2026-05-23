import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

# =============================
# Dataset（最速版）
# =============================
class ETLDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.images = np.load(img_path, mmap_mode="r")   # uint8 (N,1,64,64)
        self.labels = np.load(label_path, mmap_mode="r") # int32 (N,)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # uint8 → float32 に変換（最速）
        x = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# =============================
# Residualブロック
# =============================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(self.conv(x) + x)


# =============================
# DEブロック
# =============================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



# =============================
# CNN（軽量・高速）
# =============================
class HiraganaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(32), 
            SEBlock(32),
            nn.MaxPool2d(2),  # 32×32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64), 
            SEBlock(64),
            nn.MaxPool2d(2),  # 16×16

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# =============================
# 学習ループ
# =============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


# =============================
# 評価
# =============================
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# =============================
# メイン処理
# =============================
def main(epochs=30, batch_size=128):
    img_path = "learning/dataset/etl_images.npy"
    label_path = "learning/dataset/etl_labels.npy"

    # Dataset
    dataset = ETLDataset(img_path, label_path)

    # 8:2 に分割
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # クラス数
    num_classes = len(np.unique(dataset.labels))

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # モデル
    model = HiraganaCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # 学習
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {loss:.2f}")
        print(f"  Test Accuracy: {acc:.4f}")

    # 保存
    torch.save(model.state_dict(), "supports/hiragana_cnn.pth")
    print("モデル保存完了")


if __name__ == "__main__":
    main()
