import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import numpy as np
import glob
import json

# -----------------------------
# char_to_idx, idx_to_char
# -----------------------------
class dictionary():
    def __init__(self):
        self.etl_url_list = ["ETL8B2C1_unpack",
                             "ETL8B2C2_unpack",
                             "ETL8B2C3_unpack"]
    def make_dict(self):
        idx_to_char = {}
        csv_idx = 0
        for etl_url in self.etl_url_list:
            csv_url = "learning/dataset/" + etl_url + "/meta.csv"

            with open(csv_url, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # ヘッダーをスキップ
                rows = list(reader)

                for i in range(0, len(rows), 160):
                    row = rows[i]
                    char = row[1]
                    idx_to_char[int(csv_idx / 160)] = char
                    csv_idx += 160
        
        with open("supports/chars_dict.json", "w", encoding="utf-8") as f:
            json.dump(idx_to_char, f, ensure_ascii=False, indent=1)

        return idx_to_char

# -----------------------------
# Dataset
# -----------------------------
class ETL_Dataset(Dataset):
    def __init__(self):
        self.files = []
        self.labels = []
        current_idx = 0
        etl_url_list = dictionary().etl_url_list
        for etl_url in etl_url_list:
            img_url = "learning/dataset/" + etl_url + "/*.png"
            for file in sorted(glob.glob(img_url)):
                self.files.append(file)
                self.labels.append(int(current_idx / 160))
                current_idx += 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file).resize((64, 64)).convert("L")
        img = np.array(img) / 255.0
        x = torch.tensor(img).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y


# -----------------------------
# CNNモデル
# -----------------------------
class HiraganaCNN(nn.Module):
    def __init__(self, num_crasses):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_crasses),
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 学習関数
# -----------------------------
class train():
    def __init__(self):
        self.idx_to_char = dictionary().make_dict()
        self.num_classes = len(self.idx_to_char)
        print("クラス数:", self.num_classes)

        self.dataset = ETL_Dataset()
        self.loader = DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=4)

        self.model = HiraganaCNN(num_crasses=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self,epochs=5):
        print("学習開始…")
        for epoch in range(epochs):
            total_loss = 0
            for x, y in self.loader:
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.2f}")

        print("学習完了！")
    
    def run(self):
        self.train_model()
        torch.save(self.model.state_dict(), "supports/hiragana_cnn.pth")
        print("モデルを保存しました")

if __name__ == "__main__":
    train().run()
