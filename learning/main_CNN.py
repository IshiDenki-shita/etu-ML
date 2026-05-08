import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import csv
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random

# -----------------------------
# ひらがなセット
# -----------------------------
chars_url = "supports/chars.txt"
menu_url = "supports/menu.csv"
FONT_PATH = "C:/Windows/Fonts/HGRPP1.TTC"


def make_chars():
    with open(menu_url, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        dct = []
        for row in reader:
            dct.append("".join(row))
    s = "".join(dct)
    chars = "".join(dict.fromkeys(s))
    return str(chars)


def write_chars_txt(chars):
    f = open(chars_url, "w", encoding="utf-8")
    f.write(chars)
    f.close()


chars = make_chars()
write_chars_txt(chars=chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# -----------------------------
# 手書き風ひらがな画像を作る
# -----------------------------


def generate_handwritten(ch):
    img = Image.new("L", (64, 64), 255)  # 64*64の画像　背景を255(白)で塗る
    draw = ImageDraw.Draw(img)  # imgに文字を書けるようにする
    font = ImageFont.truetype(
        FONT_PATH, random.randint(40, 60)
    )  # meiryobでサイズが40~60

    # ランダム位置
    x = random.randint(0, 5)
    y = random.randint(0, 5)

    draw.text((x, y), ch, font=font, fill=0)

    # 手書き風ノイズ
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.5)))

    # 傾き
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, fillcolor=255)

    # ぼかし処理
    intensity = 3
    small = img.resize((round(img.width / intensity), round(img.height / intensity)))
    img = small.resize((img.width, img.height), resample=Image.Resampling.BILINEAR)

    # CNN用に64*64に縮小
    # img = img.resize((64, 64), resample=Image.Resampling.BILINEAR)
    return np.array(img) / 255.0


if __name__ in "__main__":
    img_array = generate_handwritten("う") * 255.0
    img = Image.fromarray(np.uint8(img_array))
    img.show()


# -----------------------------
# Dataset
# -----------------------------
class HiraganaDataset(Dataset):
    def __init__(self, size=3000):
        self.data = []
        self.labels = []
        for _ in range(size):
            ch = random.choice(chars)
            img = generate_handwritten(ch)
            self.data.append(img)
            self.labels.append(char_to_idx[ch])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y


# -----------------------------
# CNNモデル
# -----------------------------
class HiraganaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, len(chars)),
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# 学習関数
# -----------------------------
def train_model(epochs=5, size=4500):
    dataset = HiraganaDataset(size=size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = HiraganaCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("学習開始…")
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.2f}")

    print("学習完了！")
    return model


if __name__ == "__main__":
    img_array = generate_handwritten("噌") * 255.0
    img = Image.fromarray(np.uint8(img_array))
    img.show()
    model = train_model()
    torch.save(model.state_dict(), "supports/hiragana_cnn.pth")
    print("モデルを保存しました")
