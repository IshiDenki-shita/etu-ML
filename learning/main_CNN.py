import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import csv
from PIL import Image
import numpy as np
import random
import json
import glob
from torchvision import transforms

# -----------------------------
# ひらがなセット
# -----------------------------
chars_url = "supports/chars.txt"
menu_url = "supports/menu.csv"
FONT_PATH = "C:/Windows/Fonts/HGRPP1.TTC"
def make_chars():
    chars = ""
    with open('learning/dataset/labels.json', encoding="utf-8") as f:
        labels = json.load(f)
        for n in range(len(labels)):
            chars += labels[str(n)]
    return chars

def write_chars_txt(chars):
    f = open(chars_url,"w",encoding="utf-8")
    f.write(chars)
    f.close()

chars = make_chars()
write_chars_txt(chars=chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# -----------------------------
# 画像読み込み
# -----------------------------
IMG_DICT = None

def load_images_once():
    global IMG_DICT
    if IMG_DICT is not None:
        return IMG_DICT

    IMG_DICT = {}
    with open('learning/dataset/labels.json', encoding="utf-8") as f:
        labels = json.load(f)

    for ch in range(len(labels)):
        img_url = f"learning/dataset/train_photos/{ch}/*.jp*g"
        for file in glob.glob(img_url):
            img = Image.open(file).rotate(90).resize((64, 64)).convert("L")
            img = np.array(img)
            binary_np = (img > 127).astype(np.uint8)
            result = Image.fromarray(binary_np * 255) # 0-255の範囲に戻して保存
            IMG_DICT.setdefault(labels[str(ch)], []).append(result)

    return IMG_DICT

# -----------------------------
# ランダムにズーム
# -----------------------------
def random_zoom_and_shift(img):
    # img: numpy (64,64) or PIL Image
    if isinstance(img, np.ndarray):
        imgg = Image.fromarray((img * 255).astype(np.uint8))
    else:
        img = img
    transform = transforms.RandomResizedCrop(
        size=64,
        scale=(0.6, 1.0),
        ratio=(0.9, 1.1)
    )
    return np.array(transform(img)) / 255.0


# -----------------------------
# ノイズ生成
# -----------------------------
def add_noise(img):
    noise = np.random.normal(0, 0.05, img.shape)  # 平均0, 標準偏差0.05
    img = img + noise
    img = np.clip(img, 0, 1)
    return img

# -----------------------------
# モザイク処理
# -----------------------------
def add_mosaic(img, min_size=7, max_size=12):
    # img: numpy (64,64) or PIL Image
    if isinstance(img, np.ndarray):
        pil = Image.fromarray((img * 255).astype(np.uint8))
    else:
        pil = img

    # ランダムな縮小サイズ（モザイクの粗さ）
    mosaic_size = random.randint(min_size, max_size)

    # 縮小 → 拡大
    small = pil.resize((mosaic_size, mosaic_size), Image.NEAREST)
    mosaic = small.resize((64, 64), Image.NEAREST)

    return np.array(mosaic) / 255.0

# -----------------------------
# 回転処理
# -----------------------------
def add_rotate(img,min_angle=-10, max_angle=10):
    if isinstance(img, np.ndarray):
        pil = Image.fromarray((img * 255).astype(np.uint8))
    else:
        pil = img
    angle = random.randint(min_angle, max_angle)
    img = pil.rotate(angle, expand=True, fillcolor='white')
    img = img.resize((64, 64))
    return np.array(img) / 255.0

# -----------------------------
# モザイク画像生成
# -----------------------------
def generate_handwritten_2(ch):
    img_dict = load_images_once()
    img = random.choice(img_dict[ch])
    img = add_rotate(img=img)
    img = random_zoom_and_shift(Image.fromarray((img*255).astype(np.uint8)))
    #img = add_mosaic(img=img)
    #img = add_noise(img=img)
    img = 1.0 - img
    return(img)

if __name__ in "__main__":
    for _ in range(5):
        img_array = generate_handwritten_2("婆") * 255
        img = Image.fromarray(np.uint8(img_array))
        img.show()

# -----------------------------
# Dataset
# -----------------------------
class HiraganaDataset(Dataset):
    def __init__(self, size=5000):
        self.data = []
        self.labels = []
        for _ in range(size):
            ch = random.choice(chars)
            img = generate_handwritten_2(ch)
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
            nn.Linear(256, len(chars)),
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 学習関数
# -----------------------------
def train_model(epochs=5, size=15000):
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
    model = train_model()
    torch.save(model.state_dict(), "supports/hiragana_cnn.pth")
    print("モデルを保存しました")
