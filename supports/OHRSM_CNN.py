import Levenshtein
import csv
import torch
import torch.nn as nn


class menu_classifier:
    def __init__(self):
        self.chars_url = "supports/chars.txt"
        self.menu_url = "supports/menu.csv"
        self.cnn_pth_url = "supports/hiragana_cnn.pth"

        with open(self.chars_url, "r", encoding="utf-8") as f:
            data = f.read()
        self.idx_to_char = {i: c for i, c in enumerate(data)}

        with open(self.menu_url, encoding="utf-8") as f:
            reader = csv.reader(f)
            self.menu_list = [row[0] for row in reader if row]

        self.model = HiraganaCNN()
        self.model.load_state_dict(torch.load(self.cnn_pth_url, map_location="cpu"))
        self.model.eval()

    def correct(self, menu, threshold=10):
        best = min(self.menu_list, key=lambda x: Levenshtein.distance(menu, x))
        dist = Levenshtein.distance(menu, best)
        return best if dist <= threshold else menu

    def predict_char(self, img):
        x = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        out = self.model(x)
        pred = out.argmax(1).item()
        return self.idx_to_char[pred]

    def predict_sentence(self, cell):
        chars = [self.predict_char(img) for img in cell]
        line = "".join(chars)
        return self.correct(line)

    def run(self, cells):
        res = []
        for cell in cells:
            res.append(self.predict_sentence(cell))
        return res


class HiraganaCNN(nn.Module):

    def __init__(self):
        self.menu_url = "supports/menu.csv"
        self.chars = self.make_chars()

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
            nn.Linear(128, len(self.chars)),
        )

    def make_chars(self):
        with open(self.menu_url, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            dct = []
            for row in reader:
                dct.append("".join(row))
        s = "".join(dct)
        chars = "".join(dict.fromkeys(s))
        return str(chars)

    def write_chars_txt(self, chars):
        f = open(self.menu_url, "w", encoding="utf-8")
        f.write(chars)
        f.close()

    def forward(self, x):
        return self.model(x)
