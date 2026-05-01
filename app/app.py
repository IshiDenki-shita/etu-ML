"""
purpose: extraction characters's binary img from sentence img
author: Mats

info: I uploaded a doc explaining about how deal with charless area(appear later)
      on our sharing drive
"""

import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks
import Levenshtein
import csv
import torch
import torch.nn as nn
import json
from PIL import Image
import random


class Config_utl:
    IMG_ROOT = Path("photos/")
    url_saito_cafe = "http://162.43.43.163:8080/api/v1/cafe"


@dataclass
class Config_seg:
    MIN_CHAR_WIDTH = 60  # for char detection
    CHAR_WIDTH_RATIO = 0.05
    SMOOTH_KERNEL = 5
    MAX_BLANK_DENSITY = 0.03
    MIN_CHAR_DENSITY = 0.08
    ARROW_WID_RATIO = 1 / 30
    ARROW_AVE_SURFACE = 450
    THIN_NOISE_WIDTH = 3  # for partial blank detection
    UPPER_BLANK_RATIO = 0.5
    LEFT_BLANK_RATIO = 0.05
    RIGHT_BLANK_RATIO = 0.5
    NOISE_HEIGHT = 5
    IGNORE_WID = 10
    PHOTO_HW = (64, 64)  # for regulate sizes of photos


class MLutility:
    def __init__(self):
        self.cfg = Config_utl()

    """
    File operation
    """

    def take_cell_imgs(self):
        imgs_root = self.cfg.IMG_ROOT
        dir_receives = sorted(list(imgs_root.iterdir()))[-1]
        img_dirs = sorted(list(dir_receives.iterdir()))
        
        imgs = []
        for i, dir_cap in enumerate(img_dirs):
            img = cv2.imread(dir_cap)
            if img is None:
                print(f"{i + 1}番目の画像を取得できませんでした", end="\n\n")
            imgs.append(img)

        print(f"{len(imgs)} imgs was taken")
        return imgs

    """
    Send JSON
    """

    def send_menu_json_to_saito(self, menus: list[tuple]) -> None:
        cafe_dict = self.format_mail_dict(menus=menus)
        res = requests.post(self.cfg.url_saito_cafe, json=cafe_dict)

        print("\nステータスコード\n")
        print(res.status_code)

        print("\nレスポンス本文\n")
        print(res.text)

    def format_mail_dict(self, menus: list[tuple]):
        formatted_menus = []
        for menu in menus:
            formatted_menus.append({"name": menu[0], "date": menu[1], "price": 500})
        mail = {
            "generated_at": self.now_jst_iso8601_seconds(),
            "menus": formatted_menus,
        }
        return mail

    # 斎藤VPSに送るJSONのgenerated_atの日付のフォーマットを固定する。
    def now_jst_iso8601_seconds(self) -> str:
        """
        斎藤VPS指定フォーマット:
        2024-12-02T14:30:00+09:00
        """
        JST = timezone(timedelta(hours=9))
        dt = datetime.now(JST).replace(microsecond=0)  # 秒までに丸める
        s = dt.isoformat()  # 'YYYY-MM-DDTHH:MM:SS+09:00'
        # 念のためオフセットが +0900 のようになったケースを +09:00 に補正
        if len(s) >= 5 and (s[-5] in ["+", "-"]) and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        return s


class CharacterSegmenter:
    def __init__(self, config=None, debug=None):
        self.cfg = config or Config_seg()
        self.debug = debug

    """
    Pipeline
    """

    def run(self, img):
        char_areas, chars = [], []

        binary = self.preprocess(img)

        if np.sum(binary) / binary.size < self.cfg.MIN_CHAR_DENSITY:
            return self.check_is_arrow(binary)
        else:
            binary = self.trim_upper_sides_blank(binary=binary)

            char_areas = self.detect_char_areas(binary)

            char_areas = self.filter_characters(binary, char_areas)

            char_areas = self.merge_small_areas(char_areas)

            chars = self.area_to_binary(binary, char_areas)

            # chars = self.trim_under_blank(chars)

            chars = self.regulate_size(chars)

            self.visualize(
                img=img, proj=self.compute_projection(binary=binary)[0], chars=chars
            )

            return chars

    """
    Preprocess
    """

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = (binary > 0).astype(np.uint8)
        return binary

    def compute_projection(self, binary):
        proj_v = np.sum(binary > 0, axis=0)
        proj_h = np.sum(binary > 0, axis=1)
        return proj_v, proj_h

    def smooth(self, proj):
        k = self.cfg.SMOOTH_KERNEL
        kernel = np.ones(k) / k
        return np.convolve(proj, kernel, mode="same")

    def check_is_arrow(self, binary):
        proj = self.smooth(self.compute_projection(binary=binary)[0])
        win_half = int(len(proj) * self.cfg.ARROW_WID_RATIO)
        definite_integrals = []

        for i in range(win_half, len(proj) - win_half, 1):
            definite_integrals.append(np.sum(proj[i - win_half : i + win_half]))
        tops, _ = find_peaks(definite_integrals)
        print(f"tops[0] : {tops[0]}", end=" ")
        if tops[0] > self.cfg.ARROW_AVE_SURFACE:
            print("arrow")
            return 1
        else:
            print("No characters")
            return 0

    """
    Trim
    """

    def trim_upper_sides_blank(self, binary):
        trimed = None
        flipped_bin = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # upper
        blanks_v = self.detect_blank_areas(flipped_bin)
        print(f"blanks_v\n{blanks_v}")
        upper_blanks = [
            (u, b)
            for (u, b) in blanks_v
            if b < binary.shape[0] * self.cfg.UPPER_BLANK_RATIO
        ]
        widest_upper_blank = (
            max(upper_blanks, key=lambda x: x[1] - x[0]) if upper_blanks else (0, 0)
        )
        trimed = binary[widest_upper_blank[1] :, :]

        blanks_h = self.detect_blank_areas(binary)
        print(f"blanks_h\n{blanks_h}\n")
        # left
        lefter_blanks = [
            (l, r)
            for (l, r) in blanks_h
            if r < binary.shape[1] * self.cfg.LEFT_BLANK_RATIO
        ]
        widest_left_blank = (
            max(lefter_blanks, key=lambda x: x[1] - x[0]) if lefter_blanks else (0, 0)
        )
        # right
        righter_blanks = [
            (l, r)
            for (l, r) in blanks_h
            if l > binary.shape[1] * (1 - self.cfg.LEFT_BLANK_RATIO)
        ]
        widest_right_blank = (
            max(righter_blanks, key=lambda x: x[1] - x[0])
            if righter_blanks
            else (binary.shape[1], binary.shape[1])
        )

        print(
            f"trimed outside [{widest_left_blank[1]}:{widest_right_blank[0]}] and above {widest_upper_blank[1]}\n"
        )
        return trimed[:, widest_left_blank[1] : widest_right_blank[0]]

    """
    Blank detection
    """

    def detect_flat_areas(self, proj, noise_height, ignore_wid):
        flats = []
        cur_max, cur_min = -float("inf"), float("inf")
        flat_width = 0
        is_flat = False

        for i, val in enumerate(proj):
            cur_max = val if val > cur_max else cur_max
            cur_min = val if val < cur_min else cur_min

            is_flat = cur_max - cur_min < noise_height
            is_long = flat_width >= ignore_wid

            if is_flat:
                flat_width += 1
            elif not is_flat and is_long:
                flats.append((i - flat_width, i))
                cur_max, cur_min, flat_width = -float("inf"), float("inf"), 0

            elif not is_flat and not is_long:
                cur_max, cur_min, flat_width = -float("inf"), float("inf"), 0

        if is_flat:
            flats.append((len(proj) - flat_width, len(proj)))

        return flats

    def detect_blank_areas(self, binary):
        proj, _ = self.compute_projection(binary=binary)
        proj = self.smooth(proj=proj)

        flats = self.detect_flat_areas(
            proj=proj,
            noise_height=self.cfg.NOISE_HEIGHT,
            ignore_wid=self.cfg.IGNORE_WID,
        )

        print(f"falts\n{flats}")

        lowest_peak = proj[sorted(find_peaks(proj)[0])[0]]
        blank_areas = [
            (l, r) for (l, r) in flats if np.mean(proj[l:r]) < lowest_peak / 2
        ]
        return blank_areas

    """
    Detection
    """

    def detect_char_areas(self, binary):
        proj, _ = self.compute_projection(binary=binary)
        proj = self.smooth(proj=proj)

        peaks, _ = find_peaks(
            proj,
            prominence=np.max(proj) * 0.2,
            distance=int(len(proj) * self.cfg.CHAR_WIDTH_RATIO),
        )
        bottoms, _ = find_peaks(
            -proj,
            prominence=np.max(proj) * 0.2,
            distance=int(len(proj) * self.cfg.CHAR_WIDTH_RATIO),
        )

        if len(peaks) < 2:
            return [(0, len(proj))]

        cuts = [b for b in sorted(bottoms)]
        cuts = [0] + cuts + [len(proj)]

        areas = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

        print(f"raw char areas\n{[(int(l),int(r)) for (l,r) in areas]}")
        return areas

    """
    Post-process
    """

    def merge_small_areas(self, areas):
        print("")
        merged = []
        for l, r in areas[::-1]:
            if (r - l) < self.cfg.MIN_CHAR_WIDTH and merged:
                pl, pr = merged.pop()
                print(f"merged (l, r):{(l, r)} -> (l, pr):{(l, pr)}")
                merged.append((l, pr))
            else:
                merged.append((l, r))

        print(f"merged char areas\n{[(int(l),int(r)) for (l,r) in merged[::-1]]}\n")
        return merged[::-1]

    def filter_characters(self, binary, areas):
        result = []

        for l, r in areas:
            sub = binary[:, l:r]
            density = np.sum(sub > 0) / sub.size

            if density > self.cfg.MIN_CHAR_DENSITY:
                result.append((l, r))

        print(f"removed thin char area\n{[(int(l),int(r)) for (l,r) in result]}")
        return result

    def area_to_binary(self, binary, areas):
        cell = []
        for area in areas:
            cell.append(binary[:, area[0] : area[1]])

        return cell

    def regulate_size(self, photos):
        resized = []
        for photo in photos:
            buf = photo
            lngth, wid = buf.shape
            if lngth > wid:
                diff = np.zeros((lngth, (lngth - wid) // 2), dtype=buf.dtype)
                buf = np.concatenate((diff, buf, diff), axis=1)
            elif lngth < wid:
                diff = np.zeros(((wid - lngth) // 2, wid), dtype=buf.dtype)
                buf = np.concatenate((diff, buf, diff), axis=0)
            resized.append(
                cv2.resize(buf, self.cfg.PHOTO_HW, interpolation=cv2.INTER_NEAREST)
            )

        print(f"resized {len(resized)} characters\n")
        return resized

    """
    Visualization
    """

    def visualize(self, img, proj, chars):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n = len(chars)
        if n == 0:
            return

        cols = min(n, 10)
        rows_chars = (n + cols - 1) // cols

        fig = plt.figure(figsize=(12, 6 + rows_chars * 2))
        gs = gridspec.GridSpec(2 + rows_chars, cols)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(proj)
        ax1.set_title("Projection")

        ax2 = fig.add_subplot(gs[1, :])
        ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Original")
        ax2.axis("off")

        for i, ch in enumerate(chars):
            r = i // cols
            c = i % cols
            ax = fig.add_subplot(gs[2 + r, c])
            ax.imshow(ch, cmap="gray")
            ax.set_title(f"{i}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_projection_shape(self, proj):
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks, peak_widths

        x = np.arange(len(proj))

        peaks, peak_props = find_peaks(
            proj,
            prominence=np.max(proj) * 0.2,
            distance=max(1, len(proj) // 20),
        )

        valleys, _ = find_peaks(
            -proj,
            prominence=np.max(proj) * 0.2,
            distance=max(1, len(proj) // 20),
        )

        widths, width_heights, left_ips, right_ips = peak_widths(
            proj, peaks, rel_height=0.5
        )

        plt.figure(figsize=(12, 5))
        plt.plot(x, proj, color="black", label="projection")
        plt.scatter(peaks, proj[peaks], color="red", label="peaks", zorder=3)
        plt.scatter(valleys, proj[valleys], color="blue", label="valleys", zorder=3)

        for i in range(len(peaks)):
            plt.hlines(
                y=width_heights[i],
                xmin=left_ips[i],
                xmax=right_ips[i],
                color="green",
                linestyle="--",
            )

        for i, p in enumerate(peaks):
            base = peak_props["prominences"][i]
            plt.vlines(
                p,
                proj[p] - base,
                proj[p],
                color="purple",
                linestyle=":",
            )

        plt.title("Projection Shape Analysis")
        plt.xlabel("X")
        plt.ylabel("Projection Value")
        plt.grid()

        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())

        plt.tight_layout()
        plt.show()


class check_menu:
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

        if isinstance(img, Image.Image):
            img = img.convert("L").resize((64, 64))
            img = np.array(img)/255.0
        
        #img = self.add_mosaic(img, 4, 12)
        #img = self.add_noise(img)

        x = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        out = self.model(x)
        pred = out.argmax(1).item()
        return self.idx_to_char[pred]

    def predict_sentence(self, cell):
        chars = [self.predict_char(img) for img in cell]
        line = "".join(chars)
        return self.correct(line)
    
    def add_noise(self, img):
        noise = np.random.normal(0, 0.05, img.shape)  # 平均0, 標準偏差0.05
        img = img + noise
        img = np.clip(img, 0, 1)
        return img
    
    def add_mosaic(self, img, min_size=4, max_size=12):
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
            nn.Linear(256, len(self.chars)),
        )

    def make_chars(self):
        self.chars = ""
        with open('learning/dataset/labels.json', encoding="utf-8") as f:
            self.labels = json.load(f)
            for n in range(len(self.labels)):
                self.chars += self.labels[str(n)]
        return self.chars

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    seg = CharacterSegmenter()
    cnn = check_menu()
    utl = MLutility()

    menus = []
    cell_imgs = utl.take_cell_imgs()

    for i, img in enumerate(cell_imgs):
        print(i, type(img), img is None)

    if not cell_imgs:
        print("画像を取得できませんでした。")
        exit()

    for i, img in enumerate(cell_imgs):
        print(f"{i + 1}番目のセル")
        cell = seg.run(img=img)
        # seg.visualize(img, binary, proj, areas)

        name = cnn.predict_sentence(cell=cell)

        if name == 1 and len(menus) >= 1:
            menus.append(menus[i - 1])
        elif name == 0:
            pass

        menus.append(name)
        print(name, end="\n\n")
    print(menus)

    # utl.send_menu_json_to_saito(menus=menus)
    # making JSON
    # sending to SaitoVPS
