"""
purpose: extraction characters's binary img from sentence img
author: Mats

info: I uploaded a doc explaining about how deal with charless area(appear later)
      on our sharing drive
"""

IMAGE_PATH = "sample/cells/karaage2.jpeg"

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class Config:
    MIN_CHAR_WIDTH = 60
    MAX_CHAR_WIDTH = 100
    SMOOTH_KERNEL = 5
    MAX_BLANK_DENSITY = 0.03
    MIN_CHAR_DENSITY = 0.08
    # for partial blank detection
    THIN_NOISE_WIDTH = 3
    BLANK_AREA_LEFT = (0, 100)


class CharacterSegmenter:
    def __init__(self, config=Config()):
        self.cfg = config

    """
    Pipeline
    """

    def run(self, img):
        binary = self.preprocess(img)
        proj_v, proj_h = self.compute_projection(binary)

        binary = self.trim_upper_blank(binary, proj_h)
        # binary = self.shave_left_blank(binary, proj_v)
        # binary = self.shave_right_blank(binary, proj_h, proj_v)

        proj_v, _ = self.compute_projection(binary)
        proj_smooth = self.smooth(proj_v)

        flat_areas = self.detect_flat_areas(
            proj=proj_smooth, noise_height=20, ignore_wid=30
        )
        self.visualize_flat_areas(areas=flat_areas, proj=proj_smooth)

        char_areas = self.detect_char_areas(proj_smooth)

        char_areas = self.filter_characters(binary, char_areas)

        char_areas = self.merge_small_areas(char_areas)

        return binary, proj_smooth, char_areas

    """
    Preprocess
    """

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def compute_projection(self, binary):
        proj_v = np.sum(binary > 0, axis=0)
        proj_h = np.sum(binary > 0, axis=1)
        return proj_v, proj_h

    def smooth(self, proj):
        k = self.cfg.SMOOTH_KERNEL
        kernel = np.ones(k) / k
        return np.convolve(proj, kernel, mode="same")

    """
    Trim
    """

    def trim_upper_blank(self, binary, proj_h):
        blanks = self.find_blank_areas(proj_h, binary.shape[1])
        if not blanks:
            return binary

        top_blank = max(blanks, key=lambda x: x[1] - x[0])
        print(f"\ntrimed above {top_blank[1]}")
        return binary[top_blank[1] :, :]

    def shave_left_blank(self, binary, proj_v):
        projl_v = proj_v[: self.cfg.BLANK_AREA_LEFT[1]]
        thresholdl = np.mean(projl_v) - np.std(projl_v) * 0.3  # 0.3 works well
        print(f"threshold {thresholdl}")

        print(f"projl_v:\n{projl_v}")

        count = self.cfg.THIN_NOISE_WIDTH
        for i, val in enumerate(projl_v):
            if val > thresholdl:
                print(f"shaved {i}")
                count -= 1
            if count == 0:
                print(f"\nshaved blank lefter from {i}")
                return binary[:, i:]

        print(f"\nshaved blank lefter from {i}")
        return binary[:, self.cfg.BLANK_AREA_LEFT[1] :]

    def shave_right_blank(self, binary, proj_h, proj_v):
        projr_v = proj_v[::-1]
        projr_h = proj_h

        return binary

    """
    Detection
    """

    def detect_char_areas(self, proj):
        peaks, _ = find_peaks(proj, prominence=20, distance=self.cfg.MIN_CHAR_WIDTH)
        bottoms, _ = find_peaks(-proj, prominence=10, distance=self.cfg.MIN_CHAR_WIDTH)

        if len(peaks) < 2:
            return [(0, len(proj))]

        cuts = [b for b in bottoms]
        cuts = [0] + cuts + [len(proj)]

        areas = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

        print(f"\nlaw char areas\n{[(int(l),int(r)) for (l,r) in areas]}")
        return areas

    def detect_flat_areas(self, proj, noise_height, ignore_wid):
        areas = []
        max, min = -float("inf"), float("inf")
        flat_width = 0

        for i, val in enumerate(proj):
            max = val if val > max else max
            min = val if val < min else min

            is_flat = max - min < noise_height
            is_long = flat_width >= ignore_wid

            if is_flat:
                flat_width += 1
            elif not is_flat and is_long:
                areas.append((i - flat_width, i))
                max, min, flat_width = -float("inf"), float("inf"), 0

            elif not is_flat and not is_long:
                max, min, flat_width = -float("inf"), float("inf"), 0

        if is_flat:
            areas.append((len(proj) - flat_width, len(proj)))

        print(f"\nflat areas\n{areas}")
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

        print(f"merged char areas\n{[(int(l),int(r)) for (l,r) in merged[::-1]]}")
        return merged[::-1]

    def filter_characters(self, binary, areas):
        result = []

        for l, r in areas:
            sub = binary[:, l:r]
            density = np.sum(sub > 0) / sub.size

            if density > self.cfg.MIN_CHAR_DENSITY:
                result.append((l, r))

        print(f"\nremoved thin char area\n{[(int(l),int(r)) for (l,r) in result]}")
        return result

    """
    Blank detection
    """

    def find_blank_areas(self, proj, max_num):
        blank_areas = []

        start = None

        for i, val in enumerate(proj):
            density = val / max_num
            is_blank = density <= self.cfg.MAX_BLANK_DENSITY

            if is_blank and start is None:
                start = i
            elif not is_blank and start is not None:
                blank_areas.append((start, i))
                start = None

        if start is not None:
            blank_areas.append((start, len(proj)))

        return blank_areas

    """
    Visualization
    """

    def visualize(self, img, binary, proj, areas):
        import matplotlib.gridspec as gridspec

        n = len(areas)
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(3, n)

        # projection
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(proj)
        for l, r in areas:
            ax1.axvline(l, color="red")
            ax1.axvline(r, color="green")

        # original
        ax2 = fig.add_subplot(gs[1, :])
        ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax2.axis("off")

        # chars
        for i, (l, r) in enumerate(areas):
            ax = fig.add_subplot(gs[2, i])
            ax.imshow(binary[:, l:r], cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_flat_areas(self, proj, areas):
        x = list(range(len(proj)))

        plt.figure(figsize=(10, 4))
        plt.plot(x, proj, label="projection")

        # 平坦領域を塗る
        for start, end in areas:
            plt.axvspan(start, end, alpha=0.3)

        plt.xlabel("index")
        plt.ylabel("value")
        plt.title("Flat Area Detection")
        plt.legend()
        plt.grid()

        plt.show()


"""
Usage
"""
if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        print("画像を取得できませんでした。")
        exit()

    seg = CharacterSegmenter()
    binary, proj, areas = seg.run(img=img)

    seg.visualize(img, binary, proj, areas)


#------------------------------------------------------------
#小原
#------------------------------------------------------------
import torch
import Levenshtein
import csv
from main_CNN import HiraganaCNN  # モデル定義を読み込む

class check_menue:
    def __init__(self):
        # -----------------------------
        # url読み込み
        # -----------------------------
        self.chars_url = "c://python/develop_CNN/chars.txt"
        self.menue_url = "c://python/develop_CNN/menue.csv"
        self.cnn_pth_url = "c://python/develop_CNN/hiragana_cnn.pth"

        # -----------------------------
        # 学習文字読み込み
        # -----------------------------
        with open(self.chars_url, "r", encoding="utf-8") as f:
            data = f.read()
        self.idx_to_char = {i: c for i, c in enumerate(data)}

        # -----------------------------
        # モデル読み込み
        # -----------------------------
        self.model = HiraganaCNN()
        self.model.load_state_dict(torch.load(self.cnn_pth_url, map_location="cpu"))
        self.model.eval()

    # -----------------------------
    # 最も近いメニュー
    # -----------------------------
    def correct(self,menue, threshold=3):
        with open(self.menue_url,encoding="utf-8") as f:
            reader = csv.reader(f)
            dct = []
            for row in reader:
                if row:
                    dct.append(row[0])
        best = min(dct, key=lambda x: Levenshtein.distance(menue, x))
        dist = Levenshtein.distance(menue, best)
        return best if dist <= threshold else menue

    # -----------------------------
    # 1文字推論
    # -----------------------------
    def predict_char(self,img):
        x = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        out = self.model(x)
        pred = out.argmax(1).item()
        return self.idx_to_char[pred]

    # -----------------------------
    # 文字列推論
    # -----------------------------
    def predict_sentence(self,line):
        return self.correct(line)
