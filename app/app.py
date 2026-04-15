"""
purpose: extraction characters's binary img from sentence img
author: Mats

info: I uploaded a doc explaining about how deal with charless area(appear later)
      on our sharing drive
"""

from pathlib import Path
import os
from datetime import datetime, timezone, timedelta
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks


class Config_utl:
    IMG_ROOT = Path("photos/")
    Dictionary = ["唐揚げラーメン", "そぼろあんかけうどん・そば", "餃子ラーメン"]
    url_saito_cafe = "http://162.43.43.163:8080/api/v1/cafe"


class Config_seg:
    MIN_CHAR_WIDTH = 60  # for char detection
    MAX_CHAR_WIDTH = 100
    SMOOTH_KERNEL = 5
    MAX_BLANK_DENSITY = 0.03
    MIN_CHAR_DENSITY = 0.08
    THIN_NOISE_WIDTH = 3  # for partial blank detection
    BLANK_AREA_LEFT = (0, 100)
    PHOTO_HW = (64, 64)  # for regulate sizes of photos


class MLutility:
    def __init__(self):
        self.cfg = Config_utl()

    """
    File operation
    """

    def take_cell_imgs(self):
        imgs_root = self.cfg.IMG_ROOT
        dir_receives = sorted(list(imgs_root.iterdir()))
        dir_caps = sorted(list(dir_receives[-1].iterdir()))

        imgs = []
        for i, dir_cap in enumerate(dir_caps):
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
    def __init__(self, config=Config_seg()):
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

        chars = self.area_to_binary(binary, char_areas)

        chars = self.regulate_size(chars)

        return binary, proj_smooth, char_areas, chars

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

    def area_to_binary(self, binary, areas):
        cell = []
        for area in areas:
            cell.append(binary[area[0] : area[1], :])

        return cell

    def regulate_size(self, photos):
        resized = []
        for photo in photos:
            lngth, wid = photo.shape
            if lngth > wid:
                diff = np.zeros((lngth, (lngth - wid) // 2))
                photo = np.concatenate((diff, photo, diff), axis=1)
            elif lngth < wid:
                diff = np.zeros(((wid - lngth) // 2), wid)
                photo = np.concatenate((diff, photo, diff), axis=0)
            resized.append(cv2.resize(photo, (64, 64)))

        return resized

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

#------------------------------------------------------------
#小原
#------------------------------------------------------------
import torch
import Levenshtein
import csv
from main_CNN import HiraganaCNN  # モデル定義を読み込む

class check_menu:
    def __init__(self):
        # -----------------------------
        # url読み込み
        # -----------------------------
        self.chars_url = "c://python/develop_CNN/chars.txt"
        self.menu_url = "c://python/develop_CNN/menue.csv"
        self.cnn_pth_url = "c://python/develop_CNN/hiragana_cnn.pth"

        # -----------------------------
        # 学習文字読み込み
        # -----------------------------
        with open(self.chars_url, "r", encoding="utf-8") as f:
            data = f.read()
        self.idx_to_char = {i: c for i, c in enumerate(data)}

        # -----------------------------
        # # メニュー読み込み
        # -----------------------------
        with open(self.menu_url, encoding="utf-8") as f:
            reader = csv.reader(f)
            self.menu_list = [row[0] for row in reader if row]

        # -----------------------------
        # モデル読み込み
        # -----------------------------
        self.model = HiraganaCNN()
        self.model.load_state_dict(torch.load(self.cnn_pth_url, map_location="cpu"))
        self.model.eval()

    # -----------------------------
    # 最も近いメニュー
    # -----------------------------
    def correct(self,menu, threshold=3):
        best = min(self.menue_list, key=lambda x: Levenshtein.distance(menu, x))
        dist = Levenshtein.distance(menu, best)
        return best if dist <= threshold else menu

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
    def predict_sentence(self, cell):
        chars = [self.predict_char(img) for img in cell]
        line = "".join(chars)
        return self.correct(line)

    
    # -----------------------------
    # run
    # -----------------------------
    def run(self,cells):
        res = []
        for cell in cells:
            res.append(self.predict_sentence(cell))
        return res
"""
Main function
"""
if __name__ == "__main__":
    seg = CharacterSegmenter()
    # cnn =
    utl = MLutility()

    menus = []
    imgs = utl.take_cell_imgs()
    for img in imgs:
        binary, proj, areas, cell = seg.run(img=img)
        # seg.visualize(img, binary, proj, areas)

        cell_ans = []
        for char in cell:
            name = check_menu.run(char)
            cell_ans.append((name))

        menus.append(cell_ans)

    utl.send_menu_json_to_saito(menus=menus)
    # making JSON
    # sending to SaitoVPS
