import numpy as np
from PIL import Image
import glob
import json
import csv
import os
import cv2
import random

class main:
    def __init__(self):
        self.etl_dirs = [
            "learning/dataset/ETL8B2C1_unpack",
            "learning/dataset/ETL8B2C2_unpack",
            "learning/dataset/ETL8B2C3_unpack"
        ]

        self.out_img = "learning/dataset/etl_images.npy"
        self.out_label = "learning/dataset/etl_labels.npy"
        self.out_dict = "supports/chars_dict.json"

        # 太字化用カーネル（太さ調整）
        self.kernel = np.ones((1, 1), np.uint8)
        self.angle = 20
        self.dx = 10

    # ----------------------------------------------------
    # ① ラベル辞書を作成
    # ----------------------------------------------------
    def make_json(self):
        idx_to_char = {}
        csv_idx = 0

        for d in self.etl_dirs:
            csv_url = os.path.join(d, "meta.csv")

            with open(csv_url, encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                rows = list(reader)

                for i in range(0, len(rows), 160):
                    char = rows[i][1]
                    idx_to_char[csv_idx // 160] = char
                    csv_idx += 160

        with open(self.out_dict, "w", encoding="utf-8") as f:
            json.dump(idx_to_char, f, ensure_ascii=False, indent=1)

        print("✔ chars_dict.json 保存完了")

    # ----------------------------------------------------
    # ② PNG を太くしてから NumPy 保存
    # ----------------------------------------------------
    def preprocessing(self):
        all_files = []
        for d in self.etl_dirs:
            all_files.extend(sorted(glob.glob(f"{d}/*.png")))

        total = len(all_files)
        print(f"総画像数: {total}")

        images = np.zeros((total, 1, 64, 64), dtype=np.uint8)
        labels = np.zeros((total,), dtype=np.int32)

        label = 0

        for i, f in enumerate(all_files):
            # PNG → グレースケール
            img = Image.open(f).convert("L").resize((64, 64))

            #回転
            img = img.rotate(random.randint(-self.angle,self.angle),expand=True,fillcolor=0)
            img = img.resize((64, 64))

            #npに変換
            img = np.array(img, dtype=np.uint8)

            #太さ調整
            img = cv2.dilate(img, self.kernel, iterations=1)

            #平行移動
            afin_matrix = np.float32([[1,0,random.randint(-self.dx,self.dx)],[0,1,0]])
            img = cv2.warpAffine(img,afin_matrix,(64,64))


            images[i, 0] = img
            labels[i] = label

            if (i + 1) % 160 == 0:
                label += 1

            if i % 5000 == 0:
                print(f"{i}/{total} 処理中…")

        np.save(self.out_img, images)
        np.save(self.out_label, labels)

        print("✔ 太字化画像・ラベル保存完了")

    # ----------------------------------------------------
    def run(self):
        self.make_json()
        self.preprocessing()
        print("🎉 前処理すべて完了！")


if __name__ == "__main__":
    main().run()
