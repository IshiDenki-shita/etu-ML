from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy


@dataclass(frozen=True)
class CharacterSegmentationConfig:
    # input / output
    input_image_path: Path = Path("photos/sample/cells/toriten.jpeg")
    output_dir: Path = Path("experiments/CharSeg/outputs")
    # make gradient map
    grad_kernel_size = 1  # maybe this can be only 1
    # image processing
    binary_threshold: int = 0
    resize_width: int = 1280
    # debug
    save_debug_image: bool = True


class CharacterSegmenter:
    def __init__(self, config: CharacterSegmentationConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self) -> List[np.ndarray]:
        print(f"画像分割開始")
        img = self.load_image()
        resized = self.resize_image(img)
        opened = self.preprocess(resized)

        # noise
        dist_map = self.make_distance_map(opened)
        bones_map = self.bones_of_chars(dist_map=dist_map)

        # bottom of valley
        grad_map = self.grad_map_nearest(opened)
        valley_points_map = self.judge_valley_point_nearest(
            binary=opened, vector=grad_map, min_theta=np.deg2rad(105)
        )

        self.visualize_valley_line(
            binary=opened,
            valley_line_map=valley_points_map,
            bones_map=bones_map,
        )

    def load_image(self) -> np.ndarray:
        """
        get image
        """
        image_path = self.config.input_image_path
        image = cv2.imread(filename=str(image_path))

        if image is None:
            raise ValueError("画像を取得できませんでした")

        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        transform size
        """
        height, width = image.shape[:2]
        scale = self.config.resize_width / width

        resized = cv2.resize(
            src=image,
            dsize=(
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        gray + blur + binarize + open
        """
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)

        binary = cv2.threshold(
            src=blurred,
            thresh=self.config.binary_threshold,
            maxval=255,
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
        opened = cv2.morphologyEx(src=binary, op=cv2.MORPH_OPEN, kernel=kernel)

        return opened

    def make_distance_map(self, binary: np.ndarray) -> np.ndarray:
        dist_map = scipy.ndimage.distance_transform_edt(binary)
        # maybe noise removing is here

        return dist_map

    def bones_of_chars(self, dist_map: np.ndarray) -> np.ndarray:
        bone_points = np.zeros_like(dist_map)

        h, w = dist_map.shape
        padded = np.pad(array=dist_map, pad_width=1, mode="constant", constant_values=0)

        for i in range(1, h + 1, 1):
            for j in range(1, w + 1, 1):
                neighor = padded[i - 1 : i + 2, j - 1 : j + 2]

                if self.judge_bone_point_3x3(neighor=neighor):
                    bone_points[i - 1, j - 1] = 1

        return bone_points

    def judge_bone_point_3x3(self, neighor: np.ndarray) -> bool:
        """
        return if the pixel is locating bone point of char
        """
        high_and_low = np.zeros(shape=(3, 3))
        try:
            high_and_low[np.where(neighor < neighor[1, 1])] = 1
        except:
            print(neighor)
            exit()

        high_count = np.sum(high_and_low)

        if high_count > 5:
            return True
        else:
            return False

    def grad_map_nearest(self, binary: np.ndarray):
        if binary is None:
            raise ValueError("最近傍ベクトル計算時にバイナリがNoneです。")
        binary = binary > 0

        indices = scipy.ndimage.distance_transform_edt(
            input=~binary,
            return_distances=False,
            return_indices=True,
        )

        H, W = binary.shape
        yy, xx = np.indices((H, W))

        if indices is None:
            raise ValueError("最近傍ベクトル計算時に indices がNoneです。")

        nearest_y = indices[0]
        nearest_x = indices[1]

        vx = nearest_x - xx
        vy = nearest_y - yy

        vectors = np.stack([vx, vy], axis=0).astype(np.float16)
        norm = np.linalg.norm(vectors, axis=0, keepdims=True)
        vectors /= norm + np.float16(1e-6)

        regulated_x = vectors[0]
        regulated_y = vectors[1]

        return regulated_x, regulated_y

    def judge_valley_point_nearest(
        self, binary: np.ndarray, vector: Tuple[np.ndarray, np.ndarray], min_theta
    ):
        print("start finding valley line")
        thres_dotp = np.cos(min_theta)
        vx, vy = vector

        w, h = vx.shape
        valley_point_map = np.zeros_like(vx)
        count = 0

        for j in tqdm(range(0, h - 1, 1)):
            for i in range(0, w - 1, 1):

                if binary[i, j] > 0:
                    continue

                dotp = vx[i, j] * vx[i, j + 1] + vy[i, j] * vy[i, j + 1]
                is_valley_point = dotp < thres_dotp

                if is_valley_point:
                    valley_point_map[i, j] = 1
                    count += 1
        print(f"{count} valley points detected")

        return valley_point_map

    def visualize_valley_line(
        self,
        binary: np.ndarray,
        valley_line_map: np.ndarray,
        bones_map: np.ndarray,
    ) -> None:

        fig, axes = plt.subplots(3, 1, figsize=(8, 8))

        # binary
        axes[0].imshow(binary, cmap="gray")
        axes[0].set_title("Binary Image")
        axes[0].axis("off")
        # bones map
        ys2, xs2 = np.where(bones_map > 0)
        axes[1].imshow(binary, cmap="gray")
        axes[1].scatter(xs2, ys2, s=1)
        axes[1].set_title("Bones map")
        axes[1].axis("off")
        # overlay valley line
        ys1, xs1 = np.where(valley_line_map > 0)
        axes[2].imshow(binary, cmap="jet")
        axes[2].scatter(xs1, ys1, s=1)
        axes[2].set_title("Valley Line Map")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


def main() -> None:
    config = CharacterSegmentationConfig()
    segmenter = CharacterSegmenter(config=config)

    segmenter.run()


if __name__ == "__main__":
    main()
