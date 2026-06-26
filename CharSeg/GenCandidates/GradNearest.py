# play ground for calculating gradient

import logging
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import scipy
from typing import Tuple
import matplotlib.pyplot as plt

from CharSeg.protetypes.context import Context

"""
nearest true pixel vector
"""


@dataclass
class GradNearestConfig:
    min_theta: np.float16 = np.deg2rad(105.0, dtype=np.float16)


class GradNearest:
    def __init__(self, debug: bool = True):
        self.config = GradNearestConfig()
        self.debug = debug

    def process(self, context: Context):
        ("最近点までの方向ベクトルを用いて分割境界線候補を列挙します。")

        blank_trimmed = context.blank_trimmed

        if blank_trimmed is None:
            raise ValueError("contextのblank_trimmedがNoneです。")

        vectors = self.vector_map_nearest(blank_trimmed)
        valley_point_map = self.judge_valley_point_nearest(
            binary=blank_trimmed, vectors=vectors
        )
        candidates_map = self.raise_candidates(valley_point_map)
        context.candidates = candidates_map

        self.visualize(
            line_removed=line_removed,
            valley_point_map=valley_point_map,
            candidates_map=candidates_map,
        )

    def visualize(
        self,
        line_removed: np.ndarray,
        valley_point_map: np.ndarray,
        candidates_map: np.ndarray,
    ) -> None:
        if not self.debug:
            return

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(8, 8),
            constrained_layout=False,
        )

        axes[0].imshow(line_removed, cmap="gray")
        axes[0].set_title("Line Removed")
        axes[0].axis("off")

        axes[1].imshow(line_removed, cmap="gray")

        ys, xs = np.where(valley_point_map > 0)
        axes[1].scatter(
            xs,
            ys,
            c="red",
            s=4,
            marker="o",
        )
        axes[1].set_title("Valley Points")
        axes[1].axis("off")

        axes[2].imshow(line_removed, cmap="gray")
        axes[2].imshow(
            np.ma.masked_where(candidates_map == 0, candidates_map),
            cmap="Blues",
            alpha=0.6,
        )
        axes[2].set_title("Candidates")
        axes[2].axis("off")

        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.98,
            bottom=0.02,
            hspace=0.03,
        )

        plt.show()

    def vector_map_nearest(self, binary: np.ndarray):
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

        vectors = np.stack([vx, vy], axis=0).astype(np.float64)
        norm = np.linalg.norm(vectors, axis=0, keepdims=True)
        vectors /= norm + np.float16(1e-6)

        return vectors

    def judge_valley_point_nearest(
        self,
        binary: np.ndarray,
        vectors: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        print("start finding valley line")

        threshold_dot = np.cos(self.config.min_theta)
        vx, vy = vectors

        height, width = binary.shape

        valley_point_map = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        count = 0

        for y in tqdm(range(height), disable=self.debug):
            for x in range(width - 1):

                if binary[y, x] > 0:
                    continue

                dot = vx[y, x] * vx[y, x + 1] + vy[y, x] * vy[y, x + 1]

                if dot < threshold_dot:
                    valley_point_map[y, x] = 1
                    count += 1

        print(f"{count} valley points detected")

        return valley_point_map

    def raise_candidates(self, valley_points_map: np.ndarray):
        candidates_map = np.zeros_like(valley_points_map)

        # raising borderline candidates is here
        # note: the candidates_map has to be made as the mask image of border lines
        ...
        return candidates_map
