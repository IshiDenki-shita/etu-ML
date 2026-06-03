# play ground for calculating gradient

from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import scipy
from typing import Tuple

from experiments.CharSeg.protetypes.context import Context

"""
nearest true pixel vector
"""


@dataclass
class GradNearestConfig:
    min_theta: np.float16 = np.float16(105.0)


class GradNearest:
    def __init__(self):
        self.config = GradNearestConfig()

    def process(self, context: Context):

        line_removed = context.line_removed

        if line_removed is None:
            raise ValueError("contextのline_removedがNoneです。")

        vectors = self.vector_map_nearest(line_removed)
        valley_point_map = self.judge_valley_point_nearest(
            binary=line_removed, vectors=vectors
        )
        context.candidates = self.raise_candidates(valley_point_map)

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

        vectors = np.stack([vx, vy], axis=0).astype(np.float16)
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

        for y in tqdm(range(height)):
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
        ...
        return candidates_map
