from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import scipy
from typing import Tuple

from experiments.CharSeg.protetypes.context import Context
from experiments.CharSeg.protetypes.GenCandidates.ConnectLine import 

"""
simple grad and 3x3 window
"""


@dataclass
class GradSimpleConfig:
    grad_kernel_size = 2  # some config values will be here


class GradSimple:
    def __init__(self):
        self.config = GradSimpleConfig()

    def process(self, context: Context):
        blank_trimmed = context.blank_trimmed

        if blank_trimmed is None:
            raise ValueError("contextのblank_trimmedがNoneです。")

        valley_points_map = self.find_valley_line_3x3(blank_trimmed)
        context.candidates = self.raise_candidates(valley_points_map)

    def find_valley_line_3x3(self, binary: np.ndarray) -> np.ndarray:
        print("start finding valley line")

        heat_map = scipy.ndimage.distance_transform_edt(
            input=~binary,
            return_distances=False,
            return_indices=True,
        )

        buf_map = heat_map.copy()
        ksize = self.config.grad_kernel_size

        buf_map = np.pad(
            array=buf_map,
            pad_width=ksize,
            mode="edge",  # extend char pixel on edge to padding zone
        )

        buf_map = -buf_map
        w, h = buf_map.shape
        valley_points_map = np.zeros_like(heat_map)

        for i in tqdm(range(ksize, w - ksize, 1)):
            for j in range(ksize, h - ksize, 1):
                neighor = buf_map[
                    i - ksize : i + ksize + 1,
                    j - ksize : j + ksize + 1,
                ]

                is_valley_point = self.judge_valley_point_3x3(neighor=neighor)

                if is_valley_point:
                    valley_points_map[i - ksize, j - ksize] = 1

        return valley_points_map

    def judge_valley_point_3x3(self, neighor: np.ndarray) -> bool:
        """
        return if the pixel is the bottom point of valley
        """
        high_and_low = np.zeros(shape=(3, 3))
        try:
            high_and_low[np.where(neighor > neighor[1, 1])] = 1
        except:
            print(neighor)
            exit()

        high_count = np.sum(high_and_low)

        if high_count > 5:
            return True
        else:
            return False

    def neighor_grad_bin_3x3(
        self, neighor: np.ndarray, ksize: int
    ) -> Tuple[np.int8, np.int8]:
        """
        calc gradient vector in binary map with ksize*ksize window
        """
        x, y = 0, 0

        up_and_low = np.array(
            object=[[1] * (2 * ksize + 1)] * ksize
            + [[0] * (2 * ksize + 1)]
            + [[-1] * (2 * ksize + 1)] * ksize,
            dtype="int8",
        )

        y += np.sum(up_and_low * neighor, dtype="int8")
        left_and_right = np.rot90(m=up_and_low)
        x += np.sum(left_and_right * neighor)

        return x, np.int8(y)

    def raise_candidates(self, valley_points_map: np.ndarray):
        candidates_map = np.zeros_like(valley_points_map)
        ...
        return candidates_map
