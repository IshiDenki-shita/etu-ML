from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import scipy
from typing import Tuple

"""
simple grad and 3x3 window
"""

@dataclass
class GradSimpleConfig:
    

class GradSimple:

    def find_valley_line_3x3(self, heat_map: np.ndarray) -> np.ndarray:
        print("start finding valley line")

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
