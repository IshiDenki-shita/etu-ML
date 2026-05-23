# play ground for calculating gradient

import math
from tqdm import tqdm
import numpy as np
import scipy
from typing import Tuple

"""
nearest true pixel vector
"""


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

    regulated_x = vectors[0]
    regulated_y = vectors[1]

    return regulated_x, regulated_y


def judge_valley_point_nearest(self, vector: Tuple[np.ndarray, np.ndarray], min_theta):
    print("start finding valley line")
    thres_dotp = -np.cos(min_theta)
    vx, vy = vector

    w, h = vx.shape
    valley_line_map = np.zeros_like(vx)

    for i in tqdm(range(0, h - 1, 1)):
        for j in range(0, w - 1, 1):
            dotp = vx[i, j] * vx[i, j + 1] + vy[i, j] * vy[i, j + 1]
            is_valley_point = dotp > thres_dotp

            if is_valley_point:
                valley_line_map[i, j] = 1

    return valley_line_map


"""
simple grad and 3x3 window
"""


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
    valley_line_map = np.zeros_like(heat_map)

    for i in tqdm(range(ksize, w - ksize, 1)):
        for j in range(ksize, h - ksize, 1):
            neighor = buf_map[
                i - ksize : i + ksize + 1,
                j - ksize : j + ksize + 1,
            ]

            is_valley_point = self.judge_valley_point_3x3(neighor=neighor)

            if is_valley_point:
                valley_line_map[i - ksize, j - ksize] = 1

    return valley_line_map
