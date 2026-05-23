import numpy as np
import scipy


def bones_of_chars(self, binary: np.ndarray) -> np.ndarray:
    bone_points = np.zeros_like(binary)

    dist = scipy.ndimage.distance_transform_edt(
        input=binary,
        return_distances=True,
        return_indices=False,
    )
    h, w = dist.shape
    padded = np.pad(array=dist, pad_width=1, mode="constant", constant_values=0)

    for i in range(1, h + 1, 1):
        for j in range(1, w + 1, 1):
            neighor = padded[i - 1 : i + 2, j - 1 : j + 2]

            if judge_bone_point_3x3(neighor=neighor):
                bone_points[i, j] = 1

    return bone_points


def judge_bone_point_3x3(neighor: np.ndarray) -> bool:
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
