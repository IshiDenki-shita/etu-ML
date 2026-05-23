import numpy as np


def find_long_curve_v(self, binary: np.ndarray):

    h, w = binary.shape
    prev_column = binary[:, 0]

    for i in range(w):
        column = binary[:, i]
        common = np.bitwise_and(column, prev_column)

        in_true = np.zeros(shape=(h,))
        for idx in range(len(column)):
            
            if common[i] == 0:
                continue
            if 