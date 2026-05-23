import numpy as np

minimum_gap_length = 10


def horizontal_middle_points(self, binary: np.ndarray) -> np.ndarray:
    middles_map = np.zeros_like(binary)
    h, w = binary.shape
    binary = binary > 0

    for i in range(h):
        start = 0
        in_gap = not binary[i, 0]

        for j in range(w):
            now = binary[i, j]

            if now and in_gap:
                length = j - start
                if length >= minimum_gap_length:
                    middle = start + length // 2
                    middles_map[i, middle] = 1
                in_gap = False

            if not now and not in_gap:
                start = j
                in_gap = True

        if now and in_gap:
            middle = start + (j - start) // 2
            middles_map[i, middle] = 1

    return middles_map
