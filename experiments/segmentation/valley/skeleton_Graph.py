import numpy as np


class SkeletonGraphAnalyzer:
    def bones_of_chars(self, dist_map: np.ndarray) -> np.ndarray:
        bone_points = np.zeros_like(dist_map, dtype=np.uint8)

        h, w = dist_map.shape

        padded = np.pad(
            dist_map,
            pad_width=1,
            mode="constant",
            constant_values=0,
        )

        for y in range(1, h + 1):
            for x in range(1, w + 1):
                neighbor = padded[y - 1 : y + 2, x - 1 : x + 2]

                if self.judge_bone_point_3x3(neighbor):
                    bone_points[y - 1, x - 1] = 1

        return bone_points

    def judge_bone_point_3x3(self, neighbor: np.ndarray) -> bool:
        center = neighbor[1, 1]

        lower_map = neighbor < center

        lower_count = np.sum(lower_map)

        return lower_count > 5
