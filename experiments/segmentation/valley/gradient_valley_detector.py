from typing import Tuple

import numpy as np
import scipy
from tqdm import tqdm

from experiments.segmentation.context.segmentation_context import SegmentationContext
from experiments.segmentation.valley.valley_detector_base import ValleyDetectorBase


class GradientValleyDetector(ValleyDetectorBase):
    def __init__(self, min_theta: float, debug: bool = False) -> None:
        self._min_theta = min_theta
        self.debug = debug

    def process(self, context: SegmentationContext) -> SegmentationContext:
        if context.removed_binary is None:
            raise ValueError("removed_binary is required for valley detection")

        vx, vy = self._grad_map_nearest(context.removed_binary)

        context.valley_points = self._judge_valley_point_nearest(
            binary=context.removed_binary,
            vector=(vx, vy),
            min_theta=self._min_theta,
        )

        return context

    def _grad_map_nearest(
        self,
        binary: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        binary_bool = binary > 0

        indices = scipy.ndimage.distance_transform_edt(
            input=~binary_bool,
            return_distances=False,
            return_indices=True,
        )

        yy, xx = np.indices(binary.shape)

        nearest_y = indices[0]
        nearest_x = indices[1]

        vx = nearest_x - xx
        vy = nearest_y - yy

        vectors = np.stack([vx, vy], axis=0).astype(np.float32)

        norm = np.linalg.norm(vectors, axis=0, keepdims=True)
        vectors /= norm + 1e-6

        return vectors[0], vectors[1]

    def _judge_valley_point_nearest(
        self,
        binary: np.ndarray,
        vector: Tuple[np.ndarray, np.ndarray],
        min_theta: float,
    ) -> np.ndarray:
        threshold_dot = np.cos(min_theta)

        vx, vy = vector

        height, width = binary.shape

        valley_point_map = np.zeros((height, width), dtype=np.uint8)
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
