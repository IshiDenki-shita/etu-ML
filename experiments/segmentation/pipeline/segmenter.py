from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import scipy
from tqdm import tqdm

from experiments.segmentation.context.segmentation_context import SegmentationContext
from experiments.segmentation.pipeline.character_segmentation_pipeline import (
    CharacterSegmentationPipeline,
)


@dataclass(frozen=True)
class CharacterSegmentationConfig:
    # input / output
    input_image_path: Path = Path("photos/sample/cells/ebiten.jpeg")
    output_dir: Path = Path("experiments/CharSeg/outputs")

    # image processing
    resize_width: int = 1280
    binary_threshold: int = 0

    # valley detection
    min_valley_theta_deg: float = 105.0

    # line remove
    line_theta_deg: float = 0.0
    line_theta_tolerance_deg: float = 5.0
    line_min_length: np.float16 = np.float16(20)

    # debug
    save_debug_image: bool = False


class CharacterSegmenter:
    def __init__(self, config: CharacterSegmentationConfig) -> None:
        self.config = config

        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.pipeline = CharacterSegmentationPipeline(self.config)
        self.context: SegmentationContext | None = None

    def run(self) -> List[np.ndarray]:
        print("画像分割開始")

        image = self.load_image()
        resized = self.resize_image(image)

        self.context = SegmentationContext(original_image=resized)
        self.context = self.pipeline.process(self.context)

        removed_binary = self.context.removed_binary
        valley_points_map = self.context.valley_points

        if removed_binary is None or valley_points_map is None:
            raise ValueError("Pipeline did not produce required segmentation outputs")

        # visualization is separated into experiments.segmentation.visualization.debug_visualizer
        # Pipeline and algorithm modules do not call plotting directly.
        line_map = np.zeros_like(removed_binary)

        return [
            self.context.binary,
            removed_binary,
            valley_points_map,
        ]

    """
    legacy. I can use these someday
    """

    def load_image(self) -> np.ndarray:
        image = cv2.imread(str(self.config.input_image_path))

        if image is None:
            raise ValueError("画像を取得できませんでした")

        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]

        scale = self.config.resize_width / width

        resized = cv2.resize(
            src=image,
            dsize=(
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess_image(image, self.config.binary_threshold)

    def make_distance_map(self, binary: np.ndarray) -> np.ndarray:
        dist_map = scipy.ndimage.distance_transform_edt(binary > 0)

        return dist_map

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

    def grad_map_nearest(
        self,
        binary: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if binary is None:
            raise ValueError("最近傍ベクトル計算時にbinaryがNoneです")

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

        vectors = np.stack(
            [vx, vy],
            axis=0,
        ).astype(np.float32)

        norm = np.linalg.norm(
            vectors,
            axis=0,
            keepdims=True,
        )

        vectors /= norm + 1e-6

        return vectors[0], vectors[1]

    def judge_valley_point_nearest(
        self,
        binary: np.ndarray,
        vector: Tuple[np.ndarray, np.ndarray],
        min_theta: float,
    ) -> np.ndarray:
        print("start finding valley line")

        threshold_dot = np.cos(min_theta)

        vx, vy = vector

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

    def visualize_result(
        self,
        removed_binary: np.ndarray,
        line_map: np.ndarray,
        valley_line_map: np.ndarray,
    ) -> None:
        # kept for backwards compatibility but no-op; use debug_visualizer.visualize_context(context)
        return None
