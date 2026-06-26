from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import scipy

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
