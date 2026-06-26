"""
Preprocess (get, resize, binarize, morphorogy)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

from CharSeg.context import Context


@dataclass
class PreprocesserConfig:
    # image processing
    resize_width: int = 1280
    binary_threshold: int = 0


class Preprocesser:
    def __init__(self) -> None:
        self.config = PreprocesserConfig()

    def process(self, context: Context):
        logging.debug("画像の取得、リサイズ、2値化、ノイズ除去を行います。")
        self.image_path = context.image_path
        buf = self.load_image()
        buf = self.resize_image(image=buf)
        buf = self.binarize(image=buf)
        context.preprocessed = buf

    def load_image(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path))

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

    def binarize(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        binary = cv2.threshold(
            blurred,
            self.config.binary_threshold,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        )

        opened = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

        return opened
