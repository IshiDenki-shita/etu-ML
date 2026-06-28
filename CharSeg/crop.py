# crop char images from cell image with plotted border lines

import logging
from dataclasses import dataclass
from typing import List
import numpy as np

from CharSeg.context import Context, Borderline

logger = logging.getLogger(__name__)


@dataclass
class CropperConfig:
    is_map_border: bool = True  # example of configration


class Cropper:
    def __init__(self, debug: bool = True):
        cfg = CropperConfig()
        self.debug = debug

    def process(self, context: Context):
        logger.debug("文字分割境界線に沿って文字を切り出します")

        borders = context.selected
        binary = context.blank_trimmed

        if borders is None:
            raise ValueError("contextのBordersがNoneです。")
        if binary is None:
            raise ValueError("contextのがNoneです。")

        extracts = self.extract(binary, borders)
        context.crops = self.resize(extracts)

    def extract(
        self, binary: np.ndarray, borders: List[Borderline]
    ) -> List[np.ndarray]:

        # binary から Borderline に沿って文字を切り出す。
        # 取り出した文字画像の長い辺に合わせて文字画像の１辺の長さを決める。

        extracts = []
        return extracts

    def resize(self, extracts: List[np.ndarray]) -> List[np.ndarray]:
        char_images = []

        for img in extracts:
            # 取り出した文字の画像を1辺 64px の画像にリサイズする。
            ...

        return char_images
