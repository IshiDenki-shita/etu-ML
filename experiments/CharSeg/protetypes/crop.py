# crop char images from cell image with plotted border lines

import logging
from dataclasses import dataclass
from typing import List
import numpy as np
from tqdm import tqdm

from experiments.CharSeg.protetypes.context import Context, Borderline


@dataclass
class CropperConfig:
    is_map_border: bool = True  # example of configration


class Cropper:
    def __init__(self, debug: bool = True):
        config = CropperConfig()
        self.debug = debug

    def process(self, context: Context):
        logging.info("文字分割境界線に沿って文字を切り出します")
        borders = context.selected

        if borders is None:
            raise ValueError("contextのselectedがNoneです。")

        context.crops = self.harvest(borders)

    def harvest(self, borders: List[Borderline]) -> List[np.ndarray]:
        char_images = []
        return char_images
