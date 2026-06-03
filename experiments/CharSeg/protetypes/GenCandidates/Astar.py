# generate border line candidate with A* algorythm

import logging
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from experiments.CharSeg.protetypes.context import Context


@dataclass
class AstarConfig:
    DiscargeCost: int = 500  # example of value


class Astar:
    def __init__(self, debug: bool = True):
        self.config = AstarConfig()
        self.debug = debug

    def process(self, context: Context):
        logging.info("A*アルゴリズムを用いて文字の分割境界線の候補を列挙します。")
        line_removed = context.line_removed

        if line_removed is None:
            raise ValueError("context の line_removed が None です")

        context.candidates = self.AstarDischarge(line_removed)

    def AstarDischarge(self, line_removed: np.ndarray):
        candidates_map = np.zeros_like(line_removed)

        # concrete code is here

        return candidates_map
