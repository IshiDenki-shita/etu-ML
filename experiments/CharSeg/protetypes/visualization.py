"""
common visualization of 5 picture along five process
(raw, binarized, line_removed, candidates, selected)
"""

import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

from experiments.CharSeg.protetypes.context import Context


@dataclass
class visualizationConfig:
    window_size: int = 5  # the accurate source code is not completed yet


class Visualizer:
    def __init__(self) -> None:
        self.config = visualizationConfig()

    def process(self, context: Context):
        logging.info("結果を表示します。（共通項目）")
        ...
        # visualization code will be here
