"""
common visualization of 5 picture along five process
(raw, binarized, line_removed, candidates, selected)
"""

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
        ...
        # visualization code will be here
