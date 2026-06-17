# python -m py_compile experiments/CharSeg/protetypes/border_line/raise_candidates.py

import numpy as np

from experiments.CharSeg.protetypes.LineNoise.DirectContour import (
    ContourNoiseRemover,
    CNRConfig,
)


class LineRemover:
    def __init__(self) -> None:
        cnr_config = CNRConfig()
        self._remover = ContourNoiseRemover(cnr_config)

    def remove_lines(self, img: np.ndarray, visualize: bool = False) -> np.ndarray:
        return self._remover.process(img=img, visualize=visualize)
