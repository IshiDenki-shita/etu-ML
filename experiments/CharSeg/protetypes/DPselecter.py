"""
select suitable border line from candidates
"""

import logging
from dataclasses import dataclass
import numpy as np

from experiments.CharSeg.protetypes.context import Context


@dataclass
class DPselecterConfig:
    idonknow: int = 10  # some config values will be here


class DPselecter:
    def __init__(self) -> None:
        self.config = DPselecterConfig()

    def process(self, context: Context):
        logging.info("分割境界線の候補から、採用する境界線をDPで選択します。")

        candidates_map = context.candidates

        if candidates_map is None:
            raise ValueError("contextのcandidatesがNoneです。")

        context.selected = self.SelectBoarderLine(candidates_map)

    def SelectBoarderLine(self, candidates_map: np.ndarray):
        selected = np.zeros_like(candidates_map)

        # the DP process will be here

        return selected
