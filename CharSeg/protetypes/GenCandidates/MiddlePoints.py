from dataclasses import dataclass
import numpy as np

from experiments.CharSeg.protetypes.context import Context


@dataclass
class MiddlePointerConfig:
    minimum_gap_length: int = 10


class MiddlePointer:
    def __init__(self):
        self.config = MiddlePointerConfig()

    def process(self, context: Context):
        middles_map = self.horizontal_middle_points(context.resized)
        context.candidates = self.raise_candidates(middles_map)

    def horizontal_middle_points(self, binary) -> np.ndarray:
        middles_map = np.zeros_like(binary)
        h, w = binary.shape
        binary = binary > 0

        for i in range(h):
            start = 0
            in_gap = not binary[i, 0]

            for j in range(w):
                now = binary[i, j]

                if now and in_gap:
                    length = j - start
                    if length >= self.config.minimum_gap_length:
                        middle = start + length // 2
                        middles_map[i, middle] = 1
                    in_gap = False

                if not now and not in_gap:
                    start = j
                    in_gap = True

            if now and in_gap:
                middle = start + (j - start) // 2
                middles_map[i, middle] = 1

        return middles_map

    def raise_candidates(self, middles_map: np.ndarray):
        raise RuntimeError(
            "MiddlePointer.raise_candidatesはまだ実装されていません"
        )  # delete this column after completion
