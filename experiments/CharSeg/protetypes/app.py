# python -m experiments.CharSeg.protetypes.border_line.raise_candidates

from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from experiments.CharSeg.protetypes.context import Context
from experiments.CharSeg.protetypes.LineNoise import ContourNoiseRemover
from experiments.CharSeg.protetypes.preprocess import Preprocesser
from experiments.CharSeg.protetypes.GenCandidates.GradNearest import GradNearest
from experiments.CharSeg.protetypes.DPselecter import DPselecter
from experiments.CharSeg.protetypes.visualization import Visualizer


@dataclass(frozen=True)
class CharacterSegmentationConfig:
    # input / output
    input_image_path: Path = Path("photos/sample/cells/ebiten.jpeg")
    output_dir: Path = Path("experiments/CharSeg/outputs")

    # line remove
    line_theta_deg: float = 0.0
    line_theta_tolerance_deg: float = 5.0
    line_min_length: int = 20

    # debug
    save_debug_image: bool = True


class CharacterSegmenter:
    def __init__(self, config: CharacterSegmentationConfig) -> None:
        self.config = config

        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.context = Context()
        self.preprocesser = Preprocesser()
        self.line_remover = ContourNoiseRemover()
        self.candidate = GradNearest()
        self.dpselecter = DPselecter()
        self.visualizer = Visualizer()

    def run(self) -> List[np.ndarray]:
        print("画像分割開始")

        self.context.image_path = self.config.input_image_path
        self.preprocesser.process(self.context)
        self.line_remover.process(self.context)
        self.candidate.process(self.context)
        self.dpselecter.process(self.context)
        self.visualizer.process(self.context)


def main() -> None:
    config = CharacterSegmentationConfig()
    segmenter = CharacterSegmenter(config)
    segmenter.run()


if __name__ == "__main__":
    main()
