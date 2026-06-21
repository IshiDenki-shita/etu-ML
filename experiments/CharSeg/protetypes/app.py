# cd /Users/matsukou/Desktop/projects/ETU/etu-ML
# python -m experiments.CharSeg.protetypes.app

from dataclasses import dataclass
from pathlib import Path
import logging

from experiments.CharSeg.protetypes.context import Context
from experiments.CharSeg.protetypes.preprocess import Preprocesser
from experiments.CharSeg.protetypes.LineNoise import LineNoiseRemover
from experiments.CharSeg.protetypes.BlankTrim import BlankTrimmer
from experiments.CharSeg.protetypes.GenCandidates.AstarInterval import AstarInterval
from experiments.CharSeg.protetypes.DPselecter import DPselecter
from experiments.CharSeg.protetypes.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def setup_logging(enable_logging: bool, log_level: int) -> None:
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


@dataclass(frozen=True)
class CharacterSegmentationConfig:
    # input / output
    input_image_path: Path = Path("photos/sample/cells/karaage.jpeg")
    output_dir: Path = Path("experiments/CharSeg/outputs")

    # debug
    save_debug_image: bool = True
    enable_logging: bool = True
    log_level: int = logging.INFO


class CharacterSegmenter:
    def __init__(self, config: CharacterSegmentationConfig) -> None:
        self.config = config

        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.context = Context()
        self.preprocesser = Preprocesser()
        self.line_remover = LineNoiseRemover(debug=True)
        self.blank_trimmer = BlankTrimmer(debug=True)
        self.candidate = AstarInterval(debug=True)
        self.dpselecter = DPselecter()
        self.visualizer = Visualizer(debug=True)

        setup_logging(
            enable_logging=config.enable_logging,
            log_level=config.log_level,
        )

    def run(self):
        print("画像分割開始")

        self.context.image_path = self.config.input_image_path
        self.preprocesser.process(self.context)
        self.line_remover.process(self.context)
        self.blank_trimmer.process(self.context)
        self.candidate.process(self.context)
        self.dpselecter.process(self.context)
        self.visualizer.process(self.context)


def main() -> None:
    config = CharacterSegmentationConfig(log_level=logging.DEBUG)
    segmenter = CharacterSegmenter(config)
    segmenter.run()


if __name__ == "__main__":
    main()
