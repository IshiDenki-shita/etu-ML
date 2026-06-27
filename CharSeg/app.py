# python -m CharSeg.app

from dataclasses import dataclass
from pathlib import Path
import logging

from CharSeg.context import Context
from CharSeg.preprocess import Preprocesser
from CharSeg.LineNoise import LineNoiseRemover
from CharSeg.BlankTrim import BlankTrimmer
from CharSeg.GenCandidates.AstarInterval import AstarInterval
from CharSeg.DPselecter import DPselecter
from CharSeg.visualization import Visualizer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def setup_logging(enable_logging: bool, log_level: int) -> None:
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return

    logging.getLogger("matplotlib").setLevel(
        logging.WARNING
    )  # matplotlibのログを WARNING 以上に制限する

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


# 画像として扱う拡張子
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tiff"}


@dataclass(frozen=True)
class CharacterSegmentationConfig:
    # input / output
    input_dir: Path = Path("photos/sample/cells")
    input_one_path = Path("photos/sample/cells/gyoza4.jpeg")

    # debug
    enable_logging: bool = True
    log_level: int = logging.DEBUG
    go_all_sample: bool = False

    # show debug image or not
    show_line_remover: bool = True
    show_blank_trimmer: bool = False
    show_GenCandidate: bool = False
    show_visualizer: bool = False


class CharacterSegmenter:
    def __init__(self, config: CharacterSegmentationConfig) -> None:
        self.cfg = config

        self.preprocesser = Preprocesser()
        self.line_remover = LineNoiseRemover(debug=self.cfg.show_line_remover)
        self.blank_trimmer = BlankTrimmer(debug=self.cfg.show_blank_trimmer)
        self.candidate = AstarInterval(debug=self.cfg.show_GenCandidate)
        self.dpselecter = DPselecter()
        self.visualizer = Visualizer(debug=self.cfg.show_visualizer)

        setup_logging(
            enable_logging=config.enable_logging,
            log_level=config.log_level,
        )

    def _iter_image_paths(self):
        """input_dir 内の画像ファイルをソート済みで列挙する"""
        return sorted(
            p
            for p in self.cfg.input_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def run_one(self, image_path: Path) -> None:
        """1枚の画像に対してパイプラインを実行する"""
        logger.debug(f"文字分割開始: {image_path.name}")

        context = Context()
        context.image_path = image_path

        self.preprocesser.process(context)
        self.line_remover.process(context)
        self.blank_trimmer.process(context)
        self.candidate.process(context)
        self.dpselecter.process(context)
        self.visualizer.process(context)

        logger.debug(f"画像分割完了: {image_path.name}")

    def run(self) -> None:
        image_paths = self._iter_image_paths()

        if not image_paths:
            logging.warning(f"画像が見つかりません: {self.cfg.input_dir}")
            return

        logger.debug(f"{len(image_paths)} 件の画像を処理します")

        if self.cfg.go_all_sample:
            """input_dir 内のサンプル画像全てに対して順番に実行する"""
            for image_path in image_paths:
                try:
                    self.run_one(image_path)
                except Exception:
                    logging.exception(
                        f"処理中にエラーが発生しました: {image_path.name}"
                    )
        else:
            self.run_one(self.cfg.input_one_path)


def main() -> None:
    config = CharacterSegmentationConfig(log_level=logging.DEBUG)
    segmenter = CharacterSegmenter(config)
    segmenter.run()


if __name__ == "__main__":
    main()
