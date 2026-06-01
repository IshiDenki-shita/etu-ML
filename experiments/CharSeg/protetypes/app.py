# python -m experiments.CharSeg.protetypes.border_line.raise_candidates

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from experiments.CharSeg.protetypes.context import Context
from experiments.CharSeg.protetypes.LineNoise import ContourNoiseRemover
from experiments.CharSeg.protetypes.preprocess import Preprocesser


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

    def run(self) -> List[np.ndarray]:
        print("画像分割開始")

        self.context.image_path = self.config.input_image_path
        self.preprocesser.process(self.context)
        self.line_remover.process(self.context)

        grad_map = self.grad_map_nearest(removed_binary)

        valley_points_map = self.judge_valley_point_nearest(
            binary=removed_binary,
            vector=grad_map,
            min_theta=np.deg2rad(self.config.min_valley_theta_deg),
        )

        self.visualize_result(
            removed_binary=removed_binary,
            line_map=line_map,
            valley_line_map=valley_points_map,
        )

    def remove_line_noise(
        self,
        image: np.ndarray,
        binary: np.ndarray,
    ) -> np.ndarray:
        contours = self.line_remover.detect_contours(binary)

        cont_vecs, contours = self.line_remover.arrange_contour_vectors2(
            contours=contours,
        )

        direct_lines = self.line_remover.detect_direct_line(
            cont_vecs=cont_vecs,
            contours=contours,
        )

        direct_lines = self.line_remover.pick_needed_line(
            direct_lines=direct_lines,
            target_theta=np.deg2rad(self.config.line_theta_deg),
        )

        connected_lines = self.line_remover.connect_splitted_line(
            straight_lines=direct_lines,
            binary_shape=binary.shape,
        )

        line_map = self.line_remover.draw_staraight_line(
            binary=binary,
            straight_lines=connected_lines,
        )

        removed = self.line_remover.remove_noise_line(
            binary=binary,
            line_map=line_map,
        )

        return removed, line_map

    def make_distance_map(self, binary: np.ndarray) -> np.ndarray:
        dist_map = scipy.ndimage.distance_transform_edt(binary > 0)

        return dist_map

    def visualize_result(
        self,
        removed_binary: np.ndarray,
        line_map: np.ndarray,
        valley_line_map: np.ndarray,
    ) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(9, 6))

        # =========================
        # line removed binary
        # =========================

        axes[0].imshow(removed_binary, cmap="gray")
        axes[0].set_title("Line Removed Binary")
        axes[0].axis("off")

        # =========================
        # detected line map
        # =========================

        axes[1].imshow(line_map, cmap="gray")
        axes[1].set_title("Detected Line Map")
        axes[1].axis("off")

        # =========================
        # valley point scatter
        # =========================

        ys_valley, xs_valley = np.where(valley_line_map > 0)

        axes[2].imshow(removed_binary, cmap="gray")
        axes[2].scatter(xs_valley, ys_valley, s=1)
        axes[2].set_title("Valley Point Scatter")
        axes[2].axis("off")

        plt.subplots_adjust(
            hspace=0.02,
            top=0.98,
            bottom=0.02,
        )

        plt.show()


def main() -> None:
    config = CharacterSegmentationConfig()

    segmenter = CharacterSegmenter(config)

    segmenter.run()


if __name__ == "__main__":
    main()
