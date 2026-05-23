from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class WatershedCharacterSegmentationConfig:
    # input / output
    input_path: Path = Path("photos/sample/cells/toriten.jpeg")
    output_dir: Path = Path("experiments/outputs")

    # preprocessing
    gaussian_kernel_size: int = 5
    adaptive_block_size: int = 31
    adaptive_c: int = 15

    # morphology
    opening_kernel_size: int = 3
    dilation_iterations: int = 2

    # distance transform
    distance_transform_mask_size: int = 5
    distance_threshold_ratio: float = 0.35

    # filtering
    min_character_width: int = 8
    min_character_height: int = 12
    min_character_area: int = 80

    # debug
    debug_visualization: bool = True


class WatershedCharacterSegmenter:
    def __init__(
        self,
        config: WatershedCharacterSegmentationConfig,
    ) -> None:
        self.config = config

    def run(self) -> None:
        image = self.load_image(
            image_path=self.config.input_path,
        )

        gray = self.convert_to_grayscale(
            image=image,
        )

        binary = self.binarize_image(
            gray_image=gray,
        )

        cleaned = self.remove_noise(
            binary_image=binary,
        )

        markers = self.create_watershed_markers(
            binary_image=cleaned,
        )

        watershed_result = self.apply_watershed(
            original_image=image,
            markers=markers,
        )

        character_regions = self.extract_character_regions(
            watershed_result=watershed_result,
        )

        sorted_regions = self.sort_regions_left_to_right(
            character_regions=character_regions,
        )

        # self.save_characters(
        #     original_image=image,
        #     character_regions=sorted_regions,
        # )

        # if self.config.debug_visualization:
        #     self.visualize_results(
        #         original_image=image,
        #         binary_image=cleaned,
        #         watershed_result=watershed_result,
        #         character_regions=sorted_regions,
        #     )

        self.visualize_watershed_boundaries(
            original_image=image,
            watershed_result=watershed_result,
        )

    def load_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(
            filename=str(image_path),
            flags=cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return image

    def convert_to_grayscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        return cv2.cvtColor(
            src=image,
            code=cv2.COLOR_BGR2GRAY,
        )

    def binarize_image(
        self,
        gray_image: np.ndarray,
    ) -> np.ndarray:
        blurred = cv2.GaussianBlur(
            src=gray_image,
            ksize=(
                self.config.gaussian_kernel_size,
                self.config.gaussian_kernel_size,
            ),
            sigmaX=0,
        )

        binary = cv2.adaptiveThreshold(
            src=blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=self.config.adaptive_block_size,
            C=self.config.adaptive_c,
        )

        return binary

    def remove_noise(
        self,
        binary_image: np.ndarray,
    ) -> np.ndarray:
        kernel = np.ones(
            (
                self.config.opening_kernel_size,
                self.config.opening_kernel_size,
            ),
            dtype=np.uint8,
        )

        opened = cv2.morphologyEx(
            src=binary_image,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
        )

        return opened

    def create_watershed_markers(
        self,
        binary_image: np.ndarray,
    ) -> np.ndarray:
        kernel = np.ones(
            (
                self.config.opening_kernel_size,
                self.config.opening_kernel_size,
            ),
            dtype=np.uint8,
        )

        sure_background = cv2.dilate(
            src=binary_image,
            kernel=kernel,
            iterations=self.config.dilation_iterations,
        )

        distance_map = cv2.distanceTransform(
            src=binary_image,
            distanceType=cv2.DIST_L2,
            maskSize=self.config.distance_transform_mask_size,
        )

        _, sure_foreground = cv2.threshold(
            src=distance_map,
            thresh=(self.config.distance_threshold_ratio * distance_map.max()),
            maxval=255,
            type=0,
        )

        sure_foreground = np.uint8(
            sure_foreground,
        )

        unknown_region = cv2.subtract(
            src1=sure_background,
            src2=sure_foreground,
        )

        _, markers = cv2.connectedComponents(
            image=sure_foreground,
        )

        markers = markers + 1

        markers[unknown_region == 255] = 0

        return markers

    def apply_watershed(
        self,
        original_image: np.ndarray,
        markers: np.ndarray,
    ) -> np.ndarray:
        watershed_markers = cv2.watershed(
            image=original_image.copy(),
            markers=markers,
        )

        return watershed_markers

    def extract_character_regions(
        self,
        watershed_result: np.ndarray,
    ) -> List[np.ndarray]:
        character_regions: List[np.ndarray] = []

        unique_labels = np.unique(
            watershed_result,
        )

        for label in unique_labels:
            if label <= 1:
                continue

            mask = np.zeros_like(
                watershed_result,
                dtype=np.uint8,
            )

            mask[watershed_result == label] = 255

            contours, _ = cv2.findContours(
                image=mask,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(
                    array=contour,
                )

                area = w * h

                if (
                    w < self.config.min_character_width
                    or h < self.config.min_character_height
                    or area < self.config.min_character_area
                ):
                    continue

                character_regions.append(np.array([x, y, w, h]))

        return character_regions

    def sort_regions_left_to_right(
        self,
        character_regions: List[np.ndarray],
    ) -> List[np.ndarray]:
        return sorted(
            character_regions,
            key=lambda region: region[0],
        )

    def save_characters(
        self,
        original_image: np.ndarray,
        character_regions: List[np.ndarray],
    ) -> None:
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, (x, y, w, h) in enumerate(character_regions):
            cropped = original_image[
                y : y + h,
                x : x + w,
            ]

            output_path = self.config.output_dir / f"char_{index:03d}.png"

            cv2.imwrite(
                filename=str(output_path),
                img=cropped,
            )

    def visualize_results(
        self,
        original_image: np.ndarray,
        binary_image: np.ndarray,
        watershed_result: np.ndarray,
        character_regions: List[np.ndarray],
    ) -> None:
        visualization_image = original_image.copy()

        for x, y, w, h in character_regions:
            cv2.rectangle(
                img=visualization_image,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(0, 255, 0),
                thickness=2,
            )

        watershed_boundary = original_image.copy()
        watershed_boundary[watershed_result == -1] = [255, 0, 0]

        plt.figure(
            figsize=(18, 6),
        )

        plt.subplot(1, 3, 1)
        plt.title("Binary")
        plt.imshow(
            binary_image,
            cmap="gray",
        )
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Watershed Boundary")
        plt.imshow(
            cv2.cvtColor(
                watershed_boundary,
                cv2.COLOR_BGR2RGB,
            )
        )
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Character Segmentation")
        plt.imshow(
            cv2.cvtColor(
                visualization_image,
                cv2.COLOR_BGR2RGB,
            )
        )
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_watershed_boundaries(
        self,
        original_image: np.ndarray,
        watershed_result: np.ndarray,
    ) -> None:
        """
        Watershed の領域境界を可視化する。

        - 境界線 : 赤
        - 各領域 : ランダム色
        """

        # watershedラベル一覧
        unique_labels = np.unique(
            watershed_result,
        )

        # 可視化用カラー画像
        colored_regions = np.zeros(
            (
                watershed_result.shape[0],
                watershed_result.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        rng = np.random.default_rng(
            seed=42,
        )

        # 各領域をランダム色で塗る
        for label in unique_labels:
            if label <= 1:
                continue

            random_color = rng.integers(
                low=0,
                high=255,
                size=3,
                dtype=np.uint8,
            )

            colored_regions[watershed_result == label] = random_color

        # 境界線を赤で描画
        boundary_visualization = colored_regions.copy()

        boundary_visualization[watershed_result == -1] = [255, 0, 0]

        # 元画像にも境界を重ねる
        overlay_image = original_image.copy()

        overlay_image[watershed_result == -1] = [0, 0, 255]

        plt.figure(
            figsize=(8, 6),
        )

        plt.subplot(3, 1, 1)
        plt.title("Original Image")
        plt.imshow(
            cv2.cvtColor(
                original_image,
                cv2.COLOR_BGR2RGB,
            )
        )
        plt.axis("off")

        plt.subplot(3, 1, 2)
        plt.title("Watershed Regions")
        plt.imshow(
            colored_regions,
        )
        plt.axis("off")

        plt.subplot(3, 1, 3)
        plt.title("Watershed Boundaries")
        plt.imshow(
            cv2.cvtColor(
                overlay_image,
                cv2.COLOR_BGR2RGB,
            )
        )
        plt.axis("off")

        # 余白を圧縮
        plt.subplots_adjust(
            hspace=0.05,
            top=0.80,
            bottom=0.03,
        )

        plt.show()


def main() -> None:
    config = WatershedCharacterSegmentationConfig()

    segmenter = WatershedCharacterSegmenter(
        config=config,
    )

    segmenter.run()


if __name__ == "__main__":
    main()
