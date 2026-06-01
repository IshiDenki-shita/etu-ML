from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SlidingWindowSegmentationConfig:
    # input / output
    input_path: Path = Path("data/sample/cells/toriten.jpeg")
    output_dir: Path = Path("practice/output_chars")

    # preprocessing
    gaussian_kernel_size: Tuple[int, int] = (5, 5)
    binary_threshold: int = 0

    # sliding window
    window_width: int = 24
    step_size: int = 4

    # character filtering
    min_character_width: int = 8
    min_character_height: int = 16
    merge_gap_threshold: int = 10

    # debug
    debug_plot: bool = True


class SlidingWindowCharacterSegmenter:
    def __init__(
        self,
        config: SlidingWindowSegmentationConfig,
    ) -> None:
        self.config = config

        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self) -> None:
        image = self.load_image(image_path=self.config.input_path)
        gray = self.convert_to_gray(image=image)
        binary = self.binarize(gray=gray)
        projection = self.calculate_projection(binary=binary)
        candidate_regions = self.sliding_window_search(projection=projection)
        merged_regions = self.merge_regions(regions=candidate_regions)
        character_regions = self.filter_regions(binary=binary, regions=merged_regions)
        characters = self.extract_characters(image=image, regions=character_regions)
        self.save_characters(characters=characters)
        if self.config.debug_plot:
            self.visualize_result(
                image=image,
                binary=binary,
                projection=projection,
                regions=character_regions,
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

    def convert_to_gray(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        return cv2.cvtColor(
            src=image,
            code=cv2.COLOR_BGR2GRAY,
        )

    def binarize(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        blurred = cv2.GaussianBlur(
            src=gray,
            ksize=self.config.gaussian_kernel_size,
            sigmaX=0,
        )

        _, binary = cv2.threshold(
            src=blurred,
            thresh=self.config.binary_threshold,
            maxval=255,
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        return binary

    def calculate_projection(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:
        projection = np.sum(
            a=binary > 0,
            axis=0,
        )

        return projection.astype(np.int32)

    def sliding_window_search(
        self,
        projection: np.ndarray,
    ) -> List[Tuple[int, int]]:

        width = projection.shape[0]
        regions: List[Tuple[int, int]] = []

        inside_character = False
        start_x = 0

        for x in range(
            0,
            width - self.config.window_width,
            self.config.step_size,
        ):
            window = projection[x : x + self.config.window_width]

            score = np.mean(window)

            if score > 2 and not inside_character:
                inside_character = True
                start_x = x

            elif score <= 2 and inside_character:
                inside_character = False
                end_x = x + self.config.window_width

                regions.append((start_x, end_x))

        return regions

    def merge_regions(
        self,
        regions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not regions:
            return []

        merged: List[Tuple[int, int]] = []

        current_start, current_end = regions[0]

        for next_start, next_end in regions[1:]:
            gap = next_start - current_end

            if gap <= self.config.merge_gap_threshold:
                current_end = next_end

            else:
                merged.append((current_start, current_end))

                current_start = next_start
                current_end = next_end

        merged.append((current_start, current_end))

        return merged

    def filter_regions(
        self,
        binary: np.ndarray,
        regions: List[Tuple[int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        valid_regions: List[Tuple[int, int, int, int]] = []

        for start_x, end_x in regions:
            roi = binary[:, start_x:end_x]

            ys, xs = np.where(roi > 0)

            if len(xs) == 0:
                continue

            min_x = start_x + int(np.min(xs))
            max_x = start_x + int(np.max(xs))

            min_y = int(np.min(ys))
            max_y = int(np.max(ys))

            char_width = max_x - min_x
            char_height = max_y - min_y

            if char_width < self.config.min_character_width:
                continue

            if char_height < self.config.min_character_height:
                continue

            valid_regions.append(
                (
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                )
            )

        return valid_regions

    def extract_characters(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        characters: List[np.ndarray] = []

        for (
            min_x,
            min_y,
            max_x,
            max_y,
        ) in regions:
            char_image = image[
                min_y:max_y,
                min_x:max_x,
            ]

            characters.append(char_image)

        return characters

    def save_characters(
        self,
        characters: List[np.ndarray],
    ) -> None:
        for index, char_image in enumerate(characters):
            output_path = self.config.output_dir / f"char_{index:03d}.png"

            cv2.imwrite(
                filename=str(output_path),
                img=char_image,
            )

    def visualize_result(
        self,
        image: np.ndarray,
        binary: np.ndarray,
        projection: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
    ) -> None:
        debug_image = image.copy()

        for (
            min_x,
            min_y,
            max_x,
            max_y,
        ) in regions:
            cv2.rectangle(
                img=debug_image,
                pt1=(min_x, min_y),
                pt2=(max_x, max_y),
                color=(0, 255, 0),
                thickness=2,
            )

        debug_image_rgb = cv2.cvtColor(
            src=debug_image,
            code=cv2.COLOR_BGR2RGB,
        )

        plt.figure(figsize=(18, 10))

        plt.subplot(3, 1, 1)
        plt.title("Binary Image")
        plt.imshow(
            binary,
            cmap="gray",
        )

        plt.subplot(3, 1, 2)
        plt.title("Vertical Projection")
        plt.plot(projection)

        plt.subplot(3, 1, 3)
        plt.title("Sliding Window Segmentation")
        plt.imshow(debug_image_rgb)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    config = SlidingWindowSegmentationConfig()
    segmenter = SlidingWindowCharacterSegmenter(config=config)

    segmenter.run()
