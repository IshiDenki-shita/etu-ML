from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class TemplateMatchingSegmentationConfig:
    # input / output
    input_path: Path = Path("data/sample/cells/toriten.jpeg")
    output_dir: Path = Path("practice/output_chars")

    # preprocessing
    gaussian_kernel_size: Tuple[int, int] = (3, 3)

    # template matching
    template_width_ratio: float = 0.015
    template_slide_step: int = 1

    # peak detection
    peak_threshold_ratio: float = 0.85
    peak_prominence: float = 0.03

    # segmentation
    min_character_width: int = 10

    # known number of characters
    expected_num_characters: int = 9

    # visualization
    figure_size: Tuple[int, int] = (18, 8)


class TemplateMatchingCharacterSegmenter:
    def __init__(
        self,
        config: TemplateMatchingSegmentationConfig,
    ) -> None:
        self.config = config

    def run(self) -> None:
        image = self.load_image(
            image_path=self.config.input_path,
        )

        binary = self.preprocess_image(
            image=image,
        )

        cropped_binary = self.crop_text_region(
            binary=binary,
        )

        response_map = self.compute_template_matching_response(
            binary=cropped_binary,
        )

        split_positions = self.detect_split_positions(
            response_map=response_map,
            image_width=cropped_binary.shape[1],
        )

        character_regions = self.build_character_regions(
            split_positions=split_positions,
            image_width=cropped_binary.shape[1],
        )

        characters = self.extract_characters(
            binary=cropped_binary,
            character_regions=character_regions,
        )

        # self.save_characters(characters=characters)

        self.visualize_results(
            original_image=image,
            binary=cropped_binary,
            response_map=response_map,
            split_positions=split_positions,
            character_regions=character_regions,
            characters=characters,
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

    def preprocess_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            src=image,
            code=cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            src=gray,
            ksize=self.config.gaussian_kernel_size,
            sigmaX=0,
        )

        _, binary = cv2.threshold(
            src=blurred,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        return binary

    def crop_text_region(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:
        ys, xs = np.where(binary > 0)

        top = np.min(ys)
        bottom = np.max(ys)

        cropped = binary[top : bottom + 1, :]

        return cropped

    def compute_template_matching_response(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:
        height, width = binary.shape

        template_width = max(
            int(width * self.config.template_width_ratio),
            3,
        )

        # 空白テンプレート
        template = np.zeros(
            shape=(height, template_width),
            dtype=np.uint8,
        )

        response_map = np.zeros(
            shape=(width,),
            dtype=np.float32,
        )

        for x in range(
            0,
            width - template_width,
            self.config.template_slide_step,
        ):
            roi = binary[
                :,
                x : x + template_width,
            ]

            score = cv2.matchTemplate(
                image=roi,
                templ=template,
                method=cv2.TM_SQDIFF_NORMED,
            )[0][0]

            # SQDIFFは小さいほど一致なので反転
            response_map[x] = 1.0 - score

        # 平滑化
        response_map = cv2.GaussianBlur(
            src=response_map.reshape(1, -1),
            ksize=(1, 11),
            sigmaX=0,
        ).flatten()

        return response_map

    def detect_split_positions(
        self,
        response_map: np.ndarray,
        image_width: int,
    ) -> List[int]:
        estimated_character_width = image_width / self.config.expected_num_characters

        threshold = np.max(response_map) * self.config.peak_threshold_ratio

        peaks, _ = find_peaks(
            response_map,
            height=threshold,
            distance=int(estimated_character_width * 0.7),
            prominence=self.config.peak_prominence,
        )

        split_positions = [0]

        split_positions.extend(peaks.tolist())

        split_positions.append(image_width)

        split_positions = sorted(list(set(split_positions)))

        return split_positions

    def build_character_regions(
        self,
        split_positions: List[int],
        image_width: int,
    ) -> List[Tuple[int, int]]:
        character_regions: List[Tuple[int, int]] = []

        estimated_character_width = image_width / self.config.expected_num_characters

        for i in range(len(split_positions) - 1):
            start_x = split_positions[i]
            end_x = split_positions[i + 1]

            width = end_x - start_x

            if width < self.config.min_character_width:
                continue

            # 極端に広い領域を除外
            if width > estimated_character_width * 2.5:
                continue

            character_regions.append((start_x, end_x))

        return character_regions

    def extract_characters(
        self,
        binary: np.ndarray,
        character_regions: List[Tuple[int, int]],
    ) -> List[np.ndarray]:
        characters: List[np.ndarray] = []

        for start_x, end_x in character_regions:
            character = binary[
                :,
                start_x:end_x,
            ]

            if character.size == 0:
                continue

            characters.append(character)

        return characters

    def save_characters(
        self,
        characters: List[np.ndarray],
    ) -> None:
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, character in enumerate(characters):
            output_path = self.config.output_dir / f"char_{index:02d}.png"

            cv2.imwrite(
                filename=str(output_path),
                img=character,
            )

    def visualize_results(
        self,
        original_image: np.ndarray,
        binary: np.ndarray,
        response_map: np.ndarray,
        split_positions: List[int],
        character_regions: List[Tuple[int, int]],
        characters: List[np.ndarray],
    ) -> None:
        figure, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=self.config.figure_size,
        )

        # original
        axes[0].imshow(
            cv2.cvtColor(
                original_image,
                cv2.COLOR_BGR2RGB,
            )
        )

        axes[0].set_title("Original Image")

        # binary + split
        axes[1].imshow(
            binary,
            cmap="gray",
        )

        for x in split_positions:
            axes[1].axvline(
                x=x,
                color="red",
                linestyle="--",
                linewidth=1,
            )

        axes[1].set_title("Binary Image + Split Positions")

        # response
        axes[2].plot(response_map)

        axes[2].set_title("Template Matching Response")

        plt.tight_layout()
        plt.show()

        # extracted characters
        if len(characters) > 0:
            figure, axes = plt.subplots(
                nrows=1,
                ncols=len(characters),
                figsize=(
                    2 * len(characters),
                    3,
                ),
            )

            if len(characters) == 1:
                axes = [axes]

            for axis, character in zip(
                axes,
                characters,
            ):
                axis.imshow(
                    character,
                    cmap="gray",
                )

                axis.axis("off")

            plt.tight_layout()
            plt.show()


def main() -> None:
    config = TemplateMatchingSegmentationConfig()

    segmenter = TemplateMatchingCharacterSegmenter(
        config=config,
    )

    segmenter.run()


if __name__ == "__main__":
    main()
