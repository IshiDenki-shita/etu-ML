from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DPCharacterSegmentationConfig:
    # input / output
    input_path: Path = Path("data/sample/cells/toriten.jpeg")
    output_dir: Path = Path("practice/output_chars")
    # resize
    resize_height: int = 128
    # preprocessing
    gaussian_kernel_size: Tuple[int, int] = (5, 5)
    threshold_block_size: int = 31
    threshold_c: int = 15
    # projection smoothing
    projection_smooth_kernel: int = 5
    #
    bound_candidate_ratio: int = 30
    # dynamic programming
    expected_char_width_ratio: float = 0.18
    width_penalty_weight: float = 1.0
    valley_reward_weight: float = 3.0

    # split constraints
    min_char_width: int = 15
    max_char_width: int = 45

    # visualization
    save_debug_image: bool = True


class DPCharacterSegmenter:
    def __init__(
        self,
        config: DPCharacterSegmentationConfig,
    ) -> None:
        self.config = config

    def run(self) -> None:
        image = self.load_image(image_path=self.config.input_path)
        resized = self.resize_image(image=image)
        binary = self.preprocess(image=resized)
        projection = self.compute_vertical_projection(binary=binary)
        split_points = self.find_optimal_splits_dp(
            projection=projection,
            image_width=binary.shape[1],
        )
        characters = self.extract_characters(
            binary=binary,
            split_points=split_points,
        )

        print(f"{len(characters)} detected")

        self.save_characters(characters=characters)
        if self.config.save_debug_image:
            self.save_debug_visualization(
                image=resized,
                split_points=split_points,
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

    def resize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        h, w = image.shape[:2]

        scale = self.config.resize_height / h

        resized = cv2.resize(
            src=image,
            dsize=(
                int(w * scale),
                self.config.resize_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized

    def preprocess(
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

        binary = cv2.adaptiveThreshold(
            src=blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=self.config.threshold_block_size,
            C=self.config.threshold_c,
        )

        return binary

    def compute_vertical_projection(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:
        projection = np.sum(
            a=binary > 0,
            axis=0,
        ).astype(np.float32)

        kernel_size = self.config.projection_smooth_kernel

        kernel = (
            np.ones(
                shape=(kernel_size,),
                dtype=np.float32,
            )
            / kernel_size
        )

        smoothed = np.convolve(
            projection,
            kernel,
            mode="same",
        )

        return smoothed

    def find_optimal_splits_dp(
        self,
        projection: np.ndarray,
        image_width: int,
    ) -> List[int]:
        expected_width = int(
            self.config.resize_height * self.config.expected_char_width_ratio
        )

        candidate_points = self.find_candidate_boundaries(projection=projection)
        print(f"{len(candidate_points)} candidate_points detected")

        candidate_points = [0] + candidate_points + [image_width]
        n = len(candidate_points)
        dp = [float("inf")] * n
        prev = [-1] * n
        dp[0] = 0.0

        for i in range(1, n):
            current_x = candidate_points[i]

            for j in range(i):
                prev_x = candidate_points[j]
                width = current_x - prev_x

                if width < self.config.min_char_width:
                    continue
                if width > self.config.max_char_width:
                    continue

                width_cost = (
                    abs(width - expected_width) * self.config.width_penalty_weight
                )

                valley_penalty = (
                    projection[current_x - 1] * self.config.valley_reward_weight
                )

                total_cost = dp[j] + width_cost + valley_penalty

                if total_cost < dp[i]:
                    dp[i] = total_cost
                    prev[i] = j

        split_points = []
        idx = n - 1

        while idx != -1:
            split_points.append(candidate_points[idx])
            idx = prev[idx]

        split_points.reverse()

        return split_points

    def find_candidate_boundaries(
        self,
        projection: np.ndarray,
    ) -> List[int]:

        boundaries = []
        threshold = np.percentile(a=projection, q=self.config.bound_candidate_ratio)

        for x in range(1, len(projection) - 1):
            center = projection[x]

            if (
                center < projection[x - 1]
                and center < projection[x + 1]
                and center <= threshold
            ):
                boundaries.append(x)

        return boundaries

    def extract_characters(
        self,
        binary: np.ndarray,
        split_points: List[int],
    ) -> List[np.ndarray]:
        characters = []

        for i in range(len(split_points) - 1):
            left = split_points[i]
            right = split_points[i + 1]
            char_img = binary[:, left:right]

            if char_img.size == 0:
                continue

            cropped = self.crop_foreground(binary=char_img)

            characters.append(cropped)

        return characters

    def crop_foreground(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:
        coords = cv2.findNonZero(image=binary)

        if coords is None:
            return binary

        x, y, w, h = cv2.boundingRect(
            array=coords,
        )

        cropped = binary[
            y : y + h,
            x : x + w,
        ]

        return cropped

    def save_characters(
        self,
        characters: List[np.ndarray],
    ) -> None:
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, char_img in enumerate(characters):
            output_path = self.config.output_dir / f"char_{index:03d}.png"

            cv2.imwrite(
                filename=str(output_path),
                img=char_img,
            )

    def save_debug_visualization(
        self,
        image: np.ndarray,
        split_points: List[int],
    ) -> None:
        debug_image = image.copy()

        for x in split_points:
            cv2.line(
                img=debug_image,
                pt1=(x, 0),
                pt2=(x, debug_image.shape[0]),
                color=(0, 0, 255),
                thickness=2,
            )

        debug_path = self.config.output_dir / "debug_segmentation.png"

        cv2.imwrite(
            filename=str(debug_path),
            img=debug_image,
        )


def main() -> None:
    config = DPCharacterSegmentationConfig()
    segmenter = DPCharacterSegmenter(config=config)

    segmenter.run()


if __name__ == "__main__":
    main()
