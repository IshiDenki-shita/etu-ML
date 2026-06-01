from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SkeletonSegmentationConfig:
    # binarization
    gaussian_kernel_size: Tuple[int, int] = (5, 5)
    adaptive_block_size: int = 31
    adaptive_c: int = 15

    # morphology
    morph_kernel_size: Tuple[int, int] = (3, 3)
    morph_iterations: int = 1

    # skeleton split
    min_branch_distance: int = 8
    min_char_width: int = 8
    projection_smooth_kernel: int = 9

    # contour filtering
    min_contour_area: int = 30

    # output
    padding: int = 4


class SkeletonCharacterSegmenter:
    def __init__(
        self,
        config: SkeletonSegmentationConfig,
    ) -> None:
        self.config = config

    # =========================================================
    # public
    # =========================================================
    def segment_characters(
        self,
        image: np.ndarray,
    ) -> List[np.ndarray]:

        gray = self._to_gray(image=image)
        binary = self._binarize(gray=gray)
        cleaned = self._remove_noise(binary=binary)
        skeleton = self._skeletonize(binary=cleaned)
        branch_points = self._detect_branch_points(skeleton=skeleton)
        split_positions = self._estimate_split_positions(
            binary=cleaned,
            branch_points=branch_points,
        )
        characters = self._crop_characters(
            binary=cleaned,
            split_positions=split_positions,
        )

        return characters

    # =========================================================
    # preprocess
    # =========================================================
    def _to_gray(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        return cv2.cvtColor(
            src=image,
            code=cv2.COLOR_BGR2GRAY,
        )

    def _binarize(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:

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
            blockSize=self.config.adaptive_block_size,
            C=self.config.adaptive_c,
        )

        return binary

    def _remove_noise(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:

        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT,
            ksize=self.config.morph_kernel_size,
        )

        opened = cv2.morphologyEx(
            src=binary,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
            iterations=self.config.morph_iterations,
        )

        contours, _ = cv2.findContours(
            image=opened,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        cleaned = np.zeros_like(opened)

        for contour in contours:
            area = cv2.contourArea(
                contour=contour,
            )

            if area >= self.config.min_contour_area:
                cv2.drawContours(
                    image=cleaned,
                    contours=[contour],
                    contourIdx=-1,
                    color=255,
                    thickness=-1,
                )

        return cleaned

    # =========================================================
    # skeletonization
    # =========================================================
    def _skeletonize(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:

        skeleton = np.zeros_like(binary)

        element = cv2.getStructuringElement(
            shape=cv2.MORPH_CROSS,
            ksize=(3, 3),
        )

        temp_binary = binary.copy()

        while True:
            eroded = cv2.erode(
                src=temp_binary,
                kernel=element,
            )

            opened = cv2.dilate(
                src=eroded,
                kernel=element,
            )

            subtracted = cv2.subtract(
                src1=temp_binary,
                src2=opened,  # slightly small thing temp_binary
            )

            skeleton = cv2.bitwise_or(
                src1=skeleton,
                src2=subtracted,
            )

            temp_binary = eroded.copy()

            if cv2.countNonZero(src=temp_binary) == 0:
                break

        return skeleton

    def _detect_branch_points(
        self,
        skeleton: np.ndarray,
    ) -> List[int]:

        height, width = skeleton.shape

        branch_x_positions: List[int] = []

        padded = cv2.copyMakeBorder(
            src=skeleton,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        for y in range(1, height + 1):
            for x in range(1, width + 1):

                if padded[y, x] == 0:
                    continue

                roi = padded[
                    y - 1 : y + 2,
                    x - 1 : x + 2,
                ]

                neighbor_count = int(np.count_nonzero(roi)) - 1

                # 分岐点:
                # 周囲に3本以上の接続がある
                if neighbor_count >= 3:
                    branch_x_positions.append(x - 1)

        if len(branch_x_positions) == 0:
            return []

        branch_x_positions.sort()
        merged_positions = [branch_x_positions[0]]
        for x in branch_x_positions[1:]:

            if x - merged_positions[-1] >= self.config.min_branch_distance:
                merged_positions.append(x)

        return merged_positions

    def _estimate_split_positions(
        self,
        binary: np.ndarray,
        branch_points: List[int],
    ) -> List[int]:

        height, width = binary.shape

        projection = np.sum(
            binary > 0,
            axis=0,
        ).astype(np.float32)

        smooth_projection = cv2.GaussianBlur(
            src=projection.reshape(1, -1),
            ksize=(
                self.config.projection_smooth_kernel,
                1,
            ),
            sigmaX=0,
        ).flatten()

        split_positions = [0]
        for branch_x in branch_points:

            left = max(
                0,
                branch_x - self.config.min_branch_distance,
            )

            right = min(
                width - 1,
                branch_x + self.config.min_branch_distance,
            )

            local_projection = smooth_projection[left:right]

            if len(local_projection) == 0:
                continue

            min_index = int(np.argmin(local_projection))
            split_x = left + min_index

            if split_x - split_positions[-1] >= self.config.min_char_width:
                split_positions.append(split_x)

        split_positions.append(width)
        split_positions = sorted(list(set(split_positions)))

        return split_positions

    # =========================================================
    # character crop
    # =========================================================
    def _crop_characters(
        self,
        binary: np.ndarray,
        split_positions: List[int],
    ) -> List[np.ndarray]:

        characters: List[np.ndarray] = []

        height, width = binary.shape

        for i in range(len(split_positions) - 1):

            x1 = split_positions[i]
            x2 = split_positions[i + 1]

            if x2 - x1 < self.config.min_char_width:
                continue

            char_img = binary[:, x1:x2]

            coords = cv2.findNonZero(
                src=char_img,
            )

            if coords is None:
                continue

            x, y, w, h = cv2.boundingRect(
                array=coords,
            )

            pad = self.config.padding

            x_start = max(0, x - pad)
            y_start = max(0, y - pad)

            x_end = min(
                char_img.shape[1],
                x + w + pad,
            )

            y_end = min(
                char_img.shape[0],
                y + h + pad,
            )

            cropped = char_img[
                y_start:y_end,
                x_start:x_end,
            ]

            characters.append(cropped)

        return characters


if __name__ == "__main__":
    image_path = Path("data/sample/cells/toriten.jpeg")
    image = cv2.imread(filename=str(image_path))

    if image is None:
        raise ValueError("画像を取得できませんでした")

    config = SkeletonSegmentationConfig()
    segmenter = SkeletonCharacterSegmenter(config=config)
    characters = segmenter.segment_characters(image=image)

    print(f"{len(characters)} character detected")

    output_dir = Path("practice/output_chars")
    output_dir.mkdir(exist_ok=True)

    for index, char_img in enumerate(characters):
        save_path = output_dir / f"char_{index:03d}.png"
        cv2.imwrite(
            filename=str(save_path),
            img=char_img,
        )
