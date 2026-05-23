from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class HoughSegmentationConfig:
    input_path: str = "data/sample/cells/toriten.jpeg"
    output_dir: str = "practice/output_chars"

    # 前処理
    gaussian_kernel_size: int = 5
    adaptive_block_size: int = 31
    adaptive_c: int = 10

    # Morphology
    morphology_kernel_width: int = 3
    morphology_kernel_height: int = 3

    # Hough変換
    hough_threshold: int = 30
    min_line_length: int = 20
    max_line_gap: int = 5

    # 文字領域フィルタ
    min_char_width: int = 5
    min_char_height: int = 20

    # デバッグ表示
    debug: bool = True


class HoughCharacterSegmenter:
    def __init__(self, config: HoughSegmentationConfig):
        self.config = config
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        image = self.load_image()
        gray = self.preprocess(image)
        binary = self.binarize(gray)
        cleaned = self.remove_noise(binary)
        line_image, split_positions = self.detect_split_lines(cleaned)
        char_regions = self.extract_character_regions(cleaned, split_positions)
        self.save_characters(image, char_regions)

        if self.config.debug:
            self.show_debug_images(image, binary, cleaned, line_image)

    def load_image(self) -> np.ndarray:
        image = cv2.imread(self.config.input_path)

        if image is None:
            raise ValueError(
                f"画像を読み込めませんでした: " f"{self.config.input_path}"
            )

        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(
            gray,
            (self.config.gaussian_kernel_size, self.config.gaussian_kernel_size),
            0,
        )

        return gray

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )

        return binary

    def remove_noise(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morphology_kernel_width, self.config.morphology_kernel_height),
        )

        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def detect_split_lines(self, binary: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        height, width = binary.shape

        edges = cv2.Canny(binary, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            maxLineGap=self.config.max_line_gap,
        )

        line_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        split_positions = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dy > dx * 2:
                    # Vertical line → used as a split boundary (red)
                    x_center = (x1 + x2) // 2
                    split_positions.append(x_center)
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    # Non-vertical line → detected but not used (blue)
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        split_positions = sorted(split_positions)
        split_positions = self.merge_close_positions(split_positions, threshold=10)

        return line_image, split_positions

    def merge_close_positions(self, positions: List[int], threshold: int) -> List[int]:

        if len(positions) == 0:
            return []

        merged = [positions[0]]
        for pos in positions[1:]:
            if abs(pos - merged[-1]) > threshold:
                merged.append(pos)

        return merged

    def extract_character_regions(
        self, binary: np.ndarray, split_positions: List[int]
    ) -> List[Tuple[int, int, int, int]]:

        height, width = binary.shape

        boundaries = [0]
        boundaries.extend(split_positions)
        boundaries.append(width)

        char_regions = []
        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]

            if x_end <= x_start:
                continue

            roi = binary[:, x_start:x_end]
            coords = cv2.findNonZero(roi)

            if coords is None:
                continue

            x, y, w, h = cv2.boundingRect(coords)

            if w < self.config.min_char_width or h < self.config.min_char_height:
                continue

            char_regions.append((x_start + x, y, w, h))

        return char_regions

    def save_characters(
        self, original: np.ndarray, char_regions: List[Tuple[int, int, int, int]]
    ) -> None:
        debug_image = original.copy()

        for idx, (x, y, w, h) in enumerate(char_regions):
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        debug_path = Path(self.config.output_dir) / "detected_characters.png"
        cv2.imwrite(str(debug_path), debug_image)

        print(f"抽出文字数: {len(char_regions)}")

    def show_debug_images(
        self,
        original: np.ndarray,
        binary: np.ndarray,
        cleaned: np.ndarray,
        line_image: np.ndarray,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("HoughLinesP Debug", fontsize=14)

        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(binary, cmap="gray")
        axes[0, 1].set_title("binary")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(cleaned, cmap="gray")
        axes[1, 0].set_title("cleaned")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("hough_lines (HoughLinesP)")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    config = HoughSegmentationConfig()

    segmenter = HoughCharacterSegmenter(config)
    segmenter.run()
