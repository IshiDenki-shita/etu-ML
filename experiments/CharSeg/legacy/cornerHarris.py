from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class CornerSegmentationConfig:
    input_path: str = "data/sample/cells/mushidori.jpeg"
    output_dir: str = "practice/output_chars"

    # リサイズ
    resize_width: int = 2000

    # adaptive threshold
    adaptive_block_size: int = 31
    adaptive_c: int = 12

    # morphology
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1

    # corner detection
    use_harris: bool = False

    # Harris
    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04
    harris_threshold_ratio: float = 0.02

    # FAST
    fast_threshold: int = 25
    fast_nonmax_suppression: bool = True

    # connected components
    min_area: int = 80
    min_width: int = 5
    min_height: int = 10

    # 文字統合
    merge_x_distance: int = 15

    # debug
    debug: bool = True


class CornerCharacterSegmenter:
    def __init__(self, config: CornerSegmentationConfig):
        self.config = config

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        image = self.load_image()
        gray = self.preprocess(image)
        binary = self.binarize(gray)
        binary = self.remove_noise(binary)
        corner_map = self.detect_corners(gray)
        enhanced = self.combine_corner_and_binary(binary, corner_map)
        char_regions = self.extract_character_regions(enhanced)
        merged_regions = self.merge_neighbor_regions(char_regions)
        self.save_characters(image, merged_regions)
        self.visualize_result(image, merged_regions)
        print(f"抽出文字数: {len(merged_regions)}")

    def load_image(self):
        image = cv2.imread(self.config.input_path)

        if image is None:
            raise ValueError("画像を読み込めませんでした")

        h, w = image.shape[:2]
        scale = self.config.resize_width / w
        resized = cv2.resize(image, (self.config.resize_width, int(h * scale)))

        return resized

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def binarize(self, gray):
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )

        return binary

    def remove_noise(self, binary):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.morphology_kernel_size, self.config.morphology_kernel_size),
        )

        opening = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=self.config.morphology_iterations
        )

        return opening

    def detect_corners(self, gray):
        if self.config.use_harris:
            return self.detect_harris(gray)

        return self.detect_fast(gray)

    def detect_harris(self, gray):
        gray_float = gray.astype(np.float32)
        gray_umat = cv2.UMat(gray_float)

        harris = cv2.cornerHarris(
            gray_umat,
            self.config.harris_block_size,
            self.config.harris_ksize,
            self.config.harris_k,
        )

        harris = harris.get()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.harris_block_size, self.config.harris_block_size),
        )
        harris = cv2.dilate(harris, kernel, None)
        threshold = self.config.harris_threshold_ratio * harris.max()
        corner_map = np.zeros_like(gray)
        corner_map[harris > threshold] = 255

        return corner_map

    def detect_fast(self, gray):
        fast = cv2.FastFeatureDetector_create(
            threshold=self.config.fast_threshold,
            nonmaxSuppression=self.config.fast_nonmax_suppression,
        )

        keypoints = fast.detect(gray)
        corner_map = np.zeros_like(gray)

        for kp in keypoints:
            x, y = map(int, kp.pt)
            cv2.circle(corner_map, (x, y), 2, 255, -1)

        return corner_map

    def combine_corner_and_binary(self, binary, corner_map):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        expanded_corner = cv2.dilate(corner_map, kernel, iterations=1)
        combined = cv2.bitwise_and(binary, expanded_corner)
        combined = cv2.dilate(combined, kernel, iterations=1)

        return combined

    def extract_character_regions(self, binary) -> List[Tuple[int, int, int, int]]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        regions = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.config.min_area:
                continue
            if w < self.config.min_width:
                continue
            if h < self.config.min_height:
                continue
            regions.append((x, y, w, h))

        regions.sort(key=lambda r: r[0])
        return regions

    def merge_neighbor_regions(self, regions):
        if len(regions) == 0:
            return []

        merged = []
        current = list(regions[0])
        for next_region in regions[1:]:
            x, y, w, h = current
            nx, ny, nw, nh = next_region
            distance = nx - (x + w)

            if distance < self.config.merge_x_distance:
                x1 = min(x, nx)
                y1 = min(y, ny)
                x2 = max(x + w, nx + nw)
                y2 = max(y + h, ny + nh)

                current = [x1, y1, x2 - x1, y2 - y1]

            else:
                merged.append(tuple(current))
                current = list(next_region)

        merged.append(tuple(current))
        return merged

    def save_characters(self, image, regions):
        for idx, (x, y, w, h) in enumerate(regions):
            char_img = image[y : y + h, x : x + w]
            save_path = f"{self.config.output_dir}" f"/char_{idx:03d}.png"
            cv2.imwrite(save_path, char_img)

    def visualize_result(self, image, regions):
        vis = image.copy()

        for idx, (x, y, w, h) in enumerate(regions):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                vis, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

        save_path = f"{self.config.output_dir}" f"/visualization.png"
        cv2.imwrite(save_path, vis)


if __name__ == "__main__":
    config = CornerSegmentationConfig()

    segmenter = CornerCharacterSegmenter(config)
    segmenter.run()
