from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class RegionGrowingConfig:
    input_path: str = "data/sample/cells/toriten.jpeg"

    adaptive_block_size: int = 21
    adaptive_c: int = 5
    morphology_kernel_size: int = 3

    min_region_size: int = 80
    use_8_connectivity: bool = True
    char_padding: int = 2

    debug: bool = False


class RegionGrowingSegmenter:
    def __init__(self, config: RegionGrowingConfig):
        self.cfg = config

    def run(
        self,
    ) -> Tuple[list[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray]:
        image = cv2.imread(self.cfg.input_path)

        if image is None:
            raise ValueError(f"failed to load image: {self.cfg.input_path}")

        binary = self.preprocess(image)
        regions = self.region_growing(binary)
        regions = self.filter_regions(regions)
        boxes = self.regions_to_boxes(regions)
        char_images = self.extract_characters(image, boxes)

        return char_images, boxes, binary

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.cfg.adaptive_block_size,
            self.cfg.adaptive_c,
        )

        kernel = np.ones(
            (self.cfg.morphology_kernel_size, self.cfg.morphology_kernel_size), np.uint8
        )

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    def region_growing(self, binary: np.ndarray) -> List[List[Tuple[int, int]]]:
        h, w = binary.shape
        visited = np.zeros((h, w), dtype=bool)
        regions = []

        if self.cfg.use_8_connectivity:
            directions = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        else:
            directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        for y in range(h):
            for x in range(w):

                if binary[y, x] != 255:
                    continue
                if visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True
                pixels = []

                while stack:
                    cy, cx = stack.pop()
                    pixels.append((cy, cx))

                    for dy, dx in directions:
                        ny = cy + dy
                        nx = cx + dx

                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        if visited[ny, nx]:
                            continue
                        if binary[ny, nx] != 255:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))

                regions.append(pixels)
        return regions

    def filter_regions(
        self, regions: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        filtered = []
        for region in regions:

            if len(region) < self.cfg.min_region_size:
                continue
            filtered.append(region)

        return filtered

    def regions_to_boxes(
        self, regions: List[List[Tuple[int, int]]]
    ) -> List[Tuple[int, int, int, int]]:
        boxes = []

        for region in regions:
            ys = [p[0] for p in region]
            xs = [p[1] for p in region]

            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)

            boxes.append((x1, y1, x2, y2))

        return boxes

    def sort_boxes(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        return sorted(boxes, key=lambda b: b[0])

    def extract_characters(
        self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        chars = []
        h, w = image.shape[:2]
        pad = self.cfg.char_padding

        for x1, y1, x2, y2 in boxes:
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)

            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)

            char_img = image[y1 : y2 + 1, x1 : x2 + 1]
            chars.append(char_img)

        return chars

    def draw_boxes(
        self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        result = image.copy()

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result


if __name__ == "__main__":
    config = RegionGrowingConfig()
    segmenter = RegionGrowingSegmenter(config)

    chars, boxes, binary = segmenter.run()

    for i, char in enumerate(chars):
        cv2.imwrite(f"char_{i}.png", char)

    print(f"detected characteres: {len(chars)}")
