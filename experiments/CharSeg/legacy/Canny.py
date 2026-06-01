# 2値の画像には意味あまりない

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class EdgeSegmentationConfig:
    input_path: str = "data/sample/cells/toriten.jpeg"
    output_dir: str = "practice/output_chars"
    # resize
    resize_width: int = 2000
    # blur
    gaussian_kernel: Tuple[int, int] = (5, 5)

    # canny
    canny_threshold1: int = 50
    canny_threshold2: int = 150

    # morphology
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1

    # contour filtering
    min_area: int = 100
    min_width: int = 5
    min_height: int = 10

    # character padding
    padding: int = 4
    # visualization
    show_debug: bool = True


class EdgeCharacterSegmenter:
    def __init__(self, config: EdgeSegmentationConfig):
        self.config = config

    def run(self) -> List[np.ndarray]:
        image = self._load_image()
        resized = self._resize(image)
        gray = self._to_gray(resized)
        blurred = self._blur(gray)
        edges = self._detect_edges(blurred)
        morphed = self._morphology(edges)
        contours = self._find_contours(morphed)
        char_boxes = self._filter_character_boxes(contours)
        char_images = self._extract_characters(gray, char_boxes)

        self._save_characters(char_images)

        if self.config.show_debug:
            self._show_debug(resized, edges, morphed, char_boxes)

        return char_images

    def _load_image(self) -> np.ndarray:
        image = cv2.imread(self.config.input_path)

        if image is None:
            raise ValueError(f"Failed to load image: {self.config.input_path}")

        return image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.config.resize_width / w

        return cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _blur(self, gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, self.config.gaussian_kernel, 0)

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:

        edges = cv2.Canny(
            gray, self.config.canny_threshold1, self.config.canny_threshold2
        )

        return edges

    def _morphology(self, edges: np.ndarray) -> np.ndarray:

        kernel = np.ones(
            (self.config.morphology_kernel_size, self.config.morphology_kernel_size),
            np.uint8,
        )
        # edge gaps connection
        morphed = cv2.dilate(
            edges, kernel, iterations=self.config.morphology_iterations
        )
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

        return morphed

    def _find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def _filter_character_boxes(
        self, contours: List[np.ndarray]
    ) -> List[Tuple[int, int, int, int]]:

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < self.config.min_area:
                continue
            if w < self.config.min_width:
                continue
            if h < self.config.min_height:
                continue

            boxes.append((x, y, w, h))

        # sort left -> right
        boxes = sorted(boxes, key=lambda b: b[0])

        return boxes

    def _extract_characters(
        self, gray: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        characters = []

        for i, (x, y, w, h) in enumerate(boxes):
            pad = self.config.padding

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)

            char_img = gray[y1:y2, x1:x2]
            characters.append(char_img)

        return characters

    def _save_characters(self, characters: List[np.ndarray]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, char_img in enumerate(characters):
            save_path = output_dir / f"char_{i:03d}.png"
            cv2.imwrite(str(save_path), char_img)

    def _show_debug(
        self,
        image: np.ndarray,
        edges: np.ndarray,
        morphed: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
    ) -> None:

        debug_img = image.copy()

        for x, y, w, h in boxes:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("edges", edges)
        cv2.imshow("morphology", morphed)
        cv2.imshow("character_boxes", debug_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = EdgeSegmentationConfig()
    segmenter = EdgeCharacterSegmenter(config)

    characters = segmenter.run()
    print(f"detected characters: {len(characters)}")
