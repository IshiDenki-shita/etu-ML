from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SegmentConfig:
    binary_threshhold: int = 128
    open_kernel_size: int = 3
    erosion_kernel_size: int = 1
    erosion_iterations: int = 0

    min_area: int = 50
    min_width: int = 3
    min_height: int = 8

    input_path: str = "data/sample/cells/ebiten.jpeg"
    output_dir: str = "output_chars"

    save_debug_image: bool = True


class ContourCharacterSegmenter:
    def __init__(self, config: SegmentConfig):
        self.cfg = config

    def run(self, image_path: str):
        img = self.load_image(image_path)
        binary = self.preprocess(img)
        contours = self.detect_contours(binary)
        boxes = self.extract_boxes(contours)
        boxes = self.filter_boxes(boxes)
        boxes = self.sort_boxes(boxes)
        chars = self.crop_characters(binary, boxes)
        self.save_characters(chars)

        if self.cfg.save_debug_image:
            self.save_debug_image(img, boxes)

        return chars, boxes

    def load_image(self, image_path: str):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"failed to load image: {image_path}")

        return img

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        open_kernel = np.ones(
            (self.cfg.open_kernel_size, self.cfg.open_kernel_size), np.uint8
        )

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

        if self.cfg.erosion_iterations > 0:
            erosion_kernel = np.ones(
                (self.cfg.erosion_kernel_size, self.cfg.erosion_kernel_size)
            )

            binary = cv2.erode(
                binary,
                erosion_kernel,
                iterations=self.cfg.erosion_iterations,  # for some times
            )

        return binary

    def detect_contours(self, binary):

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,  # only outside edges
            cv2.CHAIN_APPROX_SIMPLE,  # only four vertexies
        )

        return contours

    def extract_boxes(self, contours):
        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

        return boxes

    def filter_boxes(self, boxes):
        filtered = []

        for x, y, w, h in boxes:
            area = w * h

            if area < self.cfg.min_area:
                continue
            if w < self.cfg.min_width:
                continue
            if h < self.cfg.min_height:
                continue

            filtered.append((x, y, w, h))

        return filtered

    def sort_boxes(self, boxes):
        return sorted(boxes, key=lambda b: b[0])

    def crop_characters(self, binary, boxes):
        chars = []

        for x, y, w, h in boxes:
            char_img = binary[y : y + h, x : x + w]
            chars.append(char_img)

        return chars

    def save_characters(self, chars):
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, char_img in enumerate(chars):
            save_path = output_dir / f"char_{i:03}.png"
            cv2.imwrite(str(save_path), char_img)

    def save_debug_image(self, img, boxes):
        debug = img.copy()

        for x, y, w, h in boxes:

            cv2.rectangle(
                debug, (x, y), (x + w, y + h), (0, 255, 0), 2  # how thick line desplay
            )

        save_path = Path(self.cfg.output_dir) / "debug.png"
        cv2.imwrite(str(save_path), debug)


if __name__ == "__main__":
    config = SegmentConfig()

    segmenter = ContourCharacterSegmenter(config=config)
    chars, boxes = segmenter.run(config.input_path)

    print("detected characters:", len(chars))
    for i, box in enumerate(boxes):
        print(f"{i} : {box}")
