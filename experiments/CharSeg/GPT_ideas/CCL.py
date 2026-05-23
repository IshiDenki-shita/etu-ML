from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class SegmentConfig:
    image_path: str = "data/sample/cells/chikuten.jpeg"
    output_dir: str = "practice/output_chars"

    threshhold_mode: int = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    kernel_size: int = 3
    morphology_iteratins: int = 1
    min_area: int = 50
    padding: int = 5
    debug: bool = True


class CharacterSegmenter:
    def __init__(self, config: SegmentConfig):
        self.config = config
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_image(self):
        image = cv2.imread(self.config.image_path)

        if image is None:
            raise ValueError(f"画像を読み込めません：{self.config.image_path}")
        return image

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, self.config.threshhold_mode)

        kernel = np.ones((self.config.kernel_size, self.config.kernel_size), np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel=kernel,
            iterations=self.config.morphology_iteratins,
        )
        return binary

    def detect_components(self, binary):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if area < self.config.min_area:
                continue

            components.append({"x": x, "y": y, "w": w, "h": h, "area": area})

            components.sort(key=lambda c: c["x"])

            return components

    def save_characters(self, binary, components):
        saved_paths = []
        for idx, comp in enumerate(components):

            x = comp["x"]
            y = comp["y"]
            w = comp["w"]
            h = comp["h"]

            char_img = binary[y : y + h, x : x + w]

            p = self.config.padding

            char_img = cv2.copyMakeBorder(
                char_img, p, p, p, p, cv2.BORDER_CONSTANT, value=0
            )

            save_path = Path(self.config.output_dir) / f"char_{idx:03}.png"

            cv2.imwrite(str(save_path), char_img)

            saved_paths.append(str(save_path))

        return saved_paths

    def draw_components(self, image, components):
        debug_img = image.copy()

        for comp in components:
            x = comp["x"]
            y = comp["y"]
            w = comp["w"]
            h = comp["h"]

            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return debug_img

    def run(self):
        image = self.load_image()
        binary = self.preprocess(image)
        components = self.detect_components(binary=binary)
        saved_paths = self.save_characters(binary, components)

        if self.config.debug:
            debug_img = self.draw_components(image, components)

            debug_path = Path(self.config.output_dir) / "debug_result.png"

            if debug_img is None:
                raise ValueError(f"文字領域を図示できませんでした。")

            cv2.imwrite(str(debug_path), debug_img)

        print(f"detected chars : {len(saved_paths)}")

        return saved_paths


if __name__ == "__main__":
    config = SegmentConfig()

    segmenter = CharacterSegmenter(config=config)
    segmenter.run()
