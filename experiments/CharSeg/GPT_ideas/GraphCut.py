from dataclasses import dataclass
from pathlib import Path

import cv2
import maxflow
import numpy as np


@dataclass
class GraphCutConfig:
    input_path: str = "data/sample/cells/toriten.jpeg"
    output_dir: str = "practice/output_chars"
    resize_width: int = 1200

    gaussian_kernel: tuple = (5, 5)

    foreground_threshold: int = 170
    background_threshold: int = 80

    morph_kernel_size: int = 3
    morph_iterations: int = 1

    CCL_connectivity: int = 8  # connected component neighbor shape

    min_component_area: int = 40
    min_width: int = 5
    min_height: int = 10


class GraphCutCharacterSegmenter:

    def __init__(self, config: GraphCutConfig):
        self.cfg = config
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self):
        image = self.load_image()
        gray = self.preprocess(image)
        binary = self.graph_cut_segmentation(gray)
        clean = self.postprocess(binary)
        char_regions = self.extract_characters(clean)
        self.save_characters(image, char_regions)
        self.visualize(image, clean, char_regions)

    def load_image(self):
        image = cv2.imread(self.cfg.input_path)
        if image is None:
            raise ValueError("画像の読み込みに失敗しました")

        h, w = image.shape[:2]

        scale = self.cfg.resize_width / w
        resized = cv2.resize(image, (self.cfg.resize_width, int(h * scale)))

        return resized

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.cfg.gaussian_kernel, 0)

        return gray

    def graph_cut_segmentation(self, gray):
        h, w = gray.shape

        graph = maxflow.Graph[float]()
        nodes = graph.add_grid_nodes((h, w))

        fg_th = self.cfg.foreground_threshold
        bg_th = self.cfg.background_threshold

        for y in range(h):
            for x in range(w):
                value = gray[y, x]

                if value > fg_th:
                    graph.add_tedge(nodes[y, x], 10, 0)
                elif value < bg_th:
                    graph.add_tedge(nodes[y, x], 0, 10)
                else:
                    graph.add_tedge(nodes[y, x], 5, 5)

        structure = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        graph.add_grid_edges(nodes, weights=3, structure=structure, symmetric=True)

        graph.maxflow()
        segments = graph.get_grid_segments(nodes)
        binary = np.logical_not(segments).astype(np.uint8) * 255

        return binary

    def postprocess(self, binary):
        kernel = np.ones((self.cfg.morph_kernel_size, self.cfg.morph_kernel_size))

        clean = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.morph_iterations
        )

        return clean

    def extract_characters(self, binary):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=self.cfg.CCL_connectivity
        )

        char_regions = []

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.cfg.min_component_area:
                continue
            if w < self.cfg.min_width:
                continue
            if h < self.cfg.min_height:
                continue

            char_regions.append((x, y, w, h))

        char_regions = sorted(char_regions, key=lambda r: r[0])

        return char_regions

    def save_characters(self, image, char_regions):

        for idx, (x, y, w, h) in enumerate(char_regions):
            char_img = image[y : y + h, x : x + w]
            save_path = Path(self.cfg.output_dir) / f"char_{idx:03d}.png"
            cv2.imwrite(str(save_path), char_img)

    def visualize(self, image, binary, char_regions):
        debug = image.copy()

        for x, y, w, h in char_regions:
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("binary", binary)
        cv2.imshow("characters", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = GraphCutConfig()
    segmenter = GraphCutCharacterSegmenter(config)

    segmenter.run()
