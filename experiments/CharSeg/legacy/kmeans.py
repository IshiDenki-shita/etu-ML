from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class KMeansSegmentationConfig:
    # K-means
    k_clusters: int = 2
    kmeans_attempts: int = 10
    kmeans_max_iter: int = 100
    kmeans_epsilon: float = 0.2
    # 前処理
    gaussian_kernel_size: tuple[int, int] = (5, 5)
    # モルフォロジー
    morphology_kernel_size: tuple[int, int] = (3, 3)
    morphology_iterations: int = 1
    # 文字領域フィルタ
    min_component_area: int = 50
    min_width: int = 5
    min_height: int = 10
    # 文字間結合を抑える
    split_projection_threshold_ratio: float = 0.15
    # デバッグ
    debug: bool = True


class KMeansCharacterSegmenter:

    def __init__(self, config: KMeansSegmentationConfig):
        self.config = config

    def segment_characters(self, image: np.ndarray) -> list[np.ndarray]:
        gray = self._to_gray(image)
        blurred = self._blur(gray)
        binary = self._kmeans_binarize(blurred)
        cleaned = self._postprocess(binary)
        separated = self._projection_split(cleaned)
        char_regions = self._extract_character_regions(separated)
        characters = self._crop_characters(gray, char_regions)

        if self.config.debug:
            self._show_debug_images(gray, binary, cleaned, separated, char_regions)

        return characters

    def _to_gray(self, image: np.ndarray) -> np.ndarray:

        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image.copy()

    def _blur(self, gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, self.config.gaussian_kernel_size, 0)

    def _kmeans_binarize(self, gray: np.ndarray) -> np.ndarray:

        pixels = gray.reshape((-1, 1)).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            int(self.config.kmeans_max_iter),
            float(self.config.kmeans_epsilon),
        )

        compactness, labels, centers = cv2.kmeans(
            data=pixels,
            K=int(self.config.k_clusters),
            bestLabels=None,
            criteria=criteria,
            attempts=int(self.config.kmeans_attempts),
            flags=cv2.KMEANS_PP_CENTERS,
        )

        centers = centers.astype(np.uint8)

        clustered = centers[labels.flatten()]
        clustered = clustered.reshape(gray.shape)

        min_value = int(clustered.min())
        max_value = int(clustered.max())

        if np.mean(clustered == min_value) < 0.5:
            binary = np.where(clustered == min_value, 255, 0).astype(np.uint8)
        else:
            binary = np.where(clustered == max_value, 255, 0).astype(np.uint8)

        return binary

    def _postprocess(self, binary: np.ndarray) -> np.ndarray:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.config.morphology_kernel_size
        )

        opened = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=self.config.morphology_iterations
        )

        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.morphology_iterations,
        )

        return closed

    def _projection_split(self, binary: np.ndarray) -> np.ndarary:
        result = binary.copy()
        vertical_projection = np.sum(binary > 0, axis=0)

        threshold = (
            np.max(vertical_projection) * self.config.split_projection_threshold_ratio
        )

        for x in range(binary.shape[1]):

            if vertical_projection[x] < threshold:
                result[:, x] = 0

        return result

    def _extract_character_regions(
        self, binary: np.ndarray
    ) -> list[tuple[int, int, int, int]]:

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

            if area < self.config.min_component_area:
                continue
            if w < self.config.min_width:
                continue
            if h < self.config.min_height:
                continue
            regions.append((x, y, w, h))

        regions.sort(key=lambda r: r[0])

        return regions

    def _crop_characters(
        self, gray: np.ndarray, regions: list[tuple[int, int, int, int]]
    ) -> list[np.ndarray]:
        characters = []

        for x, y, w, h in regions:
            char_img = gray[y : y + h, x : x + w]
            characters.append(char_img)

        return characters

    def _show_debug_images(
        self,
        gray: np.ndarray,
        binary: np.ndarray,
        cleaned: np.ndarray,
        separated: np.ndarray,
        regions: list[tuple[int, int, int, int]],
    ) -> None:

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for x, y, w, h in regions:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("gray", gray)
        cv2.imshow("binary", binary)
        cv2.imshow("cleaned", cleaned)
        cv2.imshow("separated", separated)
        cv2.imshow("result", vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    image_path = "data/sample/cells/toriten.jpeg"
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("画像を読み込めませんでした。")

    config = KMeansSegmentationConfig()
    segmenter = KMeansCharacterSegmenter(config)
    characters = segmenter.segment_characters(image)

    output_dir = Path("practice/output_chars")
    output_dir.mkdir(exist_ok=True)

    for i, char_img in enumerate(characters):
        save_path = output_dir / f"char_{i:03d}.png"
        cv2.imwrite(str(save_path), char_img)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
