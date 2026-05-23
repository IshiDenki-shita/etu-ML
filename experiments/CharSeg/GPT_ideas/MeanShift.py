from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


@dataclass
class MeanShiftSegmentationConfig:
    input_path: str = "data/sample/cells/toriten.jpeg"

    resize_height: int = 160

    adaptive_block_size: int = 31
    adaptive_c: int = 15

    morphology_kernel_size: int = 3
    morphology_iterations: int = 1

    mean_shift_quantile: float = 0.2
    mean_shift_n_samples: int = 1000
    mean_shift_bin_seeding: bool = True

    min_component_area: int = 40

    padding: int = 4

    debug: bool = True


class MeanShiftCharacterSegmenter:
    def __init__(self, config: MeanShiftSegmentationConfig) -> None:
        self.config = config

    def run(self) -> List[np.ndarray]:
        image = self._load_image(path=self.config.input_path)
        resized = self._resize_image(image=image)
        gray = self._to_gray(image=resized)
        binary = self._binarize(gray=gray)
        cleaned = self._remove_noise(binary=binary)
        projection = self._vertical_projection(binary=cleaned)
        labels = self._segment_by_mean_shift(projection=projection)
        char_regions = self._extract_regions(labels=labels, binary=cleaned)
        characters = self._crop_characters(binary=cleaned, regions=char_regions)

        if self.config.debug:
            self._debug_visualize(image=resized, binary=cleaned, regions=char_regions)

        return characters

    def _load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(filename=path)

        if image is None:
            raise FileNotFoundError(f"画像を読み込めません: {path}")

        return image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.config.resize_height / h
        resized = cv2.resize(
            src=image, dsize=(int(w * scale), self.config.resize_height)
        )

        return resized

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        binary = cv2.adaptiveThreshold(
            src=gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=self.config.adaptive_block_size,
            C=self.config.adaptive_c,
        )

        return binary

    def _remove_noise(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT,
            ksize=(
                self.config.morphology_kernel_size,
                self.config.morphology_kernel_size,
            ),
        )

        opened = cv2.morphologyEx(
            src=binary,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
            iterations=self.config.morphology_iterations,
        )

        return opened

    def _vertical_projection(self, binary: np.ndarray) -> np.ndarray:
        projection = np.sum(binary > 0, axis=0)

        return projection.astype(np.float32)

    def _segment_by_mean_shift(self, projection: np.ndarray) -> np.ndarray:
        x_positions = np.arange(projection.shape[0])
        features = np.column_stack((x_positions, projection))

        bandwidth = estimate_bandwidth(
            X=features,
            quantile=self.config.mean_shift_quantile,
            n_samples=min(self.config.mean_shift_n_samples, len(features)),
        )

        if bandwidth <= 0:
            bandwidth = 10

        mean_shift = MeanShift(
            bandwidth=bandwidth, bin_seeding=self.config.mean_shift_bin_seeding
        )

        labels = mean_shift.fit_predict(X=features)

        return labels

    def _extract_regions(
        self, labels: np.ndarray, binary: np.ndarray
    ) -> List[Tuple[int, int]]:
        regions = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            indices = np.where(labels == label)[0]

            x_min = int(indices.min())
            x_max = int(indices.max())
            width = x_max - x_min

            if width <= 1:
                continue

            region = binary[:, x_min:x_max]
            area = cv2.countNonZero(src=region)
            if area < self.config.min_component_area:
                continue
            regions.append((x_min, x_max))

        regions.sort(key=lambda r: r[0])
        merged = self._merge_overlapping_regions(regions=regions)

        return merged

    def _merge_overlapping_regions(
        self, regions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:

        if not regions:
            return []

        merged = [regions[0]]
        for current in regions[1:]:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = current

            if curr_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append(current)

        return merged

    def _crop_characters(
        self, binary: np.ndarray, regions: List[Tuple[int, int]]
    ) -> List[np.ndarray]:
        characters = []

        for idx, (x_min, x_max) in enumerate(regions):
            region = binary[:, x_min:x_max]
            coords = cv2.findNonZero(image=region)

            if coords is None:
                continue

            x, y, w, h = cv2.boundingRect(array=coords)
            pad = self.config.padding
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, region.shape[1])
            y2 = min(y + h + pad, region.shape[0])
            char_img = region[y1:y2, x1:x2]
            characters.append(char_img)

            cv2.imwrite(filename=f"character_{idx}.png", img=char_img)

        return characters

    def _debug_visualize(
        self, image: np.ndarray, binary: np.ndarray, regions: List[Tuple[int, int]]
    ) -> None:
        debug_image = image.copy()
        for x_min, x_max in regions:
            cv2.rectangle(
                img=debug_image,
                pt1=(x_min, 0),
                pt2=(x_max, image.shape[0]),
                color=(0, 255, 0),
                thickness=2,
            )
        cv2.imshow("binary", binary)
        cv2.imshow("segmentation", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = MeanShiftSegmentationConfig()
    segmenter = MeanShiftCharacterSegmenter(config=config)

    characters = segmenter.run()
    print(f"抽出文字数: {len(characters)}")
