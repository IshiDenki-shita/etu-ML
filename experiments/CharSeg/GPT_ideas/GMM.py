# Gaussian Mixture Model

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMCharacterSegmentationConfig:
    input_path: str = "data/sample/cells/toriten.jpeg"
    output_dir: str = "practice/output_chars"

    resize_width: int = 2000

    adaptive_block_size: int = 31
    adaptive_c: int = 10

    morphology_kernel_size: int = 3
    morphology_iterations: int = 1

    gmm_components: int = 2
    gmm_random_state: int = 42

    min_character_width: int = 5
    min_character_height: int = 10

    character_padding: int = 4

    debug_mode: bool = True


class GMMCharacterSegmenter:
    def __init__(self, config: GMMCharacterSegmentationConfig) -> None:
        self.config = config

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        image = self._load_image()
        gray = self._convert_to_gray(image=image)
        resized_gray = self._resize_image(gray=gray)
        binary = self._binarize_image(gray=resized_gray)
        cleaned = self._remove_noise(binary=binary)
        projection = self._calculate_vertical_projection(binary=cleaned)
        split_positions = self._segment_by_gmm(projection=projection)
        character_regions = self._extract_character_regions(
            binary=cleaned, split_positions=split_positions
        )
        self._save_characters(characters=character_regions)

        if self.config.debug_mode:
            self._show_debug_images(binary=cleaned, split_positions=split_positions)

    def _load_image(self) -> np.ndarray:
        image = cv2.imread(filename=self.config.input_path)

        if image is None:
            raise ValueError(f"画像を読み込めません: {self.config.input_path}")

        return image

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

        return gray

    def _resize_image(self, gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape

        scale = self.config.resize_width / width

        resized = cv2.resize(
            src=gray,
            dsize=(self.config.resize_width, int(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized

    def _binarize_image(self, gray: np.ndarray) -> np.ndarray:
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

        cleaned = cv2.morphologyEx(
            src=binary,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
            iterations=self.config.morphology_iterations,
        )

        return cleaned

    def _calculate_vertical_projection(self, binary: np.ndarray) -> np.ndarray:
        projection = np.sum(binary > 0, axis=0)

        return projection

    def _segment_by_gmm(self, projection: np.ndarray) -> List[int]:
        values = projection.reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=self.config.gmm_components,
            random_state=self.config.gmm_random_state,
        )
        gmm.fit(values)
        labels = gmm.predict(values)
        means = gmm.means_.flatten()
        background_label = np.argmin(means)

        split_positions = []
        in_blank = False
        for index, label in enumerate(labels):
            if label == background_label:
                if not in_blank:
                    split_positions.append(index)
                    in_blank = True
            else:
                in_blank = False

        return split_positions

    def _extract_character_regions(
        self, binary: np.ndarray, split_positions: List[int]
    ) -> List[np.ndarray]:
        height, width = binary.shape

        boundaries = [0]

        boundaries.extend(split_positions)

        boundaries.append(width)

        characters = []

        for index in range(len(boundaries) - 1):
            x1 = boundaries[index]
            x2 = boundaries[index + 1]

            if (x2 - x1) < self.config.min_character_width:
                continue

            roi = binary[:, x1:x2]

            points = cv2.findNonZero(image=roi)

            if points is None:
                continue

            x, y, w, h = cv2.boundingRect(array=points)

            if h < self.config.min_character_height:
                continue

            pad = self.config.character_padding

            x_start = max(x - pad, 0)
            y_start = max(y - pad, 0)

            x_end = min(x + w + pad, roi.shape[1])
            y_end = min(y + h + pad, roi.shape[0])

            character = roi[y_start:y_end, x_start:x_end]

            characters.append(character)

        return characters

    def _save_characters(self, characters: List[np.ndarray]) -> None:
        for index, character in enumerate(characters):
            output_path = Path(self.config.output_dir) / f"char_{index:03d}.png"

            cv2.imwrite(filename=str(output_path), img=character)

        print(f"抽出文字数: {len(characters)}")

    def _show_debug_images(
        self, binary: np.ndarray, split_positions: List[int]
    ) -> None:
        debug_image = cv2.cvtColor(src=binary, code=cv2.COLOR_GRAY2BGR)

        for x in split_positions:
            cv2.line(
                img=debug_image,
                pt1=(x, 0),
                pt2=(x, debug_image.shape[0]),
                color=(0, 0, 255),
                thickness=1,
            )

        cv2.imshow(winname="binary", mat=binary)

        cv2.imshow(winname="gmm_segmentation", mat=debug_image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


def main() -> None:
    config = GMMCharacterSegmentationConfig(
        input_path="data/sample/menu.jpg", output_dir="output/gmm_result"
    )

    segmenter = GMMCharacterSegmenter(config=config)

    segmenter.run()


if __name__ == "__main__":
    main()
