from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class GraphSegmentationConfig:
    # input / output
    input_path: Path = Path("data/sample/cells/toriten.jpeg")
    output_dir: Path = Path("practice/output_chars")

    # resize
    resize_height: int = 128

    # binarization
    gaussian_kernel_size: Tuple[int, int] = (5, 5)

    # skeleton
    skeleton_threshold: int = 127

    # graph split
    min_vertical_projection: int = 2
    min_character_width: int = 6
    merge_distance: int = 5

    # connected component
    min_component_area: int = 20


class GraphBasedCharacterSegmenter:
    """
    ストローク分解 + グラフ構造化による文字分割

    流れ:
    1. 前処理
    2. 二値化
    3. 細線化（Skeleton）
    4. ストロークをグラフ構造として扱う
    5. 縦方向投影から分割候補抽出
    6. 文字領域切り出し
    """

    def __init__(
        self,
        config: GraphSegmentationConfig,
    ) -> None:
        self.config = config

    def run(self) -> List[np.ndarray]:
        image = self._load_image()
        resized = self._resize_image(image=image)
        binary = self._binarize(image=resized)
        skeleton = self._skeletonize(binary=binary)
        graph_points = self._extract_graph_points(skeleton=skeleton)
        split_positions = self._detect_split_positions(
            graph_points=graph_points,
            skeleton=skeleton,
        )
        characters = self._extract_characters(
            binary=binary,
            split_positions=split_positions,
        )
        self._save_characters(
            characters=characters,
        )

        return characters

    def _load_image(self) -> np.ndarray:
        image = cv2.imread(
            filename=str(self.config.input_path),
            flags=cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(f"Image not found: {self.config.input_path}")

        return image

    def _resize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        scale = self.config.resize_height / height

        resized_width = int(width * scale)

        resized = cv2.resize(
            src=image,
            dsize=(resized_width, self.config.resize_height),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized

    def _binarize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            src=image,
            code=cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            src=gray,
            ksize=self.config.gaussian_kernel_size,
            sigmaX=0,
        )

        binary = cv2.threshold(
            src=blurred,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        return binary

    def _skeletonize(
        self,
        binary: np.ndarray,
    ) -> np.ndarray:
        """
        モルフォロジー細線化
        """

        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(
            shape=cv2.MORPH_CROSS,
            ksize=(3, 3),
        )
        temp_binary = binary.copy()

        while True:
            eroded = cv2.erode(
                src=temp_binary,
                kernel=element,
            )

            opened = cv2.dilate(
                src=eroded,
                kernel=element,
            )

            temp = cv2.subtract(
                src1=temp_binary,
                src2=opened,
            )

            skeleton = cv2.bitwise_or(
                src1=skeleton,
                src2=temp,
            )

            temp_binary = eroded.copy()

            if cv2.countNonZero(temp_binary) == 0:
                break

        return skeleton

    def _extract_graph_points(
        self,
        skeleton: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        スケルトン上の点をグラフノードとして抽出
        """

        points: List[Tuple[int, int]] = []

        height, width = skeleton.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):

                if skeleton[y, x] == 0:
                    continue

                neighbor_count = self._count_neighbors(
                    skeleton=skeleton,
                    x=x,
                    y=y,
                )

                # 分岐点 or 終端点
                if neighbor_count == 1 or neighbor_count >= 3:
                    points.append((x, y))

        return points

    def _count_neighbors(
        self,
        skeleton: np.ndarray,
        x: int,
        y: int,
    ) -> int:
        neighbor = skeleton[
            y - 1 : y + 2,
            x - 1 : x + 2,
        ]

        count = np.count_nonzero(neighbor)

        # 自分自身を除外
        return int(count - 1)

    def _detect_split_positions(
        self,
        graph_points: List[Tuple[int, int]],
        skeleton: np.ndarray,
    ) -> List[int]:
        """
        ストロークの疎な部分を文字境界候補として抽出
        """

        height, width = skeleton.shape

        projection = np.zeros(
            shape=(width,),
            dtype=np.int32,
        )

        for x in range(width):
            projection[x] = np.count_nonzero(skeleton[:, x])

        split_positions: List[int] = []

        for x in range(1, width - 1):

            current_value = projection[x]

            if current_value > self.config.min_vertical_projection:
                continue

            left = projection[x - 1]
            right = projection[x + 1]

            if left > current_value and right > current_value:
                split_positions.append(x)

        if not split_positions:
            return []

        merged_positions = [split_positions[0]]

        for x in split_positions[1:]:

            distance = x - merged_positions[-1]

            if distance >= self.config.merge_distance:
                merged_positions.append(x)

        return merged_positions

    def _extract_characters(
        self,
        binary: np.ndarray,
        split_positions: List[int],
    ) -> List[np.ndarray]:
        height, width = binary.shape

        boundaries = [0]
        boundaries.extend(split_positions)
        boundaries.append(width)

        characters: List[np.ndarray] = []

        for i in range(len(boundaries) - 1):

            left = boundaries[i]
            right = boundaries[i + 1]

            if (right - left) < self.config.min_character_width:
                continue

            char_img = binary[:, left:right]

            bbox = self._tight_crop(
                binary=char_img,
            )

            if bbox is None:
                continue

            x, y, w, h = bbox

            cropped = char_img[y : y + h, x : x + w]

            area = cv2.countNonZero(cropped)

            if area < self.config.min_component_area:
                continue

            characters.append(cropped)

        return characters

    def _tight_crop(
        self,
        binary: np.ndarray,
    ) -> Tuple[int, int, int, int] | None:
        points = cv2.findNonZero(
            src=binary,
        )

        if points is None:
            return None

        x, y, w, h = cv2.boundingRect(
            array=points,
        )

        return (x, y, w, h)

    def _save_characters(
        self,
        characters: List[np.ndarray],
    ) -> None:
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, char_img in enumerate(characters):

            output_path = self.config.output_dir / f"char_{index:03d}.png"

            cv2.imwrite(
                filename=str(output_path),
                img=char_img,
            )


def main() -> None:
    config = GraphSegmentationConfig()
    segmenter = GraphBasedCharacterSegmenter(
        config=config,
    )
    characters = segmenter.run()
    print(f"Detected characters: {len(characters)}")


if __name__ == "__main__":
    main()
