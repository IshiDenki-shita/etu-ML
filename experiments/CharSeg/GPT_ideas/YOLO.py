from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class CharacterSegmentationConfig:
    # input / output
    input_image_path: Path = Path("data/sample/cells/toriten.jpeg")
    output_dir: Path = Path("practice/output_chars")
    # YOLO model
    yolo_model_path: str = "yolov8n.pt"
    # detection
    detection_confidence: float = 0.25
    # image processing
    binary_threshold: int = 0
    # bounding box margin
    margin: int = 2
    # resize
    resize_width: int = 1280
    # debug
    save_debug_image: bool = True


class CharacterSegmenter:
    def __init__(
        self,
        config: CharacterSegmentationConfig,
    ) -> None:
        self.config = config

        self.model = YOLO(
            model=self.config.yolo_model_path,
        )

        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self) -> None:
        image = self._load_image(image_path=self.config.input_image_path)
        resized = self._resize_image(image=image)
        text_region = self._detect_text_region(image=resized)
        binary = self._preprocess_text_region(text_region=text_region)
        char_boxes = self._segment_characters(binary=binary)

        self._save_characters(
            text_region=text_region,
            char_boxes=char_boxes,
        )

        if self.config.save_debug_image:
            self._save_debug_image(
                text_region=text_region,
                char_boxes=char_boxes,
            )

    def _load_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(
            filename=str(image_path),
        )

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return image

    def _resize_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        scale = self.config.resize_width / width

        resized = cv2.resize(
            src=image,
            dsize=(
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized

    def _detect_text_region(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        YOLOで文字列領域を検出する。

        本来は学習済み文字検出モデルを使用する想定。
        ここでは汎用YOLOを使った構成例を示す。
        """

        results = self.model.predict(
            source=image,
            conf=self.config.detection_confidence,
            verbose=True,
        )

        if len(results[0].boxes) == 0:
            return image

        largest_area = 0
        best_box = None

        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            return image

        x1, y1, x2, y2 = best_box

        cropped = image[
            y1:y2,
            x1:x2,
        ]

        return cropped

    def _preprocess_text_region(
        self,
        text_region: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            src=text_region,
            code=cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            src=gray,
            ksize=(3, 3),
            sigmaX=0,
        )

        binary = cv2.threshold(
            src=blurred,
            thresh=self.config.binary_threshold,
            maxval=255,
            type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT,
            ksize=(3, 3),
        )

        opened = cv2.morphologyEx(
            src=binary,
            op=cv2.MORPH_OPEN,
            kernel=kernel,
        )

        return opened

    def _segment_characters(
        self,
        binary: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(
            image=binary,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        char_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(array=contour)
            area = w * h

            if area < 50:
                continue

            x = max(x - self.config.margin, 0)
            y = max(y - self.config.margin, 0)

            w = min(
                w + self.config.margin * 2,
                binary.shape[1] - x,
            )

            h = min(
                h + self.config.margin * 2,
                binary.shape[0] - y,
            )

            char_boxes.append((x, y, w, h))

        char_boxes.sort(key=lambda box: box[0])

        return char_boxes

    def _save_characters(
        self,
        text_region: np.ndarray,
        char_boxes: List[Tuple[int, int, int, int]],
    ) -> None:
        for index, (x, y, w, h) in enumerate(char_boxes):
            char_image = text_region[
                y : y + h,
                x : x + w,
            ]

            output_path = self.config.output_dir / f"char_{index:03d}.png"

            cv2.imwrite(
                filename=str(output_path),
                img=char_image,
            )

    def _save_debug_image(
        self,
        text_region: np.ndarray,
        char_boxes: List[Tuple[int, int, int, int]],
    ) -> None:
        debug = text_region.copy()

        for index, (x, y, w, h) in enumerate(char_boxes):
            cv2.rectangle(
                img=debug,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(0, 255, 0),
                thickness=2,
            )

            cv2.putText(
                img=debug,
                text=str(index),
                org=(x, y - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                thickness=1,
            )

        debug_path = self.config.output_dir / "debug_boxes.png"

        cv2.imwrite(
            filename=str(debug_path),
            img=debug,
        )


def main() -> None:
    config = CharacterSegmentationConfig()
    segmenter = CharacterSegmenter(config=config)

    segmenter.run()


if __name__ == "__main__":
    main()
