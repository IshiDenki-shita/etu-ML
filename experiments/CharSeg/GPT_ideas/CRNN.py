from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class CRNNCTCConfig:
    INPUT_IMAGE_PATH: Path = Path("data/sample/cells/toriten.jpeg")

    OUTPUT_DIR: Path = Path("practice/output_chars")

    IMAGE_HEIGHT: int = 64
    IMAGE_WIDTH: int = 640

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    BLANK_INDEX: int = 0

    MIN_CHAR_WIDTH: int = 4


class CRNN(nn.Module):

    def __init__(
        self,
        num_classes: int,
    ) -> None:

        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            input_size=256 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(
            in_features=256,
            out_features=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.cnn(x)

        batch_size, channels, height, width = features.size()

        features = features.permute(
            0,
            3,
            1,
            2,
        )

        features = features.contiguous().view(
            batch_size,
            width,
            channels * height,
        )

        rnn_output, _ = self.rnn(features)

        logits = self.fc(rnn_output)

        return logits


class CRNNCTCSegmenter:

    def __init__(
        self,
        config: CRNNCTCConfig,
    ) -> None:

        self.config = config

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    (
                        self.config.IMAGE_HEIGHT,
                        self.config.IMAGE_WIDTH,
                    )
                ),
                transforms.ToTensor(),
            ]
        )

        self.model = CRNN(
            num_classes=100,
        ).to(self.config.DEVICE)

        self.model.eval()

        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:

        image = self.load_image(image_path=self.config.INPUT_IMAGE_PATH)

        resized = cv2.resize(
            src=image,
            dsize=(
                self.config.IMAGE_WIDTH,
                self.config.IMAGE_HEIGHT,
            ),
        )

        logits = self.infer(image=resized)
        char_regions = self.decode_ctc_boundaries(
            logits=logits, image_width=resized.shape[1]
        )
        self.save_characters(image=resized, regions=char_regions)

        return len(char_regions)

    def load_image(
        self,
        image_path: Path,
    ) -> np.ndarray:

        image = cv2.imread(filename=str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        return image

    def infer(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:

        pil_image = Image.fromarray(obj=image)
        tensor = self.transform(pil_image)

        tensor = tensor.unsqueeze(
            dim=0,
        ).to(self.config.DEVICE)

        with torch.no_grad():
            logits = self.model(x=tensor)

        return logits

    def decode_ctc_boundaries(
        self,
        logits: torch.Tensor,
        image_width: int,
    ) -> List[Tuple[int, int, int, int]]:

        probs = torch.softmax(
            logits,
            dim=-1,
        )
        predicted = torch.argmax(
            probs,
            dim=-1,
        )[0]
        predicted = predicted.cpu().numpy()
        sequence_length = len(predicted)

        char_regions = []
        in_character = False
        start_index = 0
        for i, token in enumerate(predicted):
            is_blank = token == self.config.BLANK_INDEX

            if not is_blank and not in_character:
                start_index = i
                in_character = True

            elif is_blank and in_character:
                end_index = i
                x1 = int(start_index / sequence_length * image_width)
                x2 = int(end_index / sequence_length * image_width)

                if x2 - x1 >= self.config.MIN_CHAR_WIDTH:
                    char_regions.append(
                        (
                            x1,
                            0,
                            x2,
                            self.config.IMAGE_HEIGHT,
                        )
                    )
                    in_character = False

        return char_regions

    def save_characters(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int, int]],
    ) -> None:

        for index, (
            x1,
            y1,
            x2,
            y2,
        ) in enumerate(regions):
            char_image = image[y1:y2, x1:x2]
            output_path = self.config.OUTPUT_DIR / f"char_{index:03d}.png"
            cv2.imwrite(
                filename=str(output_path),
                img=char_image,
            )
            print(f"Saved: {output_path}")


def main() -> None:
    config = CRNNCTCConfig()
    segmenter = CRNNCTCSegmenter(config=config)

    num = segmenter.run()

    print(f"{num} character detected")


if __name__ == "__main__":
    main()
