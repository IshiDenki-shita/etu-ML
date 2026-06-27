import logging
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import cv2

from CharSeg.context import Context

logger = logging.getLogger(__name__)


@dataclass
class BTconfig:
    noise_size_thresh: int = 100  # this value will be changed soon
    blank_horizontal_thresh: int = 10
    blank_vertical_thresh: int = 10


import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BlankTrimmer:
    cfg = BTconfig()

    def __init__(self, debug: bool):
        self.debug = debug

    def process(self, context: Context):
        ("小さい破片ノイズを取り除いて、空白部分を切り取ります。")

        line_removed = context.line_removed

        if line_removed is None:
            raise ValueError(f"line_removed を読み込めませんでした")

        h, w = line_removed.shape[:2]  # ここでは img は ndarray と確定する
        scale = min(h, w) / 1000.0

        self.cfg.blank_horizontal_thresh = max(10, int(10 * scale))
        self.cfg.blank_vertical_thresh = max(10, int(10 * scale))
        self.cfg.noise_size_thresh = int(h * w * 5e-4)

        if line_removed is None:
            raise ValueError("contextのline_removedがNoneです。")

        small_fragment_removed = self.remove_small_noise(
            binary=line_removed, size_thresh=self.cfg.noise_size_thresh
        )

        blank_trimmed = self.trim_blank_area(binary=small_fragment_removed)

        context.blank_trimmed = blank_trimmed

        if self.debug:
            self.visualize_blank_trim(
                line_removed,
                context.blank_trimmed,
                self.cfg.blank_horizontal_thresh,
                self.cfg.blank_vertical_thresh,
            )

    def remove_small_noise(self, binary: np.ndarray, size_thresh: int):
        # findContours用にuint8へ変換
        contour_img = (binary.astype(np.uint8)) * 255

        contours, _ = cv2.findContours(
            contour_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        small_fragment_removed = binary.copy()

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < size_thresh:
                cv2.drawContours(
                    small_fragment_removed,
                    [contour],
                    contourIdx=-1,
                    color=[0],
                    thickness=cv2.FILLED,
                )

        return small_fragment_removed

    def trim_blank_area(self, binary: np.ndarray):

        # True(文字)の個数を投影
        horizontal_proj = np.count_nonzero(binary, axis=1)
        vertical_proj = np.count_nonzero(binary, axis=0)

        horizontal_thresh = self.cfg.blank_horizontal_thresh
        vertical_thresh = self.cfg.blank_vertical_thresh

        top = 0
        while top < len(horizontal_proj) and horizontal_proj[top] <= horizontal_thresh:
            top += 1

        bottom = len(horizontal_proj) - 1
        while bottom >= top and horizontal_proj[bottom] <= horizontal_thresh:
            bottom -= 1

        left = 0
        while left < len(vertical_proj) and vertical_proj[left] <= vertical_thresh:
            left += 1

        right = len(vertical_proj) - 1
        while right >= left and vertical_proj[right] <= vertical_thresh:
            right -= 1

        # 全て空白ならそのまま返す
        if top > bottom or left > right:
            return binary

        return binary[top : bottom + 1, left : right + 1]

    def visualize_blank_trim(
        self,
        line_removed: np.ndarray,
        blank_trimmed: np.ndarray,
        blank_horizontal_thresh: int,
        blank_vertical_thresh: int,
    ):
        """Visualize blank trimming.

        Displays:
        1. Original (line_removed)
        2. Blank regions highlighted
        3. Cropped result
        """

        h0, w0 = line_removed.shape
        h1, w1 = blank_trimmed.shape

        horizontal_proj = np.count_nonzero(line_removed, axis=1)
        vertical_proj = np.count_nonzero(line_removed, axis=0)
        horizontal_thresh = blank_horizontal_thresh
        vertical_thresh = blank_vertical_thresh

        top = 0
        while top < len(horizontal_proj) and horizontal_proj[top] <= horizontal_thresh:
            top += 1

        bottom = len(horizontal_proj) - 1
        while bottom >= top and horizontal_proj[bottom] <= horizontal_thresh:
            bottom -= 1

        left = 0
        while left < len(vertical_proj) and vertical_proj[left] <= vertical_thresh:
            left += 1

        right = len(vertical_proj) - 1
        while right >= left and vertical_proj[right] <= vertical_thresh:
            right -= 1

        fig, axes = plt.subplots(3, 1, figsize=(8, 8))

        axes[0].imshow(line_removed, cmap="gray")
        axes[0].set_title("Before blank trimming")
        axes[0].axis("off")

        axes[1].imshow(line_removed, cmap="gray")
        overlay_color = "lightskyblue"

        if top > 0:
            axes[1].axhspan(0, top, color=overlay_color, alpha=0.35)
        if bottom < h0 - 1:
            axes[1].axhspan(bottom + 1, h0, color=overlay_color, alpha=0.35)
        if left > 0:
            axes[1].axvspan(0, left, color=overlay_color, alpha=0.35)
        if right < w0 - 1:
            axes[1].axvspan(right + 1, w0, color=overlay_color, alpha=0.35)

        rect = patches.Rectangle(
            (left, top),
            right - left + 1,
            bottom - top + 1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].set_title("Detected blank areas")
        axes[1].axis("off")

        axes[2].imshow(blank_trimmed, cmap="gray")
        axes[2].set_title("After blank trimming")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
