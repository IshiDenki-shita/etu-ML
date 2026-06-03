"""
common visualization of 5 picture along five process
(raw, binarized, line_removed, candidates, selected)
"""

import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

import numpy as np

from experiments.CharSeg.protetypes.context import Context


@dataclass
class visualizationConfig:
    enabled: bool = True
    window_size: int = 5


class Visualizer:
    def __init__(self, enabled: bool = True) -> None:
        self.config = visualizationConfig(enabled=enabled)

    def _overlay_mask(
        self,
        ax,
        base_image: np.ndarray,
        mask: np.ndarray | None,
        cmap: str,
    ) -> None:
        ax.imshow(base_image, cmap="gray")

        if mask is None:
            return

        masked = np.ma.masked_where(mask == 0, mask)

        ax.imshow(
            masked,
            cmap=cmap,
            alpha=0.6,
        )

    def process(self, context: Context):
        if not self.config.enabled:
            return

        logging.info("結果を表示します。（共通項目）")

        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(8, 7),
            constrained_layout=False,
        )

        axes = np.atleast_1d(axes)

        images = [
            (
                "Preprocessed",
                context.preprocessed,
                None,
                None,
            ),
            (
                "Line Removed",
                context.line_removed,
                None,
                None,
            ),
            (
                "Candidate Lines",
                context.line_removed,
                getattr(context, "candidates", None),
                "Reds",
            ),
            (
                "Selected Lines",
                context.line_removed,
                getattr(context, "selected", None),
                "Blues",
            ),
        ]

        for ax, (title, image, mask, cmap) in zip(axes, images):
            if image is None:
                ax.set_title(f"{title} (None)")
                ax.axis("off")
                continue

            if mask is None:
                ax.imshow(image, cmap="gray")
            else:
                self._overlay_mask(
                    ax,
                    image,
                    mask,
                    cmap,
                )
            ax.set_title(title)
            ax.axis("off")

        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.98,
            bottom=0.02,
            hspace=0.03,
        )
        fig.canvas.manager.set_window_title("Character Segmentation Visualization")
        plt.show()
