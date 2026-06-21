"""
common visualization of 5 picture along five process
(raw, binarized, line_removed, candidates, selected)
"""

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from experiments.CharSeg.protetypes.GenCandidates.ConnectLine import (
    Borderline,
    borderline_to_points,
)
from experiments.CharSeg.protetypes.context import Context


@dataclass
class visualizationConfig:
    enabled: bool = True
    window_size: int = 5


class Visualizer:
    def __init__(self, debug: bool = True) -> None:
        self.config = visualizationConfig(enabled=debug)

    def process(self, context: Context):
        logging.info("結果を表示します。（共通項目）")

        if not self.config.enabled:
            return

        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(8, 8),
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
                "Blank Trimmed",
                context.blank_trimmed,
                None,
                None,
            ),
            (
                "Candidate Lines",
                context.blank_trimmed,
                context.candidates,
                "Reds",
            ),
            (
                "Selected Lines",
                context.blank_trimmed,
                context.selected,
                "Blues",
            ),
        ]

        for ax, (title, image, overlay, cmap) in zip(axes, images):
            if image is None:
                ax.set_title(f"{title} (None)")
                ax.axis("off")
                continue

            if overlay is None:
                ax.imshow(image, cmap="gray")
            elif isinstance(overlay, list):
                self._overlay_borderlines(ax, image, overlay, cmap)
            else:
                self._overlay_mask(ax, image, overlay, cmap)
            ax.set_title(title)
            ax.axis("off")

        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.98,
            bottom=0.02,
            hspace=0.03,
        )
        assert fig.canvas.manager is not None
        fig.canvas.manager.set_window_title("Character Segmentation Visualization")
        plt.show()

    def _overlay_borderlines(
        self,
        ax,
        base_image: np.ndarray,
        borderlines: list[Borderline],
        cmap_name: str,
    ) -> None:
        ax.imshow(base_image, cmap="gray")
        if not borderlines:
            return

        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.35, 0.95, len(borderlines)))

        for color, borderline in zip(colors, borderlines):
            if len(borderline) < 2:
                continue
            pts = borderline_to_points(borderline)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linewidth=1.5,
            )

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
