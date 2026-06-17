# /Users/matsukou/Desktop/projects/ETU/etu-ML/.venv/bin/python -c "from experiments.segmentation.pipeline.segmenter import CharacterSegmentationConfig, CharacterSegmenter; from experiments.segmentation.visualization.debug_visualizer import visualize_context; s=CharacterSegmenter(CharacterSegmentationConfig()); s.run(); visualize_context(s.context)"

import matplotlib.pyplot as plt
import numpy as np

from experiments.segmentation.context.segmentation_context import SegmentationContext


def _safe_aximshow(ax, img, cmap="gray", title=None):
    if img is None:
        ax.set_title(f"{title} (None)")
        ax.axis("off")
        return

    ax.imshow(img, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.axis("off")


def visualize_context(context: SegmentationContext) -> None:
    """
    Debug visualizer for segmentation context.

    Shows: binary, removed_binary, valley_points, candidate_boundaries, selected_boundaries
    This function is intended for research/debug use only and is not imported by algorithm modules.
    """
    titles = [
        "binary",
        "removed_binary",
        "valley_points",
        "candidate_boundaries",
        "selected_boundaries",
    ]

    fig, axes = plt.subplots(
        len(titles),
        1,
        figsize=(8, 1.5 * len(titles)),
    )
    if len(titles) == 1:
        axes = [axes]

    # binary
    _safe_aximshow(axes[0], context.binary, title="binary")

    # removed_binary
    _safe_aximshow(axes[1], context.removed_binary, title="removed_binary")

    # valley points overlay on removed_binary
    ax2 = axes[2]
    if context.removed_binary is None:
        ax2.set_title("valley_points (None)")
        ax2.axis("off")
    else:
        ax2.imshow(context.removed_binary, cmap="gray")
        if context.valley_points is not None:
            ys, xs = np.where(context.valley_points > 0)
            ax2.scatter(xs, ys, s=1, c="red")
        ax2.set_title("valley_points")
        ax2.axis("off")

    # candidate boundaries
    ax3 = axes[3]
    if context.removed_binary is not None:
        ax3.imshow(context.removed_binary, cmap="gray")
    if context.candidate_boundaries:
        for b in context.candidate_boundaries:
            # try to draw vertical lines if boundary is scalar or pair
            try:
                if isinstance(b, (int, np.integer)):
                    ax3.axvline(b, color="yellow", linewidth=1)
                elif (
                    hasattr(b, "__len__")
                    and len(b) >= 2
                    and isinstance(b[0], (int, np.integer))
                ):
                    x = int(b[0])
                    ax3.axvline(x, color="yellow", linewidth=1)
            except Exception:
                continue
    ax3.set_title("candidate_boundaries")
    ax3.axis("off")

    # selected boundaries
    ax4 = axes[4]
    if context.removed_binary is not None:
        ax4.imshow(context.removed_binary, cmap="gray")
    if context.selected_boundaries:
        for b in context.selected_boundaries:
            try:
                if isinstance(b, (int, np.integer)):
                    ax4.axvline(b, color="lime", linewidth=1)
                elif (
                    hasattr(b, "__len__")
                    and len(b) >= 2
                    and isinstance(b[0], (int, np.integer))
                ):
                    x = int(b[0])
                    ax4.axvline(x, color="lime", linewidth=1)
            except Exception:
                continue
    ax4.set_title("selected_boundaries")
    ax4.axis("off")

    plt.tight_layout(pad=0.5)
    plt.show()
