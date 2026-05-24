import matplotlib.pyplot as plt
import numpy as np


def visualize_result(
    removed_binary: np.ndarray,
    line_map: np.ndarray,
    valley_line_map: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 6))

    axes[0].imshow(removed_binary, cmap="gray")
    axes[0].set_title("Line Removed Binary")
    axes[0].axis("off")

    axes[1].imshow(line_map, cmap="gray")
    axes[1].set_title("Detected Line Map")
    axes[1].axis("off")

    ys_valley, xs_valley = np.where(valley_line_map > 0)

    axes[2].imshow(removed_binary, cmap="gray")
    axes[2].scatter(xs_valley, ys_valley, s=1)
    axes[2].set_title("Valley Point Scatter")
    axes[2].axis("off")

    plt.subplots_adjust(hspace=0.02, top=0.98, bottom=0.02)
    plt.show()
