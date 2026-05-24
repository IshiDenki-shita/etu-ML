from dataclasses import dataclass
import numpy as np


@dataclass
class SegmentationContext:
    original_image: np.ndarray

    binary: np.ndarray | None = None
    removed_binary: np.ndarray | None = None
    valley_points: np.ndarray | None = None
    candidate_boundaries: list | None = None
    selected_boundaries: list | None = None
    character_images: list | None = None
