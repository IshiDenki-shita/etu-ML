from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, TypeAlias

import numpy as np

# 上端→下端の折れ線。各要素は [x, y]（画素座標）。
Borderline: TypeAlias = list[list[int]]


@dataclass(slots=True)
class Context:
    image_path: Optional[Path] = None

    # preprocesses
    preprocessed: Optional[np.ndarray] = None

    # line removing
    line_suspect: Optional[np.ndarray] = None
    line_removed: Optional[np.ndarray] = None

    # trimming blank space
    blank_trimmed: Optional[np.ndarray] = None

    # raising borderline candidates
    candidates: Optional[List[Borderline]] = None
    candidate_costs: Optional[List[float]] = None

    # selecting borderline
    selected: Optional[List[Borderline]] = None

    # cropped images
    crops: Optional[List[np.ndarray]] = None
