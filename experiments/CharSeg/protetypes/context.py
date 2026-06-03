from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np


@dataclass(slots=True)
class Context:
    image_path: Optional[Path] = None

    # preprocesses
    preprocessed: Optional[np.ndarray] = None

    # line removing
    line_suspect: Optional[np.ndarray] = None
    line_removed: Optional[np.ndarray] = None
    # raising borderline candidates
    candidates: Optional[List[np.ndarray]] = None
    # selecting borderline
    selected: Optional[List[np.ndarray]] = None
    # cropped images
    crops: Optional[List[np.ndarray]] = None
