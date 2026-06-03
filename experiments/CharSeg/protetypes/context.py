from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Context:
    image_path: Optional[Path] = None

    # preprocesses
    preprocessed: Optional[np.ndarray] = None

    # line removing
    line_suspect: Optional[np.ndarray] = None
    line_removed: Optional[np.ndarray] = None

    candidates: Optional[np.ndarray] = None
    selected: Optional[np.ndarray] = None
