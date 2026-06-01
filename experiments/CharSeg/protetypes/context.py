from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class Context:
    image_path: Path = None
    resized: np.ndarray = None
    preprocessed: np.ndarray = None
    line_removed: np.ndarray = None
    candidates: np.ndarray = None
