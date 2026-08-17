from pathlib import Path
import numpy as np


def load_calibration(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("calibration must be a finite 4x4 matrix")
    return matrix

