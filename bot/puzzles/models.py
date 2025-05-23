from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PuzzlePiece:
    """
    Represents a single puzzle piece with its properties.
    """

    id: int
    # List of (x,y) coordinates forming the piece boundary
    corners: List[List[float]]
    center: Tuple[int, int]  # Center point of the piece
    image: np.ndarray  # The actual image data for this piece
    area: float  # Area of the piece in pixels
