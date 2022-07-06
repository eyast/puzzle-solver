from numpy.typing import NDArray
from .build_features import build_features


class Piece:
    """A Jigsaw puzzle piece

    Parameters:
    :: id = an identifier
    :: shape = Numpy Array that includes the shape
    :: md = metadata, Dict, with shape
    {"x": int,
     "y" : int,
     "width": int,
     "height": int}
    """

    def __init__(
        self,
        id: int,
        shape: NDArray,
        num_edges: int,
        duplicate_radius: int,
    ):
        self._id = id
        self._shape = shape
        self._is_assigned = False
        self._num_edges = num_edges
        self._features = build_features(
            shape, num_edges, duplicate_radius
        )
