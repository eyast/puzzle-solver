from typing import Dict, List, Tuple
from .utilities import jpg_to_pieces

import numpy as np
from numpy.typing import NDArray


class Universe:
    """A universe where pieces exist.

    params::
    None

    Methods:
    populate: Creates a new universe
    save: saves the universe to disk

    """

    def __init__(self):
        pass

    def populate(
        self,
        greenOnly: bool,
        src: str,
        threshold_value: int,
        blur_radius: int,
        num_edges: int,
        num_pieces: int,
        duplicate_radius: int,
    ):
        """Populates the universe based on JPG images in a folder

        params:
        src: source folder to scan for JPG files
        num_edges: number of edges each Jigsaw piece has
        num_pieces: number of Jigsaw pieces per JPG scan

        Returns: None
        """
        self._pieces: List = jpg_to_pieces(
            src=src,
            greenOnly=greenOnly,
            threshold_value=threshold_value,
            blur_radius=blur_radius,
            num_pieces=num_pieces,
            num_edges=num_edges,
            duplicate_radius=duplicate_radius,
        )


class Puzzle:
    """A Puzzle object composed of m rows and n columns.
    Rows and Columns are separated by edge_vectors.
    """

    def __init__(self, edges: int, pieces: Tuple[int, int]):
        self._edges: int = edges
        self._pieces: Tuple[int, int] = pieces
        _height, _width = pieces[0], pieces[1]
        self.puzzle = np.zeros((_height, _width))
