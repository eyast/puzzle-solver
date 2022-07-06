"""
Converts image files to Puzzle objects,
Eyas Taifour
29/06/2022
"""

import os
import matplotlib.pyplot as plt

from puzzlesolver.datatypes import Puzzle, Universe

FOLDER = os.path.join(os.getcwd(), "data")

if __name__ == "__main__":
    universe = Universe()
    universe.populate(
        greenOnly=False,
        src=FOLDER,
        threshold_value=110,
        blur_radius=3,
        num_edges=4,
        num_pieces=4,
        duplicate_radius=250,
    )
    for i in range(4):
        plt.imshow(
            universe._pieces[i]._features["dst_norm"],
            cmap="gray",
        )
        plt.show()
        plt.close()