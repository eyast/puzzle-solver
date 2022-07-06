import glob
import math
import os
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def jpg_to_pieces(
    src: str,
    greenOnly: bool,
    threshold_value: int,
    blur_radius: int,
    num_pieces: int,
    num_edges: int,
    duplicate_radius: int,
) -> List[NDArray]:
    """Core function that loads images, applies some CV, and returns Jigsaw pieces
    ::parameters:
    images: List of file paths to open
    blur_radius: Blur Radius to apply
    """
    img_raw = load_files(src, greenOnly)
    (
        img_threshold,
        img_connected,
    ) = find_threshold_and_connected_components(
        threshold_value, blur_radius, img_raw
    )
    return create_pieces(
        num_pieces,
        img_threshold,
        img_connected,
        num_edges,
        duplicate_radius,
    )


def create_pieces(
    num_pieces,
    img_threshold,
    img_connected,
    num_edges,
    duplicate_radius,
):
    """Returns a list of Puzzle pieces"""
    from .piece import Piece

    t = 0
    all_returns: List = []
    for _, paper in enumerate(img_connected):
        data = img_threshold[_]
        stats, idx = pick_id_of_top_components(paper, num_pieces)
        for id in idx:
            x, y, width, height = extract_coordinates(stats[id])
            section = np.zeros_like(paper[1])
            section[paper[1] == id] = 255
            section = section[y : y + height, x : x + width]
            piece = Piece(
                id=t,
                shape=section,
                num_edges=num_edges,
                duplicate_radius=duplicate_radius,
            )
            all_returns.append(piece)
            t += 1
    return all_returns


def find_threshold_and_connected_components(
    threshold_value, blur_radius, img_raw
) -> Tuple[List, List]:
    """Performs image processing"""
    img_gray: List[NDArray] = [
        cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in img_raw
    ]
    img_blurred: List[NDArray] = [
        cv2.blur(_, (blur_radius, blur_radius)) for _ in img_gray
    ]
    img_threshold: List[NDArray] = [
        ~cv2.threshold(_, threshold_value, 255, cv2.THRESH_OTSU)[1]
        for _ in img_blurred
    ]
    img_connected: List[List[NDArray]] = [
        cv2.connectedComponentsWithStats(_) for _ in img_threshold
    ]

    return (img_threshold, img_connected)


def load_files(src, greenOnly):
    """Loads files from disk
    parameters:
    src: str = a source directory that contains JPG files of Puzzle scans"""
    assert os.path.exists(src)
    files: List[str] = glob.glob(os.path.join(src, "*.JPG"))
    img_raw: List[NDArray] = [cv2.imread(_) for _ in files]
    if greenOnly:
        img_raw: List[NDArray] = [extract_green(i) for i in img_raw]
    return img_raw


def extract_green(image, debug: bool= True):
    """Experiment to extract only green pieces from an image
    
    parameters:
    -----------
    image from cv2.imread()

    returns:
    --------
    single channel with green = 1
    """
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 100, 20])
    higher_green = np.array([40, 160, 80])
    mask = cv2.inRange(image, lower_green, higher_green)
    result = cv2.bitwise_and(image, image, mask)
    if debug:
        cv2.imshow(winname="Result", mat=result)
        cv2.imshow(winname="mask", mat=mask)
        cv2.imshow(winname="original", mat=image)
        cv2.waitKey()
    return result

def extract_coordinates(row: NDArray) -> Tuple[int, int, int, int]:
    """For a row from cv2ConnectedComponentsWithStats
    return the location of x, y, width, and height respectively
    :: parameters
    row: a Numpy array row

    returns:
    x, y, width, height of connected component
    """
    x = row[0]
    y = row[1]
    width = row[2]
    height = row[3]
    return (x, y, width, height)


def pick_id_of_top_components(
    cv2_stats: NDArray, n: int
) -> Tuple[NDArray, NDArray]:
    """Given a list of IDs (result of cv2.connectedComponentsWithStats()
    determine the row ids of the n largest components
    :: parameters

    cv2_stats: numpy array that includes connectedComponentsStats
    n : number of top components to return

    returns:
    Stats: Numpy array that includes cvConnectedComponentsStats
    idx: numpy array that includes the ids of top components

    """
    stats = cv2_stats[2]
    idx = stats[:, -1]
    idx = np.argsort(idx)[::-1]
    idx = idx[1 : n + 1]
    return (stats, idx)
