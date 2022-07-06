import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Dict
import math
from scipy.signal import find_peaks


def build_features(
    shape: NDArray, num_edges: int, duplicate_radius: int
) -> Dict:
    """Builds Computer Vision features for each Jigsaw puzzle
    ::parameters
    shape: a numpy shape of the Jigsaw Puzzle

    ::returns
    Dictionary in the structure of"""
    retdic = {}
    retdic["contour"], _ = cv2.findContours(
        shape, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    retdic["dst_norm"] = np.empty(shape.shape, dtype=np.float32)
    cv2.drawContours(
        retdic["dst_norm"], retdic["contour"], -1, (255, 0, 0), 1
    )
    retdic["centroid"] = find_centroid_of_contour(retdic["contour"])
    retdic["distance"] = [
        calc_dist(point[0], retdic["centroid"])
        for point in retdic["contour"][0]
    ]
    # retdic["peaks"] = find_peaks_idx(
    #     retdic["distance"], num_edges, duplicate_radius
    # )
    distance_arr = np.array(retdic["distance"])
    retdic["peaks"], _ = find_peaks(distance_arr, distance=duplicate_radius)
    retdic["dst_norm_edges"] = draw_edges(
        retdic["dst_norm"], retdic["peaks"], retdic["contour"]
    )
    return retdic


def find_centroid_of_contour(contours):
    Xs, Ys = [], []
    for point in contours[0]:
        Ys.append(point[0, 0])
        Xs.append(point[0, 1])

    centroid = (int(sum(Ys) / len(Ys)), int(sum(Xs) / len(Xs)))
    return centroid


def calc_dist(point_A, point_B) -> float:
    """
    Calculates the distance between two points
    parameters:: Point_A

    returns:: the distance (float)
    """
    a_y = point_A[0]
    a_x = point_A[1]
    b_y = point_B[0]
    b_x = point_B[1]
    dist = ((a_y - b_y) ** 2) + ((a_x - b_x) ** 2)
    return float(math.sqrt(dist))


# def find_peaks_idx(distances, num_peaks, duplicate_radius=10):
#       Superseeded with scipy.signal.find_peaks
#     """
#     Takes a 1D array, and finds the peaks.
#     parameters:
#     distances: a 1D array, in the form of a list. Each element includes an elevation (float)
#     num_peaks: integer, number of peaks to return
#     duplicate_radius: integer to determine whether two points are duplicate or not

#     returns:
#     peaks_id: a list of length 'num_peaks'
#     """
#     peaks_idx = []
#     ids_to_remove = []
#     length = len(distances)
#     for idx, point in enumerate(distances):
#         neighbor_l_1 = (idx - 1) % length
#         neighbor_r_1 = (idx + 1) % length
#         neighbor_l_2 = (idx - 5) % length
#         neighbor_r_2 = (idx + 5) % length
#         if (
#             distances[neighbor_l_2]
#             <= distances[neighbor_l_1]
#             <= point
#             >= distances[neighbor_r_1]
#             >= distances[neighbor_r_2]
#         ):
#             for existing_point, existing_idx in peaks_idx:
#                 if abs(idx - existing_idx) < duplicate_radius:
#                     ids_to_remove.append(idx)

#             if idx not in ids_to_remove:
#                 peaks_idx.append((point, idx))

#     peaks_idx = sorted(peaks_idx)[::-1]
#     return peaks_idx[0:num_peaks]


def draw_edges(dst_norm, peaks, contours):
    dst_norm2 = dst_norm
    for idx in peaks:
        # for dist, idx in peaks:
        edge = contours[0][idx][0]
        edge_y = edge[0]
        edge_x = edge[1]
        cv2.circle(dst_norm2, (edge_y, edge_x), 10, (255, 0, 0), 3)
    return dst_norm2
