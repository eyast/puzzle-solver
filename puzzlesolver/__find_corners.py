#%% load libraries and files
from typing import Iterable
import cv2
import matplotlib.pyplot as plt
import numpy as np


filename = "02.JPG"
puzzles = cv2.imread(filename)
puzzles = cv2.cvtColor(puzzles, cv2.COLOR_BGR2GRAY)


def calc_edges(thresh):
    dst = cv2.cornerHarris(best, 2, 3, 0.04)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(
        dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    num_edges = 0
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv2.circle(dst_norm_scaled, (j, i), 5, (0), 2)
                num_edges += 1
    cv2.namedWindow("corners")
    cv2.putText(
        dst_norm_scaled,
        str(num_edges),
        (50, 50),
        cv2.FONT_HERSHEY_PLAIN,
        4,
        (0, 0, 0),
    )
    contours, hierarchy = cv2.findContours(
        best, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(dst_norm_scaled, contours, -1, (255, 0, 0), 1)
    # dst_norm_scaled = cv2.resize(dst_norm_scaled, (750, 1000))
    cv2.imshow("corners", dst_norm_scaled)


blurred = cv2.blur(puzzles, (7, 7))
best = cv2.threshold(blurred, 110, 255, type=cv2.THRESH_BINARY)[1]

cv2.namedWindow("source")
cv2.createTrackbar("Threshold:", "source", 200, 255, calc_edges)
# puzzles = cv2.resize(puzzles, (750, 1000))
cv2.imshow("source", puzzles)
cv2.waitKey()
