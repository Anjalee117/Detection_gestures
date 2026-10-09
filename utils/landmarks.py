"""Normalize one hand's 21 pixel landmarks without mutating the input."""
import numpy as np


def pre_process_landmark(landmark_list):
    points = np.asarray(landmark_list, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite (x, y) landmark pairs")
    relative = points - points[0]
    maximum = np.max(np.abs(relative))
    if maximum:
        relative = relative / maximum
    return relative.ravel().tolist()
