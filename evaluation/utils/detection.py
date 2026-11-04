import numpy as np


def get_mean_centroid_distance(bboxes_a, bboxes_b):

    x_centroids_a = (bboxes_a[:, 0] + bboxes_a[:, 2]) / 2
    y_centroids_a = (bboxes_a[:, 1] + bboxes_a[:, 3]) / 2

    x_centroids_b = (bboxes_b[:, 0] + bboxes_b[:, 2]) / 2
    y_centroids_b = (bboxes_b[:, 1] + bboxes_b[:, 3]) / 2

    mean_dist = np.sqrt((x_centroids_a - x_centroids_b) ** 2 + (y_centroids_a - y_centroids_b) ** 2).mean()

    return mean_dist
