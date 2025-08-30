from typing import Literal

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity as ssim


def chamfer_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Chamfer distance between two binary images.

    The Chamfer distance is a metric that measures the similarity between two sets of points
    by computing the average of the minimum distances from each point in one set to the
    nearest point in the other set, and vice versa.

    Args:
        image1 (np.ndarray): First binary image where non-zero pixels represent the shape.
        image2 (np.ndarray): Second binary image where non-zero pixels represent the shape.

    Returns:
        float: The Chamfer distance between the two images. Lower values indicate
               higher similarity between the shapes.

    Example:
        >>> img1 = np.zeros((100, 100))
        >>> img2 = np.zeros((100, 100))
        >>> img1[40:60, 40:60] = 1  # Square in image1
        >>> img2[45:65, 45:65] = 1  # Slightly offset square in image2
        >>> distance = chamfer_distance(img1, img2)
    """
    p1 = np.column_stack(np.where(image1 > 0))
    p2 = np.column_stack(np.where(image2 > 0))

    t1 = cKDTree(p1)
    t2 = cKDTree(p2)

    d1, _ = t1.query(p2)
    d2, _ = t2.query(p1)

    return float(np.mean(d1) + np.mean(d2))


def symmetricality_ssim(image: np.ndarray, direction: Literal["horizontal", "vertical"] = "horizontal") -> float:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    match direction:
        case "horizontal":
            pad = 1 if w % 2 == 1 else 0
            mid = w // 2
            l = image[:, :mid]
            r = np.fliplr(image[:, pad + mid :])

            return float(ssim(l, r))

        case "vertical":
            pad = 1 if h % 2 == 1 else 0
            mid = h // 2
            t = image[:mid, :]
            b = np.flipud(image[pad + mid :, :])

            return float(ssim(t, b))

        case _:
            raise NotImplementedError(f"Not implemented yet: {direction}")  # type: ignore[unreachable]


def symmetricality_chamfer_distance(
    image: np.ndarray, direction: Literal["horizontal", "vertical"] = "horizontal"
) -> float:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    match direction:
        case "horizontal":
            pad = 1 if w % 2 == 1 else 0
            mid = w // 2
            l = image[:, :mid]
            r = np.fliplr(image[:, pad + mid :])

            return chamfer_distance(l, r)

        case "vertical":
            pad = 1 if h % 2 == 1 else 0
            mid = h // 2
            t = image[:mid, :]
            b = np.flipud(image[pad + mid :, :])

            return chamfer_distance(t, b)

        case _:
            raise NotImplementedError(f"Not implemented yet: {direction}")  # type: ignore[unreachable]
