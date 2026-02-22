"""Edge detection utilities for NumPy/OpenCV images."""

import cv2
import numpy as np


def color_to_binary(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a binary mask based on a specific color.

    Args:
        image: Input RGB image as a NumPy array.

    Returns:
        Binary mask where pixels matching the specified color are set to 255, others are set to 0.
    """
    assert image.ndim == 3, "Input image must be a 3-channel RGB image."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def canny(image: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """Apply Canny edge detection after Otsu binarization.

    Args:
        image: Input BGR image with shape ``(H, W, 3)``.
        threshold1: First threshold for the hysteresis procedure.
        threshold2: Second threshold for the hysteresis procedure.

    Returns:
        Binary edge image.
    """
    assert image.ndim == 3, "Input image must be a 3-channel RGB image."

    binary = color_to_binary(image)
    return cv2.Canny(binary, threshold1, threshold2)
