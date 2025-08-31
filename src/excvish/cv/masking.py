import numpy as np


def binary_image_from_mask(mask: np.ndarray, value: int = 255) -> np.ndarray:
    """Converts a mask to a binary image.

    Args:
        mask (np.ndarray): Mask array whose elements are boolean.
        value (int, optional): Value of the binary image. Defaults to 255.
    """
    h, w = mask.shape
    ret = np.zeros((h, w), dtype=np.uint8)
    ret[mask] = value

    return ret


def image_from_mask(mask: np.ndarray, rgb: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Converts a mask to an RGB image.

    Args:
        mask (np.ndarray): Mask array whose elements are boolean.
        rgb (tuple[int, int, int], optional): RGB value. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: RGB image.
    """
    h, w = mask.shape
    ret = np.zeros((h, w, 3), dtype=np.uint8)

    ret[mask, 0] = rgb[0]
    ret[mask, 1] = rgb[1]
    ret[mask, 2] = rgb[2]

    return ret
