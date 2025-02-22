import numpy as np


def spectral_angle_mapper(image_array: np.ndarray, reference_spectrum: np.ndarray) -> np.ndarray:
    """Calculates the Spectral Angle Mapper (SAM) for all pixels in the image
    relative to the reference_spectrum.

    Parameters:
    - image_array: numpy array of shape (h, w, b) representing the spectral image.
    - x: int, x-coordinate of the reference pixel.
    - y: int, y-coordinate of the reference pixel.

    Returns:
        sam_array: numpy array of shape (h, w) with SAM values for each pixel.
    """
    # Calculate the norms of the reference spectrum
    reference_norm = float(np.linalg.norm(reference_spectrum))

    # Calculate the norms of all pixels in the image
    pixel_norms = np.linalg.norm(image_array, axis=2)

    # Calculate dot products between reference spectrum and each pixel spectrum
    dot_products = np.sum(image_array * reference_spectrum, axis=2)

    # Avoid division by zero by replacing 0 norms with a small value
    pixel_norms = np.where(pixel_norms == 0, 1e-10, pixel_norms)
    reference_norm = max(reference_norm, 1e-10)

    # Compute SAM angle (in radians) for each pixel
    cos_theta = np.clip(dot_products / (reference_norm * pixel_norms), -1.0, 1.0)
    sam_array = np.arccos(cos_theta)  # Calculate angle in radians

    return sam_array
