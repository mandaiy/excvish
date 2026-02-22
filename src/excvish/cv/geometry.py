"""Geometry utilities for NumPy/OpenCV images and masks."""

import cv2
import numpy as np

from excvish import BBox, clip


def compute_min_area_rect(
    mask: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Compute minimum-area rectangle from a binary mask.

    Args:
        mask: Binary mask array with non-zero foreground pixels.

    Returns:
        ``((cx, cy), (w, h), angle)`` from ``cv2.minAreaRect`` or ``None`` when
        no valid contour is found.
    """
    if mask is None or mask.size == 0:
        return None

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) == 0:
        return None

    center, size, angle = cv2.minAreaRect(largest_contour)
    return (
        (float(center[0]), float(center[1])),
        (float(size[0]), float(size[1])),
        float(angle),
    )


def compute_horizontal_rotation_angle(
    rect: tuple[tuple[float, float], tuple[float, float], float],
) -> float:
    """Compute rotation angle that aligns the long edge horizontally.

    Args:
        rect: Rectangle tuple returned by ``compute_min_area_rect``.

    Returns:
        Rotation angle in degrees.
    """
    _, (width, height), angle = rect
    return angle + 90 if width < height else angle


def compute_rotated_image_size(height: int, width: int, angle: float) -> tuple[int, int]:
    """Compute output size that avoids clipping after rotation.

    Args:
        height: Source image height.
        width: Source image width.
        angle: Rotation angle in degrees.

    Returns:
        Tuple of ``(new_height, new_width)``.
    """
    angle_rad = np.deg2rad(abs(angle))
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))

    new_width = int(np.ceil(width * cos_a + height * sin_a))
    new_height = int(np.ceil(width * sin_a + height * cos_a))

    return new_height, new_width


def rotate_image(
    image: np.ndarray,
    angle: float,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | tuple[int, ...] = 0,
    expand: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Rotate an image by a given angle.

    Args:
        image: Input image array.
        angle: Rotation angle in degrees.
        border_mode: OpenCV border mode.
        border_value: Constant border fill value.
        expand: If True, resize output to contain full rotated image.
        interpolation: OpenCV interpolation flag.

    Returns:
        Rotated image array.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    if expand:
        new_h, new_w = compute_rotated_image_size(h, w, angle)
    else:
        new_h, new_w = h, w

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand:
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


def rotate_rgba_by_mask_long_edge(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate RGBA image and mask so mask long edge becomes horizontal.

    Args:
        image: RGBA image array with shape ``(H, W, 4)``.

    Returns:
        Tuple of ``(rotated_image, rotated_mask)``.
    """
    alpha = image[..., 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    rect = compute_min_area_rect(mask)
    if rect is None:
        return image, mask
    angle = compute_horizontal_rotation_angle(rect)
    if angle == 0:
        return image, mask
    rotated = rotate_image(
        image,
        angle,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=(0, 0, 0, 0),
        expand=True,
    )
    rotated_mask = rotate_image(
        mask,
        angle,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
        expand=True,
        interpolation=cv2.INTER_NEAREST,
    )
    return rotated, rotated_mask


def crop_frame(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop the frame based on the bounding box.

    Args:
        frame: The input image frame to be cropped.
        bbox: The bounding box instance.

    Returns:
        The cropped portion of the frame according to the bounding box.
    """
    h, w, _ = frame.shape

    x1, y1, x2, y2 = [int(v) for v in bbox.as_xyxy_array()]
    x1 = clip(x1, 0, w)
    x2 = clip(x2, 0, w)
    y1 = clip(y1, 0, h)
    y2 = clip(y2, 0, h)

    return frame[y1:y2, x1:x2, :]


def crop_to_mask_bounds(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to non-zero mask bounds.

    Args:
        image: Input image array.
        mask: Binary mask array.

    Returns:
        Tuple of cropped ``(image, mask)``. Returns originals when mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return image, mask
    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1
    return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]
