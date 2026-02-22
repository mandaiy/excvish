"""Drawing helpers for NumPy/OpenCV images."""

import cv2
import numpy as np

from excvish import BBox


def choose_text_color_bgr(
    background_color_bgr: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Choose black or white text color based on the background color for better visibility.

    Args:
        background_color_bgr: Background color in BGR format.

    Returns:
        Text color in BGR format (either black or white).
    """
    if background_color_bgr is None:
        return (255, 255, 255)

    b, g, r = background_color_bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    background_color: tuple[int, int, int] | None = (28, 28, 28),
) -> None:
    """Draw text on an image frame at the specified position.

    Args:
        frame: NumPy array representing the image frame.
        text: String to draw on the image.
        position: Tuple of ``(x, y)`` coordinates for text origin.
        background_color: Optional BGR background color behind text.
    """
    # Check if frame is valid
    if frame is None or frame.size == 0:
        return

    image_height, image_width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    x1, y1 = position
    w, h = cv2.getTextSize(text, 0, font_scale, thickness)[0]
    h += 3  # padding

    if x1 > image_width - w:
        x1 = image_width - w

    if y1 >= h:
        y1 = y1 - 2
    else:
        y1 = y1 + h - 1

    x2 = x1 + w
    if y1 >= h:
        y2 = y1 - h
    else:
        y2 = y1 + h

    text_color = choose_text_color_bgr(background_color)

    if background_color is not None:
        # Fill rectangle for text background
        cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, -1, cv2.LINE_AA)

    # Put text on the frame
    cv2.putText(frame, text, (x1, y1), font, font_scale, text_color, thickness)


def draw_bbox(
    frame: np.ndarray,
    bbox: BBox,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Draw a bounding box on an image frame.

    Args:
        frame: The input image frame.
        bbox: The bounding box instance.
        color: Color of the bounding box in BGR format.
        thickness: Thickness of the bounding box lines.
        line_type: OpenCV line type.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox.as_xyxy_array()]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
