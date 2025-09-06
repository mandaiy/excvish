import cv2
import numpy as np

from excvish import BBox


def _non_max_suppression(
    bbox_list: list[tuple[int, int, int, int]],
    overlap_threshold: float,
) -> np.ndarray:
    if len(bbox_list) == 0:
        return np.array([])

    bboxes = np.array(bbox_list)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    pick = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:-1]])
        yy1 = np.maximum(y1[i], y1[indices[:-1]])
        xx2 = np.minimum(x2[i], x2[indices[:-1]])
        yy2 = np.minimum(y2[i], y2[indices[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[indices[:-1]]
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return bboxes[pick].astype("int")


def _match_template(
    img: np.ndarray, template: np.ndarray, threshold_matching: float = 0.99, threshold_nms: float = 0.3
) -> np.ndarray:
    result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    h, w, _ = template.shape

    locations = np.where(result >= threshold_matching)
    rects = [(x, y, x + w, y + h) for (x, y) in zip(*locations[::-1], strict=False)]

    return _non_max_suppression(rects, threshold_nms)


def find_matched_template(
    img: np.ndarray,
    template: np.ndarray,
    threshold_matching: float = 0.99,
    threshold_nms: float = 0.3,
) -> list[BBox]:
    """Finds locations in the image that match the given template.
    Args:
        img (np.ndarray): Input image in which to search for the template.
        template (np.ndarray): Template image to search for.
        threshold_matching (float, optional): Similarity threshold for template matching. Defaults to 0.99.
        threshold_nms (float, optional): Overlap threshold for non-maximum suppression. Defaults to 0.3.
    Returns:
        list[BBox]: List of bounding boxes where the template matches the image.
    """
    final_boxes = _match_template(img, template, threshold_matching, threshold_nms)

    return [BBox.from_xyxy_array(arr) for arr in final_boxes]
