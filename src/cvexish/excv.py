from typing import Literal

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity as ssim


def chamfer_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    p1 = np.column_stack(np.where(image1 > 0))
    p2 = np.column_stack(np.where(image2 > 0))

    t1 = cKDTree(p1)
    t2 = cKDTree(p2)

    d1, _ = t1.query(p2)
    d2, _ = t2.query(p1)

    return np.mean(d1) + np.mean(d2)


def symmetricality_ssim(image: np.ndarray, direction: Literal["horizontal", "vertical"] = "horizontal") -> float:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, w = image.shape

    if direction == "horizontal":
        pad = 1 if w % 2 == 1 else 0
        mid = w // 2
        l = image[:, :mid]
        r = np.fliplr(image[:, pad + mid :])

        return ssim(l, r)

    raise NotImplementedError(f"Not implemented yet: {direction}")


def symmetricality_chamfer_distance(
    image: np.ndarray, direction: Literal["horizontal", "vertical"] = "horizontal"
) -> float:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    if direction == "horizontal":
        pad = 1 if w % 2 == 1 else 0
        mid = w // 2
        l = image[:, :mid]
        r = np.fliplr(image[:, pad + mid :])
        return chamfer_distance(l, r)

    raise NotImplementedError(f"Not implemented yet: {direction}")


def remove_smaller_components(image: np.ndarray, area_thresh: int) -> np.ndarray:
    """Removes connected components whose area is smaller than or equal to the threshold.

    Args:
        image (np.ndarray): Binary image.
        area_thresh (int): Area threshold.

    Returns:
        np.ndarray: Image with smaller components removed.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image)

    labels_whose_area_is_greater = np.argwhere(stats[:, 4] >= area_thresh).flatten()
    return np.where(np.isin(labels, labels_whose_area_is_greater), image, 0)


def remove_larger_components(image: np.ndarray, area_thresh: int) -> np.ndarray:
    """Removes connected components whose area is larger than the threshold.

    Args:
        image (np.ndarray): Binary image.
        area_thresh (int): Area threshold.

    Returns:
        np.ndarray: Image with larger components removed.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image)

    labels_whose_area_is_smaller = np.argwhere(stats[:, 4] < area_thresh).flatten()
    return np.where(np.isin(labels, labels_whose_area_is_smaller), image, 0)


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


def color_labels(labels: np.ndarray, color_distance_threshold: float = 60.0) -> np.ndarray:
    """Colorizes labels so that adjacent labels do not have similar colors.

    Args:
        labels (np.ndarray): Label image. Typically cv2.connectedComponents output.
        color_distance_threshold (float, optional):
            Minimum Euclidean distance of colors between adjacent labels. Defaults to 60.0.

    Returns:
        np.ndarray: BGR image.
    """
    h, w = labels.shape

    # ユニークなラベル一覧を取得 (背景 0 を含む)
    unique_labels = np.unique(labels)
    max_label = unique_labels.max()

    # ------------------------------------------------------------
    # 1) Build adjacency relationships (graph)
    #    Here, we define "adjacent" as "different labels in neighboring right or down pixels"
    # ------------------------------------------------------------
    adjacency: dict = {lab: set() for lab in unique_labels}  # Set of labels adjacent to each label

    for r in range(h):
        for c in range(w):
            current_label = labels[r, c]
            # Right pixel
            if c < w - 1:
                right_label = labels[r, c + 1]
                if right_label != current_label:
                    adjacency[current_label].add(right_label)
                    adjacency[right_label].add(current_label)

            # Down pixel
            if r < h - 1:
                down_label = labels[r + 1, c]
                if down_label != current_label:
                    adjacency[current_label].add(down_label)
                    adjacency[down_label].add(current_label)

    # ------------------------------------------------------------
    # 2) Color assignment (graph coloring)
    #    - Ensure a minimum color distance between adjacent labels
    # ------------------------------------------------------------
    # Table to store label ID → BGR color
    color_map = np.zeros((max_label + 1, 3), dtype=np.uint8)

    # Fixed black for background (label=0)
    color_map[0] = (0, 0, 0)

    def color_distance(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """Euclidean distance in BGR space"""
        return np.sqrt(np.sum((c1 - c2) ** 2))

    def can_use_color(candidate_color: np.ndarray, assigned_neighbors: list) -> bool:
        """
        Determines if candidate_color is sufficiently distant from
        the colors of already assigned adjacent labels
        """
        for neighbor_label in assigned_neighbors:
            dist = color_distance(candidate_color, color_map[neighbor_label])
            if dist < color_distance_threshold:
                return False
        return True

    # Iterate through labels in ascending order (1 ~ max_label), skipping background (0)
    # Labels from connectedComponents are sequential (0 to num_labels-1),
    # but we process using unique_labels to be safe
    for lab in unique_labels:
        if lab == 0:
            continue  # Skip background as it's fixed
        neighbors = adjacency[lab]  # Adjacent labels

        # Generate random colors until we find one that's sufficiently distant
        trials = 0
        while True:
            trials += 1
            candidate = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            # Check only against adjacent labels that already have colors assigned
            assigned_neighbors = [n for n in neighbors if (n < lab and n != 0)]
            if can_use_color(candidate, assigned_neighbors):
                color_map[lab] = candidate
                break
            # Prevent infinite loops
            if trials > 1000:
                # If we can't find a suitable color, compromise and use the current candidate
                color_map[lab] = candidate
                break

    # ------------------------------------------------------------
    # 3) Apply the corresponding color to each pixel based on its label
    # ------------------------------------------------------------
    colored_result = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            colored_result[r, c] = color_map[labels[r, c]]

    return colored_result
