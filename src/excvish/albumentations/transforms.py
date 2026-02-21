"""Albumentations custom transforms used in excvish."""

import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

from ..cv.colors import apply_shades_of_gray
from ..cv.geometry import compute_horizontal_rotation_angle, compute_min_area_rect, rotate_image


class AlignLongEdgeHorizontal(DualTransform):
    """Rotate image and mask so the mask long edge becomes horizontal."""

    def __init__(
        self,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: int | tuple[int, ...] = 0,
        mask_value: int = 0,
        p: float = 1.0,
    ):
        """Initialize transform parameters.

        Args:
            border_mode: OpenCV border mode for image rotation.
            border_value: Border fill value for image rotation.
            mask_value: Border fill value for mask rotation.
            p: Probability of applying the transform.
        """
        super().__init__(p=p)
        self.border_mode = border_mode
        self.border_value = border_value
        self.mask_value = mask_value

    def apply(self, img: np.ndarray, angle: float = 0, **params) -> np.ndarray:
        """Rotate an image with the computed angle.

        Args:
            img: Input image.
            angle: Rotation angle in degrees.
            **params: Additional Albumentations parameters.

        Returns:
            Rotated image.
        """
        if angle == 0:
            return img
        return rotate_image(
            img,
            angle,
            border_mode=self.border_mode,
            border_value=self.border_value,
            expand=True,
        )

    def apply_to_mask(self, mask: np.ndarray, angle: float = 0, **params) -> np.ndarray:
        """Rotate a mask with the computed angle.

        Args:
            mask: Input mask.
            angle: Rotation angle in degrees.
            **params: Additional Albumentations parameters.

        Returns:
            Rotated mask.
        """
        if angle == 0:
            return mask
        return rotate_image(
            mask,
            angle,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=self.mask_value,
            expand=True,
        )

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        """Compute transform parameters from input sample data.

        Args:
            params: Current Albumentations parameters.
            data: Input sample dictionary.

        Returns:
            Parameter dictionary containing rotation angle.
        """
        mask = data.get("mask")
        if mask is None:
            return {"angle": 0}
        rect = compute_min_area_rect(mask)
        if rect is None:
            return {"angle": 0}
        return {"angle": compute_horizontal_rotation_angle(rect)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return constructor argument names for serialization."""
        return ("border_mode", "border_value", "mask_value")


class FillOutsideMaskWithMeanColor(DualTransform):
    """Fill non-mask pixels with the masked-region mean color."""

    def __init__(self, p: float = 1.0):
        """Initialize transform parameters.

        Args:
            p: Probability of applying the transform.
        """
        super().__init__(p=p)

    def apply(self, img: np.ndarray, mean_color: np.ndarray | None = None, **params) -> np.ndarray:
        """Fill non-mask pixels by masked-region mean color.

        Args:
            img: Input image.
            mean_color: Mean color computed from mask region.
            **params: Additional Albumentations parameters.

        Returns:
            Color-filled image.
        """
        if mean_color is None:
            return img
        if img.ndim == 2:
            return np.where(params["mask"] == 255, img, mean_color).astype(img.dtype)
        mean_color = mean_color.reshape(1, 1, -1)
        filled = np.where(params["mask"][..., None] == 255, img, mean_color)
        return filled.astype(img.dtype)

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """Return mask unchanged.

        Args:
            mask: Input mask.
            **params: Additional Albumentations parameters.

        Returns:
            Unchanged mask.
        """
        return mask

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        """Compute transform parameters from input sample data.

        Args:
            params: Current Albumentations parameters.
            data: Input sample dictionary.

        Returns:
            Parameter dictionary containing mean color and mask.
        """
        mask = data.get("mask")
        if mask is None:
            return {"mean_color": None}
        masked = mask == 255
        if not np.any(masked):
            return {"mean_color": None}
        img = data.get("image")
        if img is None:
            return {"mean_color": None}
        if img.ndim == 2:
            mean_color = img[masked].mean()
            return {"mean_color": np.array(mean_color, dtype=np.float32), "mask": mask}
        mean_color = img[masked].mean(axis=0)
        return {"mean_color": mean_color.astype(np.float32), "mask": mask}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return constructor argument names for serialization."""
        return ()


class ShadesOfGray(ImageOnlyTransform):
    """Apply shades-of-gray color constancy with a sampled p-norm."""

    def __init__(self, p_norm_range: tuple[float, float] = (4.0, 8.0), p: float = 1.0):
        """Initialize transform parameters.

        Args:
            p_norm_range: Range for random p-norm sampling.
            p: Probability of applying the transform.
        """
        super().__init__(p=p)
        self.p_norm_range = p_norm_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply shades-of-gray color constancy.

        Args:
            img: Input image.
            **params: Additional Albumentations parameters.

        Returns:
            Color-adjusted image.
        """
        p_min, p_max = self.p_norm_range
        p_norm = float(np.random.uniform(p_min, p_max))
        return apply_shades_of_gray(img, p_norm)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return constructor argument names for serialization."""
        return ("p_norm_range",)
