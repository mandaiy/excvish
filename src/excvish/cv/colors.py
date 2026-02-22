"""Color processing utilities for NumPy/OpenCV images."""

import cv2
import numpy as np


def compute_luminance(image: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB channels.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.

    Returns:
        Luminance image as ``float32``.
    """
    rgb = image.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def apply_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to 3-channel grayscale.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.

    Returns:
        Grayscale image with shape ``(H, W, 3)``.
    """
    y = compute_luminance(image)
    gray = np.clip(y, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def apply_gray_world(image: np.ndarray) -> np.ndarray:
    """Apply gray-world white balance.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.

    Returns:
        White-balanced RGB image as ``uint8``.
    """
    img = image.astype(np.float32)
    mean = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(mean.mean())
    scale = gray / mean
    balanced = img * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


def quantize_with_dither(img01: np.ndarray, seed: int = 0) -> np.ndarray:
    """Quantize a normalized image with random dithering.

    Args:
        img01: Image values normalized to ``[0, 1]``.
        seed: Random seed for reproducible dithering.

    Returns:
        Quantized ``uint8`` image.
    """
    rng = np.random.default_rng(seed)
    dither = rng.uniform(-0.5 / 255.0, 0.5 / 255.0, size=img01.shape).astype(np.float32)
    img01_d = np.clip(img01 + dither, 0.0, 1.0)
    return (img01_d * 255.0 + 0.5).astype(np.uint8)


def apply_ordered_dither(image: np.ndarray) -> np.ndarray:
    """Apply ordered Bayer dithering.

    Args:
        image: Input image array.

    Returns:
        Dithered image array in floating-point domain.
    """
    bayer = np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    height, width = image.shape[:2]
    reps = ((height + 3) // 4, (width + 3) // 4)
    tiled = np.tile(bayer, reps)[:height, :width]
    dither = tiled / 16.0 - 0.5
    return image + dither[..., None]


def apply_shades_of_gray(image: np.ndarray, p_norm: float) -> np.ndarray:
    """Apply shades-of-gray color constancy.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.
        p_norm: Minkowski norm value.

    Returns:
        Color-balanced ``uint8`` image.
    """
    img = image.astype(np.float32)
    powered = np.power(img, p_norm)
    mean = np.power(powered.reshape(-1, 3).mean(axis=0) + 1e-6, 1.0 / p_norm)
    gray = float(mean.mean())
    scale = gray / mean
    balanced = img * scale
    return quantize_with_dither(balanced / 255.0)


def apply_retinex(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply single-scale Retinex on luminance.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.
        sigma: Gaussian sigma for the illumination estimate.

    Returns:
        Retinex-adjusted ``uint8`` image.
    """
    rgb = image.astype(np.float32)
    y = compute_luminance(image)
    y_img = y.astype(np.float32)
    # OpenCV Gaussian blur keeps this utility NumPy/OpenCV-only.
    blurred = cv2.GaussianBlur(y_img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    y_safe = y + 1.0
    b_safe = blurred + 1.0
    retinex = np.log(y_safe) - np.log(b_safe)
    ret_min = float(retinex.min())
    ret_max = float(retinex.max())
    ret_scaled = (retinex - ret_min) / (ret_max - ret_min + 1e-6)
    y_new = ret_scaled * 255.0
    scale = y_new / (y + 1e-6)
    out = rgb * scale[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_color_histograms(
    image: np.ndarray, alpha: np.ndarray | None, bins: int = 256
) -> list[np.ndarray]:
    """Compute per-channel RGB histograms.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.
        alpha: Optional alpha mask. Pixels with alpha ``<= 0`` are ignored.
        bins: Histogram bins per channel.

    Returns:
        List of three histograms for R, G, and B channels.
    """
    if alpha is None:
        mask = np.ones(image.shape[:2], dtype=bool)
    else:
        mask = alpha > 0
    if not np.any(mask):
        return [np.zeros(bins, dtype=np.int64) for _ in range(3)]
    histograms = []
    for idx in range(3):
        values = image[..., idx][mask]
        hist, _ = np.histogram(values, bins=bins, range=(0, 255))
        histograms.append(hist)
    return histograms


def compute_luminance_histogram(
    image: np.ndarray, alpha: np.ndarray | None, bins: int = 256
) -> np.ndarray:
    """Compute luminance histogram with optional alpha masking.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.
        alpha: Optional alpha mask. Pixels with alpha ``<= 0`` are ignored.
        bins: Number of histogram bins.

    Returns:
        Luminance histogram.
    """
    if alpha is None:
        mask = np.ones(image.shape[:2], dtype=bool)
    else:
        mask = alpha > 0
    if not np.any(mask):
        return np.zeros(bins, dtype=np.int64)
    y = compute_luminance(image)
    values = y[mask]
    hist, _ = np.histogram(values, bins=bins, range=(0, 255))
    return hist


def normalize_histogram(hist: np.ndarray) -> np.ndarray:
    """Normalize a histogram to unit sum.

    Args:
        hist: Input histogram values.

    Returns:
        Normalized histogram as ``float32``.
    """
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist, dtype=np.float32)
    return hist.astype(np.float32) / float(total)


def fill_transparent_with_mean(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill pixels outside a mask with masked-region mean color.

    Args:
        rgb: RGB image array with shape ``(H, W, 3)``.
        mask: Binary mask where non-zero means valid region.

    Returns:
        RGB image where masked-out pixels are replaced by mean color.
    """
    alpha_mask = mask > 0
    if not np.any(alpha_mask):
        return rgb
    mean_color = rgb[alpha_mask].mean(axis=0)
    filled = np.where(alpha_mask[..., None], rgb, mean_color.reshape(1, 1, 3))
    return np.clip(filled, 0, 255).astype(np.uint8)


def apply_random_background_np(
    np_img: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Composite an RGBA NumPy image over a random background color.

    Args:
        np_img: Input image array. When RGBA, expected shape is ``(H, W, 4)``.
        rng: Optional NumPy random generator.

    Returns:
        RGB-like image array with dtype preserved when possible.
    """
    if np_img.ndim != 3 or np_img.shape[2] != 4:
        return np_img[..., :3] if np_img.ndim == 3 and np_img.shape[2] > 3 else np_img

    alpha = np_img[..., 3:4]
    generator = rng or np.random.default_rng()
    if np.issubdtype(np_img.dtype, np.floating) and np.nanmax(np_img[..., :3]) <= 1.0:
        bg_color = generator.random(3, dtype=np.float32)
        fg = np_img[..., :3].astype(np.float32)
        out = fg * alpha + bg_color * (1.0 - alpha)
        return out.astype(np_img.dtype)

    bg_color = generator.integers(0, 256, size=3, dtype=np.int32).astype(np.float32)
    alpha_f = alpha.astype(np.float32) / 255.0
    fg = np_img[..., :3].astype(np.float32)
    out = fg * alpha_f + bg_color * (1.0 - alpha_f)
    if np.issubdtype(np_img.dtype, np.integer):
        out = np.rint(out)
    return out.astype(np_img.dtype)
