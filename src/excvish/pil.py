"""PIL-based image utilities."""

import hashlib
import itertools
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
DEFAULT_TILE_SIZE = (256, 256)


@dataclass(frozen=True)
class TileItem:
    """Describe one image path and optional label for tile rendering."""

    path: Path
    label: str | None = None


def concat_image_on_grid(images: list[Image.Image], columns: int) -> Image.Image:
    """Concatenate multiple images into a grid.

    Args:
        images: Images to concatenate. All images must have the same size.
        columns: Number of columns in the grid.

    Returns:
        The concatenated grid image.
    """
    assert all(images[0].size == i.size for i in images[1:])

    w, h = images[0].size
    rows = len(images) // columns + 1
    grid = Image.new("RGB", size=(columns * w, rows * h), color="white")
    for i, image in enumerate(images):
        grid.paste(image, box=(i % columns * w, i // columns * h))

    return grid


def resize_and_pad(
    image: Image.Image,
    target_width: int,
    target_height: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Resize an image to fit target dimensions while preserving aspect ratio.

    Args:
        image (PIL.Image.Image): Input image.
        target_width (int): Target width.
        target_height (int): Target height.
        color (tuple): RGB color for padding. Default is white (255, 255, 255).

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    # Original dimensions
    original_width, original_height = image.size

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Determine new size based on aspect ratio
    if target_aspect_ratio > aspect_ratio:
        # Match height, adjust width
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        # Match width, adjust height
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with the target size and the specified background color
    new_image = Image.new("RGB", (target_width, target_height), color)

    # Calculate padding to center the resized image
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2

    # Paste resized image onto the new image
    new_image.paste(resized_image, (pad_left, pad_top))

    return new_image


def text_image(
    text: str,
    width: int = 300,
    height: int = 200,
    background_color: str = "white",
    text_color: str = "black",
    font_size: int = 40,
    font_name: str | None = "ヒラギノ丸ゴ ProN W4.ttc",
) -> Image.Image:
    """Create an image with text centered in the middle.

    Args:
        text: The text to draw.
        width: Width of the image.
        height: Height of the image.
        background_color: Background color of the image.
        text_color: Text color.
        font_size: Font size.
        font_name: Path to the font file. If ``None``, default PIL font is used.

    Returns:
        The generated image.
    """
    image = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(image)

    if font_name is None:
        font = ImageFont.load_default(size=font_size)
    else:
        font = ImageFont.truetype(font_name, size=font_size)

    # Calculate text position
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2

    # Draw the text.
    draw.text((text_x, text_y), text, fill=text_color, font=font, size=font_size)

    return image


def extract_rgb_alpha(
    image: Image.Image, size: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract RGB pixels and optional alpha channel from a PIL image.

    Args:
        image: Input PIL image.
        size: Optional resize target as ``(width, height)``.

    Returns:
        A tuple of ``(rgb, alpha)`` where ``rgb`` is an ``(H, W, 3)`` array and
        ``alpha`` is either an ``(H, W)`` array or ``None``.
    """
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    has_transparency = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_transparency:
        rgba = np.array(image.convert("RGBA"))
        return rgba[..., :3], rgba[..., 3]
    rgb = np.array(image.convert("RGB"))
    return rgb, None


def get_image_files(directory: Path) -> list[Path]:
    """Collect supported image files below a directory.

    Args:
        directory: Root directory to scan recursively.

    Returns:
        Paths whose suffix matches ``IMAGE_EXTENSIONS``.
    """
    return [
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def composite_alpha_to_background(
    img: Image.Image,
    background_color: tuple[int, int, int],
) -> Image.Image:
    """Flatten transparent PIL images onto a fixed RGB background.

    Args:
        img: Input PIL image.
        background_color: RGB background color.

    Returns:
        RGB image with transparency composited when present.
    """
    if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, background_color)
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def apply_random_background(
    pil_img: Image.Image,
    rng: random.Random | None = None,
) -> Image.Image:
    """Composite an RGBA PIL image over a random RGB background.

    Args:
        pil_img: Input PIL image.
        rng: Optional random generator for deterministic background colors.

    Returns:
        RGB PIL image after alpha compositing.
    """
    if pil_img.mode != "RGBA":
        return pil_img.convert("RGB")
    rand = rng or random
    bg_color = tuple(rand.randint(0, 255) for _ in range(3))
    background = Image.new("RGBA", pil_img.size, bg_color + (255,))
    return Image.alpha_composite(background, pil_img).convert("RGB")


def _label_color(label: str) -> tuple[int, int, int]:
    """Pick a deterministic label color from a fixed palette."""
    palette = [
        (33, 158, 188),
        (2, 48, 71),
        (251, 133, 0),
        (250, 163, 7),
        (33, 37, 41),
        (52, 78, 65),
        (106, 153, 78),
        (188, 71, 73),
    ]
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    return palette[digest[0] % len(palette)]


def _contrast_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose a readable text color based on background luminance."""
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def annotate_tile_with_label(img: Image.Image, label: str | None) -> Image.Image:
    """Draw a bottom label banner when a label is available.

    Args:
        img: Tile image to annotate in place.
        label: Optional label text.

    Returns:
        The input image with a label banner when ``label`` is truthy.
    """
    if not label:
        return img
    color = _label_color(label)
    text_color = _contrast_color(color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    padding = 4
    text = str(label)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_height = text_bbox[3] - text_bbox[1]
    banner_height = text_height + padding * 2
    y0 = img.height - banner_height
    draw.rectangle([0, y0, img.width, img.height], fill=color)
    draw.text((padding, y0 + padding), text, font=font, fill=text_color)
    return img


def _prepare_tile_image(
    img: Image.Image,
    tile_size: tuple[int, int],
    background_color: tuple[int, int, int] | None,
) -> Image.Image:
    """Resize one source image and optionally flatten transparency."""
    if background_color is None:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img.resize(tile_size, Image.Resampling.NEAREST)
    rgb = composite_alpha_to_background(img, background_color)
    return rgb.resize(tile_size, Image.Resampling.NEAREST)


def build_tiles(
    items: Iterable[TileItem],
    rows: int,
    cols: int,
    tile_size: tuple[int, int] = DEFAULT_TILE_SIZE,
    background_color: tuple[int, int, int] | None = None,
    show_label: bool = False,
) -> Image.Image:
    """Assemble image files into a fixed-size tile grid.

    Args:
        items: Image path and label records.
        rows: Number of grid rows.
        cols: Number of grid columns.
        tile_size: Output size for each tile.
        background_color: Optional fixed RGB background for alpha compositing.
        show_label: Whether to draw each item's label banner.

    Returns:
        PIL image containing up to ``rows * cols`` tiles.
    """
    mode = "RGB" if background_color is not None else "RGBA"
    canvas_bg = background_color if background_color is not None else (0, 0, 0, 0)
    canvas = Image.new(mode, (cols * tile_size[0], rows * tile_size[1]), canvas_bg)
    limit = rows * cols
    for idx, item in enumerate(itertools.islice(items, limit)):
        with Image.open(item.path) as img:
            tile = _prepare_tile_image(img, tile_size, background_color)
        if show_label:
            tile = annotate_tile_with_label(tile, item.label)
        row = idx // cols
        col = idx % cols
        offset = (col * tile_size[0], row * tile_size[1])
        if tile.mode == "RGBA" and canvas.mode == "RGBA":
            canvas.paste(tile, offset, tile)
        else:
            canvas.paste(tile, offset)
    return canvas
