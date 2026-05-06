"""Tests for PIL image utilities."""

from pathlib import Path

import numpy as np
from PIL import Image

from excvish.pil import (
    TileItem,
    annotate_tile_with_label,
    build_tiles,
    composite_alpha_to_background,
    get_image_files,
)


def test_get_image_files_filters_supported_extensions(tmp_path: Path) -> None:
    """Recursive image discovery should include only supported image suffixes."""
    image_path = tmp_path / "nested" / "sample.PNG"
    text_path = tmp_path / "nested" / "sample.txt"
    image_path.parent.mkdir()
    image_path.write_bytes(b"not actually decoded")
    text_path.write_text("skip", encoding="utf-8")

    assert get_image_files(tmp_path) == [image_path]


def test_composite_alpha_to_background_flattens_transparency() -> None:
    """Transparent pixels should be replaced with the requested background color."""
    rgba = np.array([[[255, 0, 0, 255], [255, 0, 0, 0]]], dtype=np.uint8)
    image = Image.fromarray(rgba)

    flattened = composite_alpha_to_background(image, background_color=(10, 20, 30))

    np.testing.assert_array_equal(
        np.asarray(flattened),
        np.array([[[255, 0, 0], [10, 20, 30]]], dtype=np.uint8),
    )


def test_annotate_tile_with_label_draws_banner() -> None:
    """A label should modify the bottom of the tile while preserving image dimensions."""
    image = Image.new("RGB", (32, 24), (255, 255, 255))

    annotated = annotate_tile_with_label(image.copy(), "skipjack")

    assert annotated.size == image.size
    assert not np.array_equal(np.asarray(annotated), np.asarray(image))


def test_build_tiles_assembles_limited_grid(tmp_path: Path) -> None:
    """Tile assembly should paste up to the requested grid capacity."""
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(first)
    Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(second)

    grid = build_tiles(
        [TileItem(first, "a"), TileItem(second, "b")],
        rows=1,
        cols=1,
        tile_size=(4, 4),
        background_color=(0, 0, 0),
    )

    assert grid.mode == "RGB"
    assert grid.size == (4, 4)
    np.testing.assert_array_equal(np.asarray(grid)[0, 0], np.array([255, 0, 0], dtype=np.uint8))
