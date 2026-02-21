"""Tests for the BBox data type."""

from excvish.types import BBox


def test_bbox_area() -> None:
    """Area should be computed from width and height."""
    bbox = BBox(1, 2, 11, 12)

    assert bbox.area == 100


def test_bbox_from_xyxy_array() -> None:
    """Factory should map XYXY array to coordinates."""
    bbox = BBox.from_xyxy_array([3, 4, 13, 24])

    assert bbox == BBox(3, 4, 13, 24)


def test_bbox_sort_by_x() -> None:
    """Sort helper should order by x1 ascending."""
    bboxes = [BBox(50, 0, 60, 10), BBox(10, 0, 20, 10), BBox(30, 0, 40, 10)]

    sorted_bboxes = BBox.sort_by_x(bboxes)

    assert [b.x1 for b in sorted_bboxes] == [10, 30, 50]


def test_bbox_offset_and_shift_helpers() -> None:
    """Offset and shift helpers should update only intended coordinates."""
    bbox = BBox(10, 20, 30, 40)

    assert bbox.offset_y(5) == BBox(10, 15, 30, 35)
    assert bbox.shift_y1(3) == BBox(10, 17, 30, 40)
    assert bbox.shift_x1(4) == BBox(14, 20, 30, 40)
    assert bbox.shift_x2(7) == BBox(10, 20, 37, 40)


def test_bbox_as_xyxy_array() -> None:
    """Array conversion should preserve coordinate order."""
    bbox = BBox(2, 4, 6, 8)

    assert bbox.as_xyxy_array() == [2, 4, 6, 8]


def test_bbox_scale_scales_area_around_center() -> None:
    """Scaling by factor `a` should scale area by `a` for symmetric cases."""
    bbox = BBox(0, 0, 10, 20)

    scaled = bbox.scale(4.0)

    assert scaled == BBox(-5, -10, 15, 30)
    assert scaled.area == bbox.area * 4
