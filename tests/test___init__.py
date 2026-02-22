"""Tests for core utility helpers."""

import pytest

from excvish import clip, list_from_nth_elems


def test_list_from_nth_elems_extracts_target_index() -> None:
    """`list_from_nth_elems` should collect the nth value from each element."""
    values = [(1, "a"), (2, "b"), (3, "c")]

    result = list_from_nth_elems(values, 1)

    assert result == ["a", "b", "c"]


def test_list_from_nth_elems_raises_for_invalid_index() -> None:
    """Out-of-range index access should raise IndexError."""
    values = [[10], [20], [30]]

    with pytest.raises(IndexError):
        list_from_nth_elems(values, 2)


@pytest.mark.parametrize(
    ("value", "min_value", "max_value", "expected"),
    [
        (-1, 0, 10, 0),
        (5, 0, 10, 5),
        (15, 0, 10, 10),
        (1.25, 0.0, 1.0, 1.0),
    ],
)
def test_clip_clamps_to_inclusive_range(
    value: int | float,
    min_value: int | float,
    max_value: int | float,
    expected: int | float,
) -> None:
    """`clip` should clamp values into [min_value, max_value]."""
    assert clip(value, min_value, max_value) == expected
