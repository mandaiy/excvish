"""Core public utilities for the excvish package."""

from typing import TypeVar

from .types import BBox

T = TypeVar("T", bound=int | float)


def list_from_nth_elems(l: list, n: int) -> list:
    """Extract the nth element from each item in a list.

    Args:
        l: Input list containing indexable items.
        n: Zero-based index to extract from each item.

    Returns:
        A list containing the extracted elements.
    """
    return [e[n] for e in l]


def clip(value: T, min_value: T, max_value: T) -> T:
    """Clip a value to the inclusive range [min_value, max_value].

    Args:
        value: The value to clamp.
        min_value: Lower bound.
        max_value: Upper bound.

    Returns:
        The clamped value.
    """
    return max(min(value, max_value), min_value)


__all__ = ["BBox", "list_from_nth_elems", "clip"]
