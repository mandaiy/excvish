from typing import TypeVar

from .types import BBox

T = TypeVar("T", bound=int | float)


def list_from_nth_elems(l: list, n: int) -> list:
    return [e[n] for e in l]


def clip(value: T, min_value: T, max_value: T) -> T:
    """Clips a value to be within the specified range."""
    return max(min(value, max_value), min_value)


__all__ = ["BBox", "list_from_nth_elems", "clip"]
