"""Shared data types for excvish."""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class BBox:
    """A bounding box with coordinates in 2D space.

    This class represents a bounding box defined by its top-left (x1, y1) and
    bottom-right (x2, y2) coordinates. The class provides methods to manipulate
    and convert the bounding box.

    Attributes:
        x1: The x-coordinate of the top-left corner.
        y1: The y-coordinate of the top-left corner.
        x2: The x-coordinate of the bottom-right corner.
        y2: The y-coordinate of the bottom-right corner.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        """Calculates the area of the bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @classmethod
    def from_xyxy_array(cls, arr: list[int]) -> Self:
        """Create a bounding box from XYXY coordinates.

        Args:
            arr: A list containing [x1, y1, x2, y2] coordinates.

        Returns:
            A new BBox instance.
        """
        return cls(x1=arr[0], y1=arr[1], x2=arr[2], y2=arr[3])

    @classmethod
    def sort_by_x(cls, bboxes: list[Self]) -> list[Self]:
        """Sort bounding boxes by left x-coordinate.

        Args:
            bboxes: A list of BBox instances to sort.

        Returns:
            A new list of BBoxes sorted by their x1 values in ascending order.
        """
        return sorted(bboxes, key=lambda bbox: bbox.x1)

    def offset_y(self, y: int) -> "BBox":
        """Apply a vertical offset to both y-coordinates.

        Args:
            y: The vertical offset to apply.

        Returns:
            A new BBox with the offset applied.
        """
        return BBox(x1=self.x1, y1=self.y1 - y, x2=self.x2, y2=self.y2 - y)

    def shift_y1(self, dy: int) -> "BBox":
        """Shift only the top y-coordinate by ``dy``.

        Args:
            dy: The vertical offset to apply to y1.

        Returns:
            A new BBox with the top edge shifted.
        """
        return BBox(x1=self.x1, y1=self.y1 - dy, x2=self.x2, y2=self.y2)

    def shift_x1(self, dx: int) -> "BBox":
        """Shift only the left x-coordinate by ``dx``.

        Args:
            dx: The horizontal offset to apply to x1.

        Returns:
            A new BBox with the left edge shifted.
        """
        return BBox(x1=self.x1 + dx, y1=self.y1, x2=self.x2, y2=self.y2)

    def shift_x2(self, dx: int) -> "BBox":
        """Shift only the right x-coordinate by ``dx``.

        Args:
            dx: The horizontal offset to apply to x2.

        Returns:
            A new BBox with the right edge shifted.
        """
        return BBox(x1=self.x1, y1=self.y1, x2=self.x2 + dx, y2=self.y2)

    def as_xyxy_array(self) -> list[int]:
        """Convert the bounding box to an ``[x1, y1, x2, y2]`` list.

        Returns:
            A list of integers [x1, y1, x2, y2].
        """
        return [int(c) for c in [self.x1, self.y1, self.x2, self.y2]]

    def scale(self, a: float) -> "BBox":
        """Scale box area by factor ``a`` around its center.

        Args:
            a: Area scaling factor. For example, ``2.0`` doubles area and
                ``0.5`` halves area.

        Returns:
            A scaled bounding box.
        """
        # Calculate the center point of the current bbox
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2

        # Calculate current width and height
        width = self.x2 - self.x1
        height = self.y2 - self.y1

        # Scale width and height by sqrt(a) to scale area by a
        scale_factor = a**0.5
        new_width = width * scale_factor
        new_height = height * scale_factor

        # Calculate new coordinates based on the center point
        new_x1 = int(center_x - new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)

        # Return new BBox with scaled dimensions
        return BBox(x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2)
