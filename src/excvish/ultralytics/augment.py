"""Ultralytics augmentation adapters shared by excvish training pipelines.

The helpers in this module bridge Albumentations transforms with Ultralytics'
``Instances`` containers so we can reuse rich augmentation policies while
preserving alignment between images and geometric annotations. They also expose
utility transforms that resize inputs without breaking detection targets.
"""

import random
from typing import Any, Literal

import cv2
import numpy as np
from ultralytics.utils.instance import Instances

import excvish.albumentations as exA


class Albumentations:
    """Wrap an Albumentations transform for Ultralytics dataloaders.

    This adapter mirrors Ultralytics' expected callable signature—taking a
    labels dictionary with ``img``/``instances``/``cls``—and forwards it through
    an Albumentations transform while keeping ``Instances`` in sync with the
    mutated tensors.

    Args:
        transform: Albumentations transform or composition executed on the data
            sample. It must accept the same targets exposed by the Ultralytics
            dataloader (e.g., image, bboxes, keypoints).
        p: Probability with which the wrapped transform is applied to each
            sample. A lower value increases the chance of returning untouched
            labels.
    """

    def __init__(self, transform, p: float = 1.0):
        self.p = p
        self.transform = transform
        self.contains_spatial = exA.has_dual_transform(self.transform)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.transform is None or random.random() > self.p:
            return labels

        im = labels["img"]
        if im.shape[2] != 3:  # Only apply Albumentation on 3-channel images
            return labels

        if not self.contains_spatial:
            labels["img"] = self.transform(image=im)["image"]  # transformed
            return labels

        kwargs = {}

        W, H = im.shape[1], im.shape[0]
        instances: Instances = labels["instances"]
        existing_cls = labels.get("cls")
        cls_dtype = np.asarray(existing_cls).dtype if existing_cls is not None else None

        # keypoints
        if instances.keypoints is not None:
            kpts = instances.keypoints
            _, k, dim = kpts.shape
            # Flatten to (N*K, 2) for xy coordinates
            xs = np.clip(kpts[..., 0], 0, W - 1)
            ys = np.clip(kpts[..., 1], 0, H - 1)
            keypoints = np.stack([xs, ys], axis=-1).reshape(-1, dim - 1)
            kwargs["keypoints"] = keypoints
        else:
            keypoints = None
            k, dim = None, None

        instances.convert_bbox("xywh")
        instances.normalize(W, H)

        kwargs["bboxes"] = instances.bboxes
        kwargs["class_labels"] = labels["cls"]
        kwargs["image"] = im

        # Apply transform
        new = self.transform(**kwargs)

        if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
            labels["img"] = new["image"]
            cls = (
                np.asarray(new["class_labels"], dtype=cls_dtype)
                if cls_dtype is not None
                else np.asarray(new["class_labels"])
            )
            labels["cls"] = cls
            bboxes = np.array(new["bboxes"], dtype=np.float32)

            # Update keypoints if they were transformed
            if "keypoints" in new:
                assert k is not None and dim is not None
                kpts = new["keypoints"] / np.array([W, H])  # Normalize to [0, 1]
                kpts = kpts.reshape(-1, k, dim - 1)
                original_visibility = instances.keypoints[..., 2].reshape(-1, k, 1)

                keypoints = np.concatenate([kpts, original_visibility], axis=-1).astype(np.float32)
            else:
                keypoints = None

            instances.update(bboxes=bboxes, keypoints=keypoints)
            instances.cls = cls

        return labels


class AspectPreservingResize:
    """Resize while preserving aspect ratio and updating Ultralytics instances.

    The transform pads resized images to a fixed canvas size and updates the
    associated ``Instances`` container (normalization, padding, bbox format) so
    downstream Ultralytics components continue to receive consistent targets.

    Args:
        height: Height in pixels of the padded output canvas.
        width: Width in pixels of the padded output canvas.
    """

    def __init__(self, height: int, width: int) -> None:
        self.target_h = height
        self.target_w = width
        self.target_aspect_ratio = width / height

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        assert "img" in data, "img not found in data"
        assert "instances" in data, "instances not found in data"

        img: np.ndarray = data["img"]
        instances: Instances = data["instances"]

        orig_h, orig_w = img.shape[:2]
        assert orig_h > 0 and orig_w > 0, f"Non-positive image size found: ({orig_w}, {orig_h})"

        orig_aspect_ratio = orig_w / orig_h

        if orig_aspect_ratio > self.target_aspect_ratio:
            new_w = self.target_w
            new_h = int(round(new_w / orig_aspect_ratio))
        else:
            new_h = self.target_h
            new_w = int(round(orig_aspect_ratio * new_h))

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w
        padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
        img = np.pad(img, padding, mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))

        pad_top, _ = padding[0]
        pad_left, _ = padding[1]
        scale_w = new_w / orig_w
        scale_h = new_h / orig_h

        # Update instances
        original_format = instances._bboxes.format
        instances.convert_bbox("xyxy")

        was_normalized = instances.normalized
        if was_normalized:
            instances.denormalize(orig_w, orig_h)

        instances.scale(scale_w, scale_h)
        instances.add_padding(pad_left, pad_top)

        if was_normalized:
            instances.normalize(self.target_w, self.target_h)

        if original_format != "xyxy":
            instances.convert_bbox(original_format)

        data["img"] = img
        data["instances"] = instances

        return data
