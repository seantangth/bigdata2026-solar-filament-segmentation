"""COCO RLE helpers in the exact form the official scorer consumes: {"size":[2048,2048],"counts":<compressed str>}."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

from . import H, W


def encode_mask(mask: np.ndarray) -> str:
    """Binary (H, W) mask → compressed COCO RLE counts as an ASCII string (no size, as the competition requires)."""
    if mask.shape != (H, W):
        raise ValueError(f"mask shape {mask.shape} != {(H, W)}")
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def decode_rle(counts: str | bytes) -> np.ndarray:
    """Compressed counts → boolean (H, W) mask, exactly like the official `rles_to_layers`."""
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return mask_util.decode({"size": [H, W], "counts": counts}).astype(bool)


def rle_area(counts: str | bytes) -> int:
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return int(mask_util.area({"size": [H, W], "counts": counts}))


def assert_disjoint(masks, where: str = "") -> None:
    """Raise ValueError if any pixel belongs to more than one mask.

    Kaggle's gate rejects overlapping predictions with status ERROR and still consumes a daily slot
    (host, topic 735377, 2026-08-15). The self-evaluation notebook does NOT catch this.
    """
    if len(masks) < 2:
        return
    acc = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        acc += m.astype(np.uint8)
    if acc.max() > 1:
        raise ValueError(f"overlapping masks{(' in ' + where) if where else ''}: {int((acc > 1).sum())} px shared")
