"""Prediction masks → submission CSV, with the checks that matter on Kaggle's side (disjointness, ids, decodability)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import H, W
from .rle import assert_disjoint, decode_rle, encode_mask

COLUMNS = ["filament_id", "segmentation_rle"]


def build_submission(preds: dict) -> pd.DataFrame:
    """{stem: [bool mask, ...]} → DataFrame. Empty masks are dropped (an empty row is a guaranteed FP)."""
    rows = []
    for stem in sorted(preds):
        masks = [m for m in preds[stem] if m.any()]
        assert_disjoint(masks, stem)
        for k, m in enumerate(masks, 1):
            rows.append((f"{stem}_{k}", encode_mask(m)))
    return pd.DataFrame(rows, columns=COLUMNS)


def validate_submission(df: pd.DataFrame, test_stems: list[str], check_overlap: bool = True) -> dict:
    assert list(df.columns) == COLUMNS, f"columns must be {COLUMNS}"
    assert df["filament_id"].is_unique, "filament_id must be unique"
    assert df["segmentation_rle"].notna().all() and (df["segmentation_rle"].str.len() > 0).all(), "empty RLE cell"
    assert not df["segmentation_rle"].str.contains('["\',]').any(), "RLE must not contain quotes/commas"
    stems = df["filament_id"].str.split("_", n=1).str[0]
    unknown = sorted(set(stems) - set(test_stems))
    assert not unknown, f"ids not matching test stems: {unknown[:5]}"
    report = dict(n_rows=len(df), n_images_with_rows=int(stems.nunique()),
                  missing_images=sorted(set(test_stems) - set(stems)), overlap_checked=check_overlap)
    areas = []
    for stem, g in df.groupby(stems, sort=False):
        acc = np.zeros((H, W), dtype=np.uint8)
        for c in g["segmentation_rle"]:
            m = decode_rle(c)
            areas.append(int(m.sum()))
            if check_overlap:
                acc += m
        if check_overlap and acc.max() > 1:
            raise ValueError(f"overlapping masks in {stem}")
    report.update(min_area=min(areas) if areas else 0, median_area=float(np.median(areas)) if areas else 0,
                  rows_per_image=round(len(df) / max(report["n_images_with_rows"], 1), 2))
    return report


def write_submission(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)
    head = open(path).read(200)
    assert '"' not in head, "unexpected quoting in CSV"
