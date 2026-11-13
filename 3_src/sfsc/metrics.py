"""Panoptic Quality scorer with the official self-evaluation semantics (verified equal, E-009):

* one row per annotator-image (GT id '<annotator>-<stem>_<k>'); the single prediction set of <stem> is scored against
  every annotator set of that image; TP/FP/FN and ΣIoU are pooled globally (no per-image averaging);
* a pair is TP iff IoU > 0.5 (strict); FP = prediction with no TP partner; FN = GT with no TP partner;
* images with GT but no prediction rows → all GT are FN; prediction rows for images without GT are ignored;
* IoU/Dice distributions use pairs with IoU > 0; one-to-many / many-to-one degrees use IoU > 0 as well.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from . import H, W
from .rle import encode_mask

IOU_THR = 0.5


def _rle(c):
    return {"size": [H, W], "counts": c if isinstance(c, bytes) else c.encode("ascii")}


def iou_matrix(pred_counts: list, gt_counts: list) -> np.ndarray:
    """(n_pred, n_gt) IoU matrix via pycocotools (identical values to the official torch implementation)."""
    if not pred_counts or not gt_counts:
        return np.zeros((len(pred_counts), len(gt_counts)))
    P = [_rle(c) for c in pred_counts]
    G = [_rle(c) for c in gt_counts]
    return np.asarray(mask_util.iou(P, G, [0] * len(G)), dtype=float).reshape(len(P), len(G))


@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    sum_iou: float
    rows: pd.DataFrame                      # per annotator-image: image, annotator, n_gt, n_pred, tp, fp, fn, sum_iou
    ious_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))   # all IoU > 0 pairs
    gt_degree: np.ndarray = field(default_factory=lambda: np.zeros(0, int))    # per GT: #preds with IoU>0 (1:n)
    pred_degree: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # per pred: #GT with IoU>0 (n:1)

    def summary(self) -> dict:
        return dict(PQ=round(self.pq, 4), SQ=round(self.sq, 4), RQ=round(self.rq, 4), TP=self.tp, FP=self.fp, FN=self.fn,
                    gt_1toN=int((self.gt_degree >= 2).sum()), pred_Nto1=int((self.pred_degree >= 2).sum()),
                    n_rows=len(self.rows))

    def subset(self, mask) -> dict:
        """PQ/SQ/RQ on a subset of annotator-image rows (e.g. single- vs multi-annotated files)."""
        r = self.rows[mask]
        tp, fp, fn, s = r.tp.sum(), r.fp.sum(), r.fn.sum(), r.sum_iou.sum()
        den = tp + 0.5 * fp + 0.5 * fn
        return dict(PQ=s / den if den else 0.0, SQ=s / tp if tp else 0.0, RQ=tp / den if den else 0.0,
                    TP=int(tp), FP=int(fp), FN=int(fn), n_rows=len(r))


def pq_from_counts(tp, fp, fn, sum_iou) -> tuple[float, float, float]:
    den = tp + 0.5 * fp + 0.5 * fn
    return (sum_iou / den if den else 0.0, sum_iou / tp if tp else 0.0, tp / den if den else 0.0)


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame, iou_thr: float = IOU_THR, distributions: bool = True) -> PQResult:
    gt_key = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_key = pred_df["filament_id"].str.split("_", n=1).str[0] if len(pred_df) else pd.Series([], dtype=str)
    gt_groups = {k: g["segmentation_rle"].tolist() for k, g in gt_df.groupby(gt_key, sort=False)}
    pred_groups = {k: g["segmentation_rle"].tolist() for k, g in pred_df.groupby(pred_key, sort=False)} if len(pred_df) else {}

    TP = FP = FN = 0
    S = 0.0
    rows, ious_pos, gt_deg, pred_deg = [], [], [], []
    for ai in gt_key.unique():
        image = ai.split("-", 1)[1]
        G = gt_groups[ai]
        P = pred_groups.get(image, [])
        n_gt, n_pred = len(G), len(P)
        if n_gt == 0:
            FP += n_pred
            rows.append((image, ai.split("-", 1)[0], 0, n_pred, 0, n_pred, 0, 0.0))
            continue
        if n_pred == 0:
            FN += n_gt
            rows.append((image, ai.split("-", 1)[0], n_gt, 0, 0, 0, n_gt, 0.0))
            if distributions:
                gt_deg.extend([0] * n_gt)
            continue
        iou = iou_matrix(P, G)                 # (n_pred, n_gt)
        hit = iou > iou_thr
        tp = int(hit.sum())
        s = float(iou[hit].sum())
        fp = int((hit.sum(axis=1) == 0).sum())
        fn = int((hit.sum(axis=0) == 0).sum())
        TP += tp; FP += fp; FN += fn; S += s
        rows.append((image, ai.split("-", 1)[0], n_gt, n_pred, tp, fp, fn, s))
        if distributions:
            pos = iou > 0
            ious_pos.extend(iou[pos].tolist())
            gt_deg.extend(pos.sum(axis=0).tolist())
            pred_deg.extend(pos.sum(axis=1).tolist())
    pq, sq, rq = pq_from_counts(TP, FP, FN, S)
    rows_df = pd.DataFrame(rows, columns=["image", "annotator", "n_gt", "n_pred", "tp", "fp", "fn", "sum_iou"])
    return PQResult(pq, sq, rq, TP, FP, FN, S, rows_df, np.asarray(ious_pos), np.asarray(gt_deg, int), np.asarray(pred_deg, int))


def masks_to_pred_df(preds: dict) -> pd.DataFrame:
    """{stem: [bool mask, ...]} → prediction dataframe (filament_id '<stem>_<k>')."""
    rows = []
    for stem in sorted(preds):
        for k, m in enumerate(preds[stem], 1):
            rows.append((f"{stem}_{k}", encode_mask(m)))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
