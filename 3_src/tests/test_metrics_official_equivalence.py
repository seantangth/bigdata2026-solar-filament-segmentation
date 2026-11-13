"""sfsc.metrics.evaluate must reproduce the host's self-evaluation notebook exactly (functions copied verbatim)."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfsc.data import ann_mask, build_gt_df, load_train
from sfsc.metrics import evaluate
from sfsc.rle import encode_mask

ROOT = Path(__file__).resolve().parents[2]
JSON = ROOT / "1_data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


# ---- official functions, verbatim (docstrings removed) ----
def get_overlap_matrices(gt_layers, pred_layers):
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(union == 0, torch.tensor(0., device=gt_layers.device), intersection / union)
    dice_matrix = torch.where(union == 0, torch.tensor(0., device=gt_layers.device), 2 * intersection / (gt_areas + pred_areas))
    return iou_matrix, dice_matrix


def fp_count_hit(hit_matrix):
    return (hit_matrix.sum(dim=0) == 0).sum().item()


def fn_count_hit(hit_matrix):
    return (hit_matrix.sum(dim=1) == 0).sum().item()


def rles_to_layers(rles, height=2048, width=2048):
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def process_entry(annotator_image, gt_annotator_image_ids, pred_image_ids, gt_df, pred_df):
    image_id = annotator_image.split("-", maxsplit=1)[1]
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
    return iou_matrix, dice_matrix, len(gt_rles), len(pred_rles)


def get_overlap_df(gt_df, pred_df):
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    annotator_images = gt_annotator_image_ids.unique()
    overlap_df = pd.DataFrame({"annotator_image": annotator_images})
    results = overlap_df["annotator_image"].apply(process_entry, gt_annotator_image_ids=gt_annotator_image_ids,
                                                  pred_image_ids=pred_image_ids, gt_df=gt_df, pred_df=pred_df)
    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df["n_pred"] = zip(*results)
    return overlap_df


def get_pq_score(overlap_df):
    iou_threshold = 0.5
    tp_iou_scores, fp_count, fn_count = [], 0, 0
    for row in overlap_df.itertuples(index=False):
        iou_matrix, n_gt, n_pred = row.iou_matrix, row.n_gt, row.n_pred
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)
    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return (sum(tp_iou_scores) / denominator if denominator > 0 else 0.0), tp_count, fp_count, fn_count
# ---- end official ----


def test_equivalence(n_files: int = 20, seed: int = 1):
    ts = load_train(JSON)
    multi = [f for f in ts.files() if ts.n_annotators(f) >= 2]
    rng = random.Random(seed)
    files = rng.sample(multi, n_files)
    gt_df = build_gt_df(ts, files)
    rows = []
    for f in files:
        ids = ts.by_file[f]
        stem = f.replace(".jpeg", "")
        src = ts.by_image[ids[-1]]
        k = 0
        for a in src[:-1]:                                   # drop one, perturb the rest, add a bogus blob
            m = ann_mask(a)
            it = rng.choice([0, 1, 2])
            if it:
                m = ndimage.binary_dilation(m, iterations=it)
            k += 1
            rows.append((f"{stem}_{k}", encode_mask(m)))
        bog = np.zeros((2048, 2048), bool)
        bog[1000:1030, 1000:1060] = True
        k += 1
        rows.append((f"{stem}_{k}", encode_mask(bog)))
    # a prediction on an image that has no GT rows must be ignored
    rows.append(("20991231000000Zh_1", encode_mask(bog)))
    pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    off = get_pq_score(get_overlap_df(gt_df, pred_df))
    mine = evaluate(gt_df, pred_df)
    assert abs(off[0] - mine.pq) < 1e-9, (off, mine.summary())
    assert off[1:] == (mine.tp, mine.fp, mine.fn), (off, mine.summary())
    # a file with GT but no prediction rows → all FN
    gt2 = build_gt_df(ts, files[:3])
    mine2 = evaluate(gt2, pred_df.iloc[:0])
    assert mine2.tp == 0 and mine2.fp == 0 and mine2.fn == len(gt2) and mine2.pq == 0.0
    print("official:", off, "| mine:", mine.summary(), "| EQUIVALENT")


if __name__ == "__main__":
    test_equivalence()
