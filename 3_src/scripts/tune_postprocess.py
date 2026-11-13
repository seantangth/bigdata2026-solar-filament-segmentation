"""A-base post-processing tuner on OOF probability maps (CPU; runs on the GPU VM per Sean's decision F).

Searches  t_lo × t_hi (hysteresis) × merge_px (fragment identity merge) × min_area × gate (instance mean prob)
and scores every config with the official PQ semantics (per annotator-image, IoU > 0.5 strict, pooled).
Self-contained: numpy, scipy, pandas, pycocotools (+ cv2 if available). The block between the SHARED markers is
copied verbatim into SFSC_05_infer (single source of truth for inference post-processing).

Usage:
  python tune_postprocess.py --probs DIR(npz with 'prob' uint8) --gt gt_train_v1.csv --geometry disk_geometry_v1.csv \
         --out OUTDIR [--workers 5] [--gdrive-dest gdrive:...]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from scipy import ndimage

# >>> SHARED POSTPROCESS (copied verbatim into SFSC_05_infer; do not edit one without the other)
try:
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None
_PP_H = _PP_W = 2048
_EIGHT = np.ones((3, 3), dtype=bool)


def pp_valid_mask(cx: float, cy: float, r: float, margin: float = 8.0) -> np.ndarray:
    yy, xx = np.mgrid[0:_PP_H, 0:_PP_W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (r + margin) ** 2


def pp_candidates(prob_u8: np.ndarray, valid: np.ndarray, t_lo: float, t_hi: float, merge_px: int):
    """Hysteresis (components of prob>=t_lo that reach t_hi) → fill holes → optional identity merge of fragments
    closer than merge_px (union of ORIGINAL pixels; shapes never grow). Returns (label map int32, n, area, mean_prob)."""
    lo, hi = int(round(t_lo * 255)), int(round(t_hi * 255))
    b = ndimage.binary_fill_holes((prob_u8 >= lo) & valid)
    lab, n = ndimage.label(b, structure=_EIGHT)
    if n == 0:
        return lab, 0, np.zeros(0), np.zeros(0)
    mx = ndimage.maximum(prob_u8, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool); keep[1:] = np.asarray(mx) >= hi
    b = keep[lab]
    if merge_px and merge_px > 0:
        k = 2 * int(np.ceil(merge_px / 2)) + 1
        if _cv2 is not None:
            dil = _cv2.dilate(b.astype(np.uint8), _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))) > 0
        else:
            dil = ndimage.binary_dilation(b, iterations=int(np.ceil(merge_px / 2)))
        dl, n = ndimage.label(dil, structure=_EIGHT)
        lab = np.where(b, dl, 0).astype(np.int32)
    else:
        lab, n = ndimage.label(b, structure=_EIGHT)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    psum = np.bincount(lab.ravel(), weights=prob_u8.ravel().astype(np.float64), minlength=n + 1)[1:]
    meanp = np.where(area > 0, psum / np.maximum(area, 1) / 255.0, 0.0)
    return lab, n, area, meanp


def pp_instances(prob_u8: np.ndarray, valid: np.ndarray, params: dict) -> list:
    """Final instance masks (disjoint by construction) for one image."""
    lab, n, area, meanp = pp_candidates(prob_u8, valid, params['t_lo'], params['t_hi'], int(params.get('merge_px', 0)))
    keep = np.flatnonzero((area >= params.get('min_area', 0)) & (meanp >= params.get('gate', 0.0))) + 1
    return [lab == i for i in keep]
# <<< SHARED POSTPROCESS

H = W = 2048
GRIDS = {
    # full grid (E-020a: 9,000 configs, 41 min on 4 workers) — use when a new model family appears
    'full': dict(T_LO=[0.30, 0.35, 0.40, 0.45, 0.50], T_HI=[0.50, 0.60, 0.70, 0.80, 0.90], MERGE=[0, 8, 16, 24, 32],
                 MIN_AREA=[0, 100, 200, 300, 400, 500, 700, 1000], GATE=[0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]),
    # fast grid (E-020a showed hysteresis ~flat, merge 16-24 and min_area 300-500 matter, gate best at the 0.85 edge)
    'fast': dict(T_LO=[0.40, 0.50], T_HI=[0.50, 0.70], MERGE=[0, 16, 24], MIN_AREA=[200, 300, 400, 500],
                 GATE=[0.0, 0.80, 0.85, 0.90, 0.95]),
    # tta grid (E-023: with 4-view TTA the best gate sits at 0.90; refine around it on the full 5-fold OOF)
    'tta': dict(T_LO=[0.40, 0.50], T_HI=[0.50, 0.70], MERGE=[0, 16, 24], MIN_AREA=[150, 200, 300, 400],
                GATE=[0.0, 0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.95]),
    # multi-resolution average (E-031): 512+1024 50/50 optimum sat at t_lo 0.3 / gate 0.5 (AC) — span low t_lo and the whole gate range
    'mr': dict(T_LO=[0.30, 0.40, 0.50], T_HI=[0.50, 0.70], MERGE=[0, 16, 24], MIN_AREA=[150, 200, 300, 400],
               GATE=[0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.89, 0.91]),
    # soft-label models (D2): probabilities ≈ annotator fraction → thresholds/gates sit lower
    'soft': dict(T_LO=[0.30, 0.40, 0.50], T_HI=[0.50, 0.70], MERGE=[0, 16, 24], MIN_AREA=[200, 300, 400],
                 GATE=[0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]),
}
T_LO, T_HI, MERGE, MIN_AREA, GATE = (GRIDS['full'][k] for k in ('T_LO', 'T_HI', 'MERGE', 'MIN_AREA', 'GATE'))


def _gt_maps(gt_rows: pd.DataFrame):
    """{annotator-image id: int16 label map} for one stem, from official-format RLE rows."""
    key = gt_rows['filament_id'].str.split('_', n=1).str[0]
    out = {}
    for ai, g in gt_rows.groupby(key, sort=False):
        lab = np.zeros((H, W), dtype=np.int16)
        for k, c in enumerate(g['segmentation_rle'], 1):
            lab[mask_util.decode({'size': [H, W], 'counts': c.encode('ascii')}).astype(bool)] = k
        out[ai] = (lab, len(g))
    return out


def _work(args):
    stem, prob_path, gt_rows, geo, grid = args
    t_los, t_his, merges = grid
    prob = np.load(prob_path)['prob']
    valid = pp_valid_mask(*geo)
    gts = _gt_maps(gt_rows)
    gt_area = {ai: np.bincount(lab.ravel(), minlength=ng + 1)[1:].astype(np.float64) for ai, (lab, ng) in gts.items()}
    res = {}
    for t_lo in t_los:
        for t_hi in t_his:
            if t_hi < t_lo:
                continue
            for m in merges:
                lab, n, area, meanp = pp_candidates(prob, valid, t_lo, t_hi, m)
                inters = {}
                for ai, (g, ng) in gts.items():
                    if n == 0:
                        inters[ai] = np.zeros((0, ng)); continue
                    idx = np.flatnonzero((lab > 0) & (g > 0))
                    tab = np.bincount(lab.ravel()[idx].astype(np.int64) * (ng + 1) + g.ravel()[idx], minlength=(n + 1) * (ng + 1))
                    inters[ai] = tab.reshape(n + 1, ng + 1)[1:, 1:].astype(np.float64)
                res[(t_lo, t_hi, m)] = (area, meanp, inters)
    return stem, res, gt_area


def _score(per_image, gt_areas, key, min_area, gate, stems=None):
    TP = FP = FN = 0; S = 0.0; deg_gt = []; deg_pred = []
    for stem, res in per_image.items():
        if stems is not None and stem not in stems:
            continue
        area, meanp, inters = res[key]
        keep = (area >= min_area) & (meanp >= gate)
        a_p = area[keep]
        for ai, inter in inters.items():
            a_g = gt_areas[stem][ai]
            I = inter[keep]
            if I.shape[0] == 0:
                FN += len(a_g); continue
            iou = I / (a_p[:, None] + a_g[None, :] - I)
            hit = iou > 0.5
            tp = int(hit.sum()); TP += tp; S += float(iou[hit].sum())
            FP += int(I.shape[0] - hit.any(1).sum()); FN += int(len(a_g) - hit.any(0).sum())
            deg_gt.extend((I > 0).sum(0).tolist()); deg_pred.extend((I > 0).sum(1).tolist())
    den = TP + 0.5 * FP + 0.5 * FN
    return dict(PQ=S / den if den else 0.0, SQ=S / TP if TP else 0.0, RQ=TP / den if den else 0.0, TP=TP, FP=FP, FN=FN,
                gt_1toN=int(np.sum(np.array(deg_gt) >= 2)), pred_Nto1=int(np.sum(np.array(deg_pred) >= 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True); ap.add_argument('--gt', required=True); ap.add_argument('--geometry', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=5); ap.add_argument('--gdrive-dest')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--grid', default='full', choices=list(GRIDS))
    a = ap.parse_args()
    global T_LO, T_HI, MERGE, MIN_AREA, GATE
    T_LO, T_HI, MERGE, MIN_AREA, GATE = (GRIDS[a.grid][k] for k in ('T_LO', 'T_HI', 'MERGE', 'MIN_AREA', 'GATE'))
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.probs).glob('*.npz'))
    if a.limit: files = files[:a.limit]
    stems = [f.stem for f in files]
    gt = pd.read_csv(a.gt); gkey = gt['filament_id'].str.split('_', n=1).str[0].str.split('-', n=1).str[1]
    geo = pd.read_csv(a.geometry).set_index('stem')
    jobs = [(s, str(f), gt[gkey == s].reset_index(drop=True), (float(geo.loc[s, 'cx']), float(geo.loc[s, 'cy']), float(geo.loc[s, 'r'])), (T_LO, T_HI, MERGE)) for s, f in zip(stems, files)]
    per_image, gt_areas = {}, {}
    with Pool(a.workers) as pool:
        for i, (stem, res, ga) in enumerate(pool.imap_unordered(_work, jobs), 1):
            per_image[stem] = res; gt_areas[stem] = ga
            if i % 10 == 0 or i == len(jobs):
                print(f'[{time.time() - t0:6.0f}s] candidates {i}/{len(jobs)}', flush=True)
    rows = []
    keys = list(next(iter(per_image.values())).keys())
    for key in keys:
        for ma in MIN_AREA:
            for g in GATE:
                r = _score(per_image, gt_areas, key, ma, g)
                rows.append(dict(t_lo=key[0], t_hi=key[1], merge_px=key[2], min_area=ma, gate=g, **r))
        print(f'[{time.time() - t0:6.0f}s] scored t_lo={key[0]} t_hi={key[1]} merge={key[2]}', flush=True)
    grid = pd.DataFrame(rows).sort_values('PQ', ascending=False)
    grid.to_csv(out / 'pp_grid.csv', index=False)
    best = grid.iloc[0].to_dict()
    bkey = (best['t_lo'], best['t_hi'], int(best['merge_px']))
    # reference points
    simple = grid[(grid.t_lo == 0.5) & (grid.t_hi == 0.5) & (grid.merge_px == 0) & (grid.min_area == 300) & (grid.gate == 0.0)].iloc[0].to_dict()
    # subsets: single- vs multi-annotated files
    n_ann = {s: len(gt_areas[s]) for s in gt_areas}
    single = {s for s in stems if n_ann[s] == 1}; multi = {s for s in stems if n_ann[s] > 1}
    sub = dict(single=_score(per_image, gt_areas, bkey, best['min_area'], best['gate'], single),
               multi=_score(per_image, gt_areas, bkey, best['min_area'], best['gate'], multi))
    gate_curve = grid[(grid.t_lo == best['t_lo']) & (grid.t_hi == best['t_hi']) & (grid.merge_px == best['merge_px']) & (grid.min_area == best['min_area'])].sort_values('gate')[['gate', 'PQ', 'RQ', 'SQ', 'TP', 'FP', 'FN']]
    params = dict(t_lo=float(best['t_lo']), t_hi=float(best['t_hi']), merge_px=int(best['merge_px']), min_area=int(best['min_area']), gate=float(best['gate']), valid_margin=8.0)
    rep = dict(grid=a.grid, params=params, best={k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()},
               simple_thr05_ma300={k: (round(v, 4) if isinstance(v, float) else v) for k, v in simple.items()},
               subsets={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in sub.items()},
               n_images=len(stems), n_single=len(single), n_multi=len(multi), n_configs=len(grid), minutes=round((time.time() - t0) / 60, 1))
    json.dump(rep, open(out / 'pp_best.json', 'w'), indent=2)
    gate_curve.to_csv(out / 'pp_gate_curve.csv', index=False)
    print(json.dumps(rep, indent=1))
    print(gate_curve.to_string(index=False))
    if a.gdrive_dest:
        rc = '/usr/local/bin/rclone' if Path('/usr/local/bin/rclone').exists() else 'rclone'
        conf = ['--config', '/content/rclone.conf'] if Path('/content/rclone.conf').exists() else []
        for f in ['pp_grid.csv', 'pp_best.json', 'pp_gate_curve.csv']:
            subprocess.run([rc, *conf, 'copy', str(out / f), a.gdrive_dest, '-q'], check=False)
        print('pushed to', a.gdrive_dest)


if __name__ == '__main__':
    main()
