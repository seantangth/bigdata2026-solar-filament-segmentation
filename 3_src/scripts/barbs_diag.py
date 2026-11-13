"""E-065 round 1: are our masks smoother than the annotators' (barbs / fine structure under-segmented)?

For every TP pair (IoU > 0.5, per annotator row) of an OOF variant (masks rebuilt exactly as `rubric_metrics.py`;
`miou` = organisers' Multiscale IoU of the pair, `miou_metric.py`):
  * shape statistics of prediction and GT: perimeter / sqrt(area), solidity (area / convex-hull area),
    thin fraction (pixels removed by a disk-r opening = protrusions narrower than ~2r+1 px, i.e. barbs / thin ends),
    skeleton branch points per 100 skeleton px;
  * where the errors sit: missed GT pixels (GT \\ pred) split into GT-thin vs GT-core, extra pixels (pred \\ GT) split
    into "adjacent" (within 3 px of GT) vs far.
Usage: python barbs_diag.py --spec spec.json --variant D21L6 --gt gt_train_v1.csv --geometry disk_geometry_v1.csv --out OUT [--r 3]
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rubric_metrics as RM  # noqa: E402
from miou_metric import miou_full  # noqa: E402

H = W = 2048
R_OPEN = 3


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]; return x * x + y * y <= r * r


def shape(m):
    import cv2
    from skimage.morphology import skeletonize
    a = m.sum(); er = ndimage.binary_erosion(m); per = (m & ~er).sum()
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cs))) if cs else 0.0
    thin = m & ~ndimage.binary_opening(m, structure=_disk(R_OPEN))
    sk = skeletonize(m); nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - 1
    branch = int((sk & (nb >= 3)).sum())
    return dict(area=float(a), per_ratio=float(per / np.sqrt(max(a, 1))), solidity=float(a / max(hull, 1.0)),
                thin_frac=float(thin.sum() / max(a, 1)), branch_per100=float(100 * branch / max(sk.sum(), 1))), thin


def _work(job):
    stem, geo, variants, keep_d1, links, gt_rows = job
    _, out = RM._work((stem, geo, variants, keep_d1, links))
    preds = [mu.decode({'size': [H, W], 'counts': r.encode('ascii')}).astype(bool) for _, r in next(iter(out.values()))]
    key = gt_rows['filament_id'].str.split('_', n=1).str[0]
    rows = []
    for ai, g in gt_rows.groupby(key, sort=False):
        for gi, rle in enumerate(g['segmentation_rle']):
            gm = mu.decode({'size': [H, W], 'counts': rle.encode('ascii')}).astype(bool)
            for pm in preds:
                inter = (gm & pm).sum()
                if inter == 0: continue
                iou = inter / (gm.sum() + pm.sum() - inter)
                if iou <= 0.5: continue
                ys, xs = np.nonzero(gm | pm); y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9); x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)
                G, P = gm[y0:y1, x0:x1], pm[y0:y1, x0:x1]
                sg, thin_g = shape(G); sp, _ = shape(P)
                miss = G & ~P; extra = P & ~G
                near = ndimage.binary_dilation(G, structure=_disk(3))
                rows.append(dict(stem=stem, ann=ai, gi=gi, iou=float(iou), **{f'gt_{k}': v for k, v in sg.items()}, **{f'pr_{k}': v for k, v in sp.items()},
                                 miss=int(miss.sum()), miss_thin=int((miss & thin_g).sum()), gt_thin=int(thin_g.sum()),
                                 extra=int(extra.sum()), extra_near=int((extra & near).sum()), miou=miou_full(gm, pm)))
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    for k in ('spec', 'gt', 'geometry', 'out'):
        ap.add_argument('--' + k, type=Path, required=True)
    ap.add_argument('--variant', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    spec = json.loads(a.spec.read_text()); V = {a.variant: spec['variants'][a.variant]}
    gt = pd.read_csv(a.gt); gkey = gt.filament_id.str.split('_', n=1).str[0].str.split('-', n=1).str[1]
    geo = pd.read_csv(a.geometry).set_index('stem')
    stems = sorted(p.stem for p in Path(V[a.variant]['probs']).glob('*.npz'))
    keep_d1, links = {}, {}
    v = V[a.variant]
    if v['method'] == 'd1':
        C = pd.read_csv(v['candidates'], usecols=['stem', 'cid', 'yhat'])
        keep_d1[a.variant] = {s: g.sort_values('cid').cid.astype(int).tolist() for s, g in C[C.yhat >= v['tau']].groupby('stem')}
    if v.get('links'):
        Lk = pd.read_csv(v['links'], usecols=['stem', 'a', 'b', 'p']); Lk = Lk[Lk.p >= v['link_thr']]
        links[(v['links'], v['link_thr'])] = {s: list(zip(g.a.astype(int), g.b.astype(int))) for s, g in Lk.groupby('stem')}
    jobs = [(s, tuple(float(geo.loc[s, k]) for k in ('cx', 'cy', 'r')), V, {x: keep_d1[x].get(s, []) for x in keep_d1},
             {k: {s: L.get(s, [])} for k, L in links.items()}, gt[gkey == s].reset_index(drop=True)) for s in stems]
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=2), 1):
            rows.extend(r)
            if i % 100 == 0 or i == len(jobs): print(f'[{time.time() - t0:5.0f}s] {i}/{len(jobs)} images, {len(rows)} TP pairs', flush=True)
    D = pd.DataFrame(rows); D.to_csv(a.out / f'tp_shapes_{a.variant}.csv', index=False)
    rep = dict(variant=a.variant, n_tp_pairs=len(D), r_open=R_OPEN, median={}, pooled={})
    for k in ('area', 'per_ratio', 'solidity', 'thin_frac', 'branch_per100'):
        rep['median'][k] = dict(gt=round(float(D[f'gt_{k}'].median()), 4), pred=round(float(D[f'pr_{k}'].median()), 4),
                                pred_lower_share=round(float((D[f'pr_{k}'] < D[f'gt_{k}']).mean()), 4))
    rep['pooled'] = dict(gt_thin_share_of_gt=round(float(D.gt_thin.sum() / D.gt_area.sum()), 4),
                         thin_share_of_missed=round(float(D.miss_thin.sum() / max(D.miss.sum(), 1)), 4),
                         thin_recall=round(float(1 - D.miss_thin.sum() / max(D.gt_thin.sum(), 1)), 4),
                         core_recall=round(float(1 - (D.miss.sum() - D.miss_thin.sum()) / max(D.gt_area.sum() - D.gt_thin.sum(), 1)), 4),
                         missed_px=int(D.miss.sum()), extra_px=int(D.extra.sum()), extra_near_share=round(float(D.extra_near.sum() / max(D.extra.sum(), 1)), 4),
                         pred_area_over_gt=round(float(D.pr_area.sum() / D.gt_area.sum()), 4))
    rep['mean'] = dict(iou=round(float(D.iou.mean()), 4), miou=round(float(D.miou.mean()), 4), miou_median=round(float(D.miou.median()), 4))   # MIoU: organisers' barbs metric (miou_metric.py)
    rep['minutes'] = round((time.time() - t0) / 60, 1)
    (a.out / f'report_{a.variant}.json').write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=1), flush=True)


if __name__ == '__main__':
    main()
