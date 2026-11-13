"""E-063 learned fragment linking — shared by the OOF diagnosis/evaluation and test-time inference.

Candidates: merge-0 connected components of the deployed hysteresis (`pp_candidates(..., merge_px=0)`).
`pair_features` (no GT): every component pair with area >= floor and nearest-pixel gap <= max_gap gets
  gap, dtheta (principal-axis angle difference), colin (max angle between each axis and the nearest-point segment),
  bridge_min / bridge_mean (probability on that segment), areas, elongations, mean probabilities, merged16.
`linked_candidates`: union of components whose pair probability p >= threshold (original pixels only, shapes never grow),
returned in the same (lab, n, area, meanp) form as `pp_candidates`.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from tune_postprocess import pp_candidates

H = W = 2048
FEATS = ['gap', 'dtheta', 'colin', 'bridge_min', 'bridge_mean', 'area_a', 'area_b', 'el_a', 'el_b', 'meanp_a', 'meanp_b']


def _axis(ys, xs):
    c = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
    w, v = np.linalg.eigh(c)
    major = v[:, 1]; elong = float(np.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9)))
    return major / np.linalg.norm(major), elong


def _ang(u, v):
    c = abs(float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return float(np.degrees(np.arccos(min(1.0, c))))


def pair_features(prob, lab, n, area, meanp, floor=50, max_gap=64):
    """component pairs (a < b) within max_gap px → list of feature dicts (keys a, b + FEATS + merged16)."""
    comps = [i + 1 for i in range(n) if area[i] >= floor]
    if len(comps) < 2:
        return []
    objs = ndimage.find_objects(lab); info = {}
    for c in comps:
        sl = objs[c - 1]; m = lab[sl] == c; ys, xs = np.nonzero(m)
        ax, el = _axis(ys + sl[0].start, xs + sl[1].start)
        info[c] = dict(sl=sl, ax=ax, el=el, area=float(area[c - 1]), meanp=float(meanp[c - 1]))
    G = max_gap; rows = []
    for c in comps:
        sl = info[c]['sl']
        y0, y1 = max(0, sl[0].start - G), min(H, sl[0].stop + G); x0, x1 = max(0, sl[1].start - G), min(W, sl[1].stop + G)
        crop = lab[y0:y1, x0:x1]
        dist, ind = ndimage.distance_transform_edt(crop != c, return_indices=True)
        for d in set(np.unique(crop).tolist()) - {0, c}:
            if d not in info or d < c:
                continue
            dd = np.where(crop == d, dist, np.inf); j = int(np.argmin(dd)); gap = float(dd.flat[j])
            if gap > G:
                continue
            py, px = np.unravel_index(j, dd.shape); qy, qx = ind[0][py, px], ind[1][py, px]   # nearest pixel of c to d
            seg = np.array([px - qx, py - qy], float)
            t = np.linspace(0, 1, max(2, int(gap) + 1))
            sy = np.clip(np.round(y0 + qy + t * (py - qy)).astype(int), 0, H - 1); sx = np.clip(np.round(x0 + qx + t * (px - qx)).astype(int), 0, W - 1)
            bp = prob[sy, sx].astype(float) / 255.0
            colin = max(_ang(info[c]['ax'], seg), _ang(info[d]['ax'], seg)) if gap > 0 else 0.0
            rows.append(dict(a=c, b=d, gap=gap, dtheta=_ang(info[c]['ax'], info[d]['ax']), colin=colin,
                             bridge_min=float(bp.min()), bridge_mean=float(bp.mean()), area_a=info[c]['area'], area_b=info[d]['area'],
                             el_a=info[c]['el'], el_b=info[d]['el'], meanp_a=info[c]['meanp'], meanp_b=info[d]['meanp'],
                             merged16=bool(gap <= 17)))
    return rows


def relabel(lab0, n0, links):
    parent = np.arange(n0 + 1)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n0 + 1)]); roots[0] = 0
    uniq, inv = np.unique(roots, return_inverse=True)       # 0 stays 0 (smallest root)
    return inv[lab0].astype(np.int32), len(uniq) - 1


def linked_candidates(prob_u8, valid, t_lo, t_hi, links):
    """pp_candidates(merge 0) + union of linked component pairs → (lab, n, area, meanp)."""
    lab0, n0, _, _ = pp_candidates(prob_u8, valid, t_lo, t_hi, 0)
    lab, n = relabel(lab0, n0, links) if (links and n0) else (lab0, n0)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    psum = np.bincount(lab.ravel(), weights=prob_u8.ravel().astype(np.float64), minlength=n + 1)[1:]
    meanp = np.where(area > 0, psum / np.maximum(area, 1) / 255.0, 0.0)
    return lab, n, area, meanp
