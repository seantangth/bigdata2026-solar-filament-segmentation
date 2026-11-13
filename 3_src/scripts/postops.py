"""E-064 / E-065 post-operations on a FINAL instance label map (after selection), shared by OOF evaluation and test.

bridge_instances  (E-064, gap filling): an instance made of several disconnected pieces (learned links or merge_px
                  identity merge) gets thin bridges along the nearest-point segments of a minimum spanning tree over its
                  pieces (gap <= max_gap). Bridge pixels are drawn with thickness `width`, kept only on background, inside
                  the valid disk and where prob >= prob_min. MAGFiLO annotators extrapolate such gaps into one mask.
grow_instances    (E-065, barbs / fine structure): background pixels within `d` px of an instance and with
                  prob >= t join their nearest instance (exclusive by construction).
                  E-071: with op['src'] == 'aux' the growth signal is an auxiliary probability map (the fold-0
                  Skeleton-Recall model, E-070d) instead of the ensemble map; instance identity stays the ensemble's.
Both never touch pixels of other instances, so the output stays pixel-disjoint.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

_EIGHT = np.ones((3, 3), bool)


def _nearest(piece_a, piece_b):
    """nearest pixels between two boolean masks (same crop) → (gap, (ya, xa), (yb, xb))."""
    dist, ind = ndimage.distance_transform_edt(~piece_a, return_indices=True)
    dd = np.where(piece_b, dist, np.inf); j = int(np.argmin(dd))
    yb, xb = np.unravel_index(j, dd.shape)
    return float(dd.flat[j]), (int(ind[0][yb, xb]), int(ind[1][yb, xb])), (int(yb), int(xb))


def bridge_instances(lab, prob, valid, width=5, prob_min=0.0, max_gap=64):
    out = lab.copy(); added = 0
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - 2), min(lab.shape[0], sl[0].stop + 2); x0, x1 = max(0, sl[1].start - 2), min(lab.shape[1], sl[1].stop + 2)
        m = lab[y0:y1, x0:x1] == i
        pl, k = ndimage.label(m, structure=_EIGHT)
        if k < 2:
            continue
        pieces = [pl == j for j in range(1, k + 1)]
        edges = []
        for a in range(k):
            for b in range(a + 1, k):
                g, pa, pb = _nearest(pieces[a], pieces[b])
                if g <= max_gap:
                    edges.append((g, a, b, pa, pb))
        parent = list(range(k))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        canvas = np.zeros(m.shape, np.uint8)
        for g, a, b, pa, pb in sorted(edges):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            parent[rb] = ra
            if cv2 is not None:
                cv2.line(canvas, (pa[1], pa[0]), (pb[1], pb[0]), 1, thickness=int(width))
            else:
                t = np.linspace(0, 1, int(g) + 2); yy = np.round(pa[0] + t * (pb[0] - pa[0])).astype(int); xx = np.round(pa[1] + t * (pb[1] - pa[1])).astype(int)
                canvas[yy, xx] = 1; canvas[:] = ndimage.binary_dilation(canvas, iterations=max(0, int(width) // 2))
        new = (canvas > 0) & (out[y0:y1, x0:x1] == 0) & valid[y0:y1, x0:x1] & (prob[y0:y1, x0:x1] >= prob_min * 255)
        out[y0:y1, x0:x1][new] = i; added += int(new.sum())
    return out, added


def dark_contrast(img, size=21):
    """(local background - pixel) / background on the H-alpha image; > 0 = darker than surroundings (filament / barb)."""
    f = img.astype(np.float32); bg = ndimage.uniform_filter(f, size=size)
    return (bg - f) / np.maximum(bg, 1.0)


def grow_instances(lab, prob, valid, d=4, t=0.35, contrast=None, c_min=None):
    if lab.max() == 0:
        return lab, 0
    dist, ind = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    grow = (lab == 0) & (dist <= d) & valid & (prob >= t * 255)
    if c_min is not None:               # image-gated growth: only pixels that are visibly dark on the H-alpha image
        grow &= contrast >= c_min
    out = lab.copy(); out[grow] = lab[ind[0][grow], ind[1][grow]]
    return out, int(grow.sum())


def apply_ops(lab, prob, valid, ops, img=None, aux=None):
    stats = {}; contrast = None
    for op in ops or []:
        if op['op'] == 'bridge':
            lab, n = bridge_instances(lab, prob, valid, op.get('width', 5), op.get('prob_min', 0.0), op.get('max_gap', 64))
        elif op['op'] == 'grow':
            if op.get('c_min') is not None and contrast is None:
                assert img is not None, 'image-gated growth needs the H-alpha image'
                contrast = dark_contrast(img, op.get('c_size', 21))
            src = prob
            if op.get('src') == 'aux':
                assert aux is not None, 'aux-sourced growth needs the auxiliary probability map'
                src = aux
            lab, n = grow_instances(lab, src, valid, op.get('d', 4), op.get('t', 0.35), contrast, op.get('c_min'))
        else:
            raise ValueError(op)
        stats[op['op']] = stats.get(op['op'], 0) + n
    return lab, stats
