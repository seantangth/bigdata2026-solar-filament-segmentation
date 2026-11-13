"""Multiscale IoU (MIoU; Ahmadzadeh, Kempton, Chen, Angryk, ICIP 2021, arXiv:2105.14572) — the organisers' metric
motivated by filament barbs. Re-implementation of bitbucket.org/gsudmlab/multiscale_iou (MIT), `miou/metrics/miou.py`
with edge_only=True and box sizes 2^0..2^9:

  edges   = Sobel(mask) != 0 (a ~2 px band on both sides of the boundary), for GT and prediction
  ratio_b = (#grid cells of side b containing GT edge AND prediction edge + 1) / (#cells containing GT edge + 1)
  MIoU    = trapezoidal area under ratio_b over the 10 box sizes, dx = 1/9   (in [0, 1]; extra predicted edges are
            not penalised -> a boundary-recall measure; report it next to IoU / PQ, never alone)

`miou_full` evaluates on a crop whose bounds are aligned to the 512 grid, so the box grid coincides with the full
2048x2048 frame used by the reference code (2048 is divisible by every box size: no padding).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BOX = 2 ** np.arange(10)
ALIGN = int(BOX.max())


def edges(mask):
    m = mask.astype(np.int32)
    return np.hypot(ndimage.sobel(m, axis=0, mode='constant'), ndimage.sobel(m, axis=1, mode='constant')) > 0


def _cells(e, b):
    h, w = e.shape
    return e.reshape(h // b, b, w // b, b).any(axis=(1, 3))


def miou(gt, pred):
    """gt / pred: boolean arrays whose shape is divisible by 512 and whose grid origin is the frame origin."""
    eg, ep = edges(gt), edges(pred); ratios = []
    for b in BOX:
        cg, cp = _cells(eg, int(b)), _cells(ep, int(b))
        ratios.append(((cg & cp).sum() + 1) / (cg.sum() + 1))
    r = np.asarray(ratios)
    return float(((r[1:] + r[:-1]) / 2).sum() / (len(BOX) - 1)), ratios   # trapezoid, dx = 1/9 (np.trapz is gone in numpy 2.x)


def miou_full(gm, pm, H=2048, W=2048):
    """gm / pm: full-frame boolean masks; evaluates on the smallest 512-aligned crop containing both (+2 px)."""
    ys, xs = np.nonzero(gm | pm)
    y0 = max(0, (ys.min() - 2) // ALIGN * ALIGN); x0 = max(0, (xs.min() - 2) // ALIGN * ALIGN)
    y1 = min(H, -(-(ys.max() + 3) // ALIGN) * ALIGN); x1 = min(W, -(-(xs.max() + 3) // ALIGN) * ALIGN)
    return miou(gm[y0:y1, x0:x1], pm[y0:y1, x0:x1])[0]
