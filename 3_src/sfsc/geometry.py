"""Robust solar-disk circle fit (v2, ray-maximum method).

Every GONG JPEG has a thin *bright* limb ring (the processed images show limb brightening, ~150–200 vs ~90 just
inside). Some images (notably Big Bear 2016–2022) also carry a bright grey halo outside the limb and the disk is
shifted by up to ~50 px inside the frame, which breaks naive threshold fits (E-006 follow-up). We therefore locate,
along 720 rays from a provisional centre, the radius of maximum intensity inside the annulus [r_lo, r_hi] and fit a
circle to those points with iterative outlier rejection (prominences/plages near the limb are rejected as outliers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import H, W


@dataclass
class DiskFit:
    cx: float
    cy: float
    r: float
    n_inliers: int   # of 720 rays
    rms: float


def _kasa(x, y):
    A = np.c_[x, y, np.ones_like(x)]
    b = -(x ** 2 + y ** 2)
    (a, bb, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -a / 2, -bb / 2
    return float(cx), float(cy), float(np.sqrt(max(cx ** 2 + cy ** 2 - c, 0.0)))


def _provisional_center(img: np.ndarray, thr: int = 70):
    m = img > thr
    lab, n = ndimage.label(m)
    if n == 0:
        return W / 2, H / 2
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    ys, xs = np.nonzero(lab == (1 + int(np.argmax(sizes))))
    return float(xs.mean()), float(ys.mean())


def fit_disk(img: np.ndarray, n_rays: int = 720, r_lo: int = 820, r_hi: int = 990, resid_px: float = 2.5,
             n_iter: int = 4, sigma: float = 1.5) -> DiskFit:
    g = ndimage.gaussian_filter(img.astype(np.float32), sigma)
    cx, cy = _provisional_center(img)
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    rr = np.arange(r_lo, r_hi + 1, dtype=np.float32)
    x = y = None
    for _ in range(n_iter):
        xs = cx + np.outer(np.cos(th), rr)
        ys = cy + np.outer(np.sin(th), rr)
        inside = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
        vals = ndimage.map_coordinates(g, [ys.ravel(), xs.ravel()], order=1, mode="constant", cval=-1.0).reshape(xs.shape)
        vals[~inside] = -1.0
        k = vals.argmax(axis=1)
        ok = vals[np.arange(n_rays), k] > 0                      # rays leaving the frame are dropped
        x, y = xs[np.arange(n_rays), k][ok], ys[np.arange(n_rays), k][ok]
        cx, cy, r = _kasa(x, y)
        for _ in range(3):                                      # outlier rejection (prominences, plages, frame)
            res = np.hypot(x - cx, y - cy) - r
            inl = np.abs(res) < resid_px
            if inl.sum() < 50:
                break
            cx, cy, r = _kasa(x[inl], y[inl])
    res = np.hypot(x - cx, y - cy) - r
    inl = np.abs(res) < resid_px
    return DiskFit(cx, cy, r, int(inl.sum()), float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("nan"))


def disk_mask(fit: DiskFit, margin_px: float = 8.0) -> np.ndarray:
    """Valid region. The ray-maximum fit sits on the limb-ring peak (≈905 px); GT pixels reach 0.999×R, so an 8 px
    margin keeps every GT pixel while excluding the off-disk halo."""
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - fit.cx) ** 2 + (yy - fit.cy) ** 2 <= (fit.r + margin_px) ** 2
