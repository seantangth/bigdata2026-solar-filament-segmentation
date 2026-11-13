"""Layer 1: Physics-exact heliocentric frontend for single-image H-alpha observations (E-120).

Computes 5 physically grounded, single-image-only channels without off-disk black-space bleed
or artificial limb bright-ring saturation:
  0. I_norm        : Standardized disk intensity ((I - 120) / 45), clamped off-disk.
  1. Delta_I_qs    : Quiet-Sun radial mu-profile flattened optical absorption depth
                     (I_70(r/R) - I) / max(I_70(r/R), 30), eliminating limb darkening.
  2. Delta_I_local : Disk-mirror-padded Gaussian local absorption contrast (sigma=22 px),
                     eliminating the bright halo artifact of naive Cartesian GaussianBlur.
  3. Ridge_dark    : Multi-scale Hessian dark-valley curvature response (sigma in {1.5, 3.0, 5.5} px)
                     enhancing 1.5-5 px barbs and faint filament threads.
  4. mu            : Heliocentric cosine sqrt(max(0, 1 - (r/R)^2)) encoding limb foreshortening.

Strictly single-image compliant: uses only pixel intensities of the given H-alpha frame (plus
optional pre-fitted DiskFit from sfsc.geometry.fit_disk, which itself uses only that image).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry import DiskFit, fit_disk

N_PHYSICS_CHANNELS = 5


def _radial_quiet_sun_profile(
    img_f32: np.ndarray,
    r_px: np.ndarray,
    radius: float,
    n_bins: int = 64,
) -> np.ndarray:
    """Estimate smooth 1D radial quiet-Sun background I_70(r) strictly inside the solar disk.

    Uses the 70th percentile per concentric annulus inside r <= 0.978 * R (avoiding the
    wavelet limb-brightening ring at r ~ 1.00 * R and dark active-belt bias), then smoothly
    extrapolates across [0.978 * R, 1.015 * R] using monotonic limb-darkening slope.
    """
    r_inner_max = 0.978 * radius
    edges = np.linspace(0.0, r_inner_max, n_bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Subsample by 2x for 4x speed (~1M pixels is more than enough for 64 radial quantiles)
    r_sub = r_px[::2, ::2].ravel()
    i_sub = img_f32[::2, ::2].ravel()
    in_mask = (r_sub <= r_inner_max) & (i_sub > 5.0)
    r_in = r_sub[in_mask]
    i_in = i_sub[in_mask]

    if r_in.size < 256:
        return np.full_like(img_f32, 128.0, dtype=np.float32)

    bin_idx = np.clip(np.searchsorted(edges, r_in, side="right") - 1, 0, n_bins - 1)
    prof = np.empty(n_bins, dtype=np.float32)
    global_q70 = float(np.percentile(i_in, 70.0))
    for b in range(n_bins):
        vals = i_in[bin_idx == b]
        if vals.size >= 16:
            prof[b] = float(np.percentile(vals, 70.0))
        else:
            prof[b] = prof[b - 1] if b > 0 else global_q70

    prof = gaussian_filter1d(prof, sigma=1.5, mode="nearest").astype(np.float32)

    # Extend smoothly from 0.978*R to 1.03*R without picking up the artificial bright wavelet ring
    slope = float(prof[-1] - prof[-5]) / max(float(centers[-1] - centers[-5]), 1.0)
    slope = min(slope, 0.0)  # Limb darkening is monotonically non-increasing outward
    ext_r = np.array([radius * 0.995, radius * 1.020], dtype=np.float32)
    ext_v = np.maximum(prof[-1] + slope * (ext_r - centers[-1]), 45.0).astype(np.float32)

    knots_r = np.concatenate([[0.0], centers, ext_r])
    knots_v = np.concatenate([[prof[0]], prof, ext_v])
    return np.interp(r_px, knots_r, knots_v).astype(np.float32)


def _mirror_pad_outside_disk(
    img_f32: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    r_px: np.ndarray,
    cx: float,
    cy: float,
    r_reflect: float,
) -> np.ndarray:
    """Radially reflect interior quiet-disk intensities across r_reflect to prevent blur bleed."""
    out = img_f32.copy()
    ext = r_px > r_reflect
    if not np.any(ext):
        return out
    rx = xx[ext] - cx
    ry = yy[ext] - cy
    rr = np.maximum(r_px[ext], 1e-4)
    # Mirror radius: r_mir = 2 * r_reflect - r, clamped inside [0.70 * r_reflect, r_reflect - 1]
    r_mir = np.clip(2.0 * r_reflect - rr, 0.70 * r_reflect, r_reflect - 1.0)
    scale = r_mir / rr
    mx = np.clip(np.round(cx + rx * scale).astype(np.int32), 0, img_f32.shape[1] - 1)
    my = np.clip(np.round(cy + ry * scale).astype(np.int32), 0, img_f32.shape[0] - 1)
    out[ext] = img_f32[my, mx]
    return out


def _multiscale_dark_ridge(
    img_mirror: np.ndarray,
    valid_disk: np.ndarray,
    sigmas: tuple[float, ...] = (1.5, 3.0, 5.5),
) -> np.ndarray:
    """Compute multi-scale Hessian principal dark-valley curvature (Frangi-like ridge strength).

    For dark filaments/barbs, intensity I has a local valley (positive principal eigenvalue of
    Hessian(I)). Scale-normalized sigma^1.6 * lambda_max highlights 1.5-5 px barbs and threads.
    """
    best = np.zeros_like(img_mirror, dtype=np.float32)
    for s in sigmas:
        sm = cv2.GaussianBlur(img_mirror, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT)
        # Second derivatives via Scharr/Sobel on Gaussian-smoothed image
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        hxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        hyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        hxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3) / 8.0

        tr = 0.5 * (hxx + hyy)
        det_disc = np.sqrt(np.maximum(0.25 * (hxx - hyy) ** 2 + hxy * hxy, 0.0))
        lam1 = tr + det_disc  # Largest eigenvalue of Hessian(I); > 0 in dark valleys/filaments
        lam2 = tr - det_disc
        # Suppress isotropic dark blobs slightly while preserving curvilinear ridges & junctions
        aniso = np.clip(1.0 - 0.35 * np.maximum(lam2, 0.0) / (np.maximum(lam1, 1e-4)), 0.35, 1.0)
        resp = np.maximum(lam1, 0.0) * (s ** 1.6) * aniso
        np.maximum(best, resp, out=best)

    # Robust scale normalization using 98th percentile inside disk
    in_vals = best[valid_disk]
    scale = float(np.percentile(in_vals, 98.5)) if in_vals.size > 0 else 1.0
    scale = max(scale, 1.0)
    ridge = np.clip(best / scale, 0.0, 2.5)
    ridge[~valid_disk] = 0.0
    return ridge.astype(np.float32)


def extract_physics_channels(
    img: np.ndarray,
    fit: DiskFit | None = None,
    margin_px: float = 8.0,
) -> tuple[np.ndarray, DiskFit, np.ndarray]:
    """Build (5, H, W) float32 physics-normalized input tensor from a single grayscale H-alpha image.

    Returns:
        channels: (5, H, W) float32 array roughly standardized in [-2.5, 2.5].
        fit: DiskFit(cx, cy, r, ...).
        valid_mask: (H, W) bool mask of valid solar disk (r <= R + margin_px).
    """
    if img.ndim == 3:
        img = img[..., 0]
    img_f32 = img.astype(np.float32)
    h, w = img_f32.shape
    if fit is None:
        fit = fit_disk(img)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r_px = np.hypot(xx - float(fit.cx), yy - float(fit.cy))
    radius = max(float(fit.r), 100.0)
    valid_mask = r_px <= (radius + margin_px)

    # 1. Quiet-Sun radial profile & relative optical absorption depth
    i_qs = _radial_quiet_sun_profile(img_f32, r_px, radius)
    delta_qs = np.clip((i_qs - img_f32) / np.maximum(i_qs, 30.0) * 4.0, -2.0, 3.0)
    delta_qs[~valid_mask] = 0.0

    # 2. Radially mirror-padded image (across 0.980 * R) for artifact-free local contrast & Hessian
    img_mirror = _mirror_pad_outside_disk(img_f32, xx, yy, r_px, float(fit.cx), float(fit.cy), 0.980 * radius)
    bg_local = cv2.GaussianBlur(img_mirror, (0, 0), sigmaX=22.0, sigmaY=22.0, borderType=cv2.BORDER_REFLECT)
    delta_local = np.clip((bg_local - img_f32) / np.maximum(bg_local, 30.0) * 4.5, -2.0, 3.0)
    # Near the artificial wavelet bright limb ring (0.985*R .. R+margin), blend smoothly toward delta_qs
    limb_ring_zone = (r_px > 0.982 * radius) & valid_mask
    if np.any(limb_ring_zone):
        w_ring = np.clip((r_px[limb_ring_zone] - 0.982 * radius) / (0.018 * radius), 0.0, 1.0)
        delta_local[limb_ring_zone] = (1.0 - w_ring) * delta_local[limb_ring_zone] + w_ring * delta_qs[limb_ring_zone]
    delta_local[~valid_mask] = 0.0

    # 3. Standardized intensity (off-disk filled with limb quiet-Sun level so conv padding has zero step)
    limb_qs_val = float(np.median(i_qs[(r_px >= 0.90 * radius) & (r_px <= 0.97 * radius)])) if np.any(valid_mask) else 100.0
    i_clean = np.where(valid_mask, img_f32, limb_qs_val)
    i_norm = np.clip((i_clean - 120.0) / 45.0, -2.5, 2.5).astype(np.float32)

    # 4. Multi-scale Hessian dark-valley ridge response (using img_mirror outside 0.988*R to avoid limb ring)
    img_for_ridge = _mirror_pad_outside_disk(img_f32, xx, yy, r_px, float(fit.cx), float(fit.cy), 0.986 * radius)
    ridge = _multiscale_dark_ridge(img_for_ridge, valid_mask)

    # 5. Heliocentric foreshortening cosine mu = sqrt(max(0, 1 - (r/R)^2))
    rho = r_px / radius
    mu = np.sqrt(np.maximum(1.0 - rho * rho, 0.0)).astype(np.float32)
    mu[~valid_mask] = 0.0

    channels = np.stack([i_norm, delta_qs.astype(np.float32), delta_local.astype(np.float32), ridge, mu], axis=0)
    return channels, fit, valid_mask
