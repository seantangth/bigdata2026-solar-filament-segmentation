"""E-120 End-to-End Trainer & Multi-Scale Equivariant Inferencer for TopoFlowNet.

Pipeline per fold k in {0, 1, 2, 3, 4}:
  1. Strict leak-free split on folds_cvA_v1.csv:
       - outer_stems     : fold == k (141-142 images, strictly untouched during training & selection)
       - inner_val_stems : 12% stratified subset of fold != k (~68 images, selects epoch & operating params)
       - train_stems     : remaining 88% of fold != k (~497 images, ~810 annotator-image sets)
  2. Fast multiprocess NVMe pre-cache of 5-ch physics tensors (float16) and 5-ch topological targets (uint8).
  3. Interleaved Mixed-Scale + Continuous 0-360 deg Rotational Equivariant Training:
       - 768x768 native high-res crops (preserving 1-3 px barbs & 2 px inter-filament gaps)
       - 1024x1024 full-disk views (capturing 800 px cross-disk magnetic neutral line continuity)
       - Layer 4 Annotator Granularity FiLM with 35% Consensus Token dropout
       - bf16 AMP + AdamW + Cosine schedule + Model EMA
  4. Multi-Scale + Rotational + Gaussian Sliding-Window Equivariant Inference on:
       - inner_val_stems (for leak-free threshold & MBR parameter calibration)
       - outer_stems     (saved as compressed .npz + evaluated once against official Pooled PQ)
       - test_stems      (180 test images saved as compressed .npz + single-fold sanity CSV)
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "3_src"))

from sfsc import H, W
from sfsc.geometry import DiskFit, disk_mask, fit_disk
from sfsc.metrics import evaluate, masks_to_pred_df
from sfsc.submission import build_submission, validate_submission, write_submission
from sfsc.topoflow_net import TopoFlowLoss, TopoFlowNet
from sfsc.topoflow_physics import extract_physics_channels
from sfsc.topoflow_targets import (
    AnnotationSet,
    build_topoflow_targets,
    decode_instance_mask,
    decode_topoflow,
    invert_rotate_flip_predictions,
    load_magfilo_annotations,
    pq_mbr_solve,
    rotate_flip_tensor_and_flow,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_save_torch(path: Path, state: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def make_fold_splits(
    folds_df: pd.DataFrame,
    outer_fold: int,
    seed: int = 2026,
    inner_frac: float = 0.12,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic stratified split keeping outer_fold 100% untouched."""
    if outer_fold >= 0:
        outer_df = folds_df[folds_df["fold"] == outer_fold]
        dev_df = folds_df[folds_df["fold"] != outer_fold].copy()
    else:
        outer_df = folds_df.iloc[0:0]
        dev_df = folds_df.copy()

    rng = np.random.default_rng(seed + max(outer_fold, 0) * 97)
    dev_df["strata"] = dev_df["station"].astype(str) + "_" + np.minimum(dev_df["n_annot"], 3).astype(str)
    inner_stems: list[str] = []
    train_stems: list[str] = []
    for _, grp in dev_df.groupby("strata", sort=True):
        stems = sorted(grp["stem"].tolist())
        rng.shuffle(stems)
        n_in = max(1, int(round(len(stems) * inner_frac))) if len(stems) >= 4 else 0
        inner_stems.extend(stems[:n_in])
        train_stems.extend(stems[n_in:])

    return sorted(train_stems), sorted(inner_stems), sorted(outer_df["stem"].tolist())


# ---- Multiprocess NVMe Pre-Caching ----

def _cache_phys_worker(job: tuple[str, str, tuple[float, float, float] | None, str]) -> tuple[str, dict]:
    stem, img_path, geom_tuple, out_npy = job
    out_p = Path(out_npy)
    if geom_tuple is not None:
        fit = DiskFit(cx=geom_tuple[0], cy=geom_tuple[1], r=geom_tuple[2], n_inliers=720, rms=0.5)
    else:
        fit = None
    if out_p.is_file() and fit is not None:
        return stem, {"cx": fit.cx, "cy": fit.cy, "r": fit.r}

    img = np.array(Image.open(img_path))
    ch, fit_out, _ = extract_physics_channels(img, fit=fit)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_name(out_p.name + ".tmp.npy")
    np.save(tmp, ch.astype(np.float16))
    tmp.replace(out_p)
    return stem, {"cx": float(fit_out.cx), "cy": float(fit_out.cy), "r": float(fit_out.r)}


def _cache_target_worker(job: tuple[AnnotationSet, str, str]) -> str:
    aset, out_npy, out_meta = job
    out_p = Path(out_npy)
    meta_p = Path(out_meta)
    if out_p.is_file() and meta_p.is_file():
        return aset.set_id

    tgts = build_topoflow_targets(aset.instances, H, W)
    # Pack [body, spine, sep, vx, vy] into (5, 2048, 2048) uint8 for 0.05ms mmap slicing!
    packed = np.empty((5, H, W), dtype=np.uint8)
    packed[0] = np.clip(np.round(tgts["body"] * 255.0), 0, 255).astype(np.uint8)
    packed[1] = np.clip(np.round(tgts["spine"] * 255.0), 0, 255).astype(np.uint8)
    packed[2] = np.clip(np.round(tgts["sep"] * 255.0), 0, 255).astype(np.uint8)
    packed[3] = np.clip(np.round((tgts["flow"][0] + 1.0) * 127.0), 0, 254).astype(np.uint8)
    packed[4] = np.clip(np.round((tgts["flow"][1] + 1.0) * 127.0), 0, 254).astype(np.uint8)

    centers = []
    for inst in aset.instances:
        bx, by, bw, bh = inst.bbox
        centers.append([float(bx + 0.5 * bw), float(by + 0.5 * bh)])
        # For long filaments (>200 px), also add spine quartile points so ends get sampled evenly!
        sp = inst.spine_xy
        if sp.shape[0] >= 6:
            for q in (0.2, 0.5, 0.8):
                idx = int((sp.shape[0] - 1) * q)
                centers.append([float(sp[idx, 0]), float(sp[idx, 1])])

    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_name(out_p.name + ".tmp.npy")
    np.save(tmp, packed)
    tmp.replace(out_p)
    atomic_write_json(meta_p, {"set_id": aset.set_id, "centers": centers})
    return aset.set_id


def prepare_nvme_cache(
    cache_dir: Path,
    train_img_dir: Path,
    test_img_dir: Path,
    geom_df: pd.DataFrame | None,
    train_sets: list[AnnotationSet],
    needed_train_stems: list[str],
    needed_test_stems: list[str],
    n_procs: int = 6,
) -> dict[str, dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    phys_dir = cache_dir / "phys"
    tgts_dir = cache_dir / "tgts"
    phys_dir.mkdir(parents=True, exist_ok=True)
    tgts_dir.mkdir(parents=True, exist_ok=True)

    geom_lookup: dict[str, tuple[float, float, float]] = {}
    if geom_df is not None:
        for row in geom_df.itertuples(index=False):
            geom_lookup[str(row.stem)] = (float(row.cx), float(row.cy), float(row.r))

    phys_jobs = []
    for stem in needed_train_stems:
        img_p = train_img_dir / f"{stem}.jpeg"
        phys_jobs.append((stem, str(img_p), geom_lookup.get(stem), str(phys_dir / f"{stem}.npy")))
    for stem in needed_test_stems:
        img_p = test_img_dir / f"{stem}.jpeg"
        phys_jobs.append((stem, str(img_p), geom_lookup.get(stem), str(phys_dir / f"{stem}.npy")))

    t0 = time.time()
    p_beat = t0
    disk_meta: dict[str, dict[str, float]] = {}
    if n_procs > 1 and len(phys_jobs) > 4:
        with Pool(processes=n_procs) as pool:
            for idx_p, (stem, meta) in enumerate(pool.imap_unordered(_cache_phys_worker, phys_jobs, chunksize=4), 1):
                disk_meta[stem] = meta
                if time.time() - p_beat >= 8.0:
                    print(f"[CACHE-PHYS] {idx_p}/{len(phys_jobs)} images processed", flush=True)
                    p_beat = time.time()
    else:
        for j in phys_jobs:
            stem, meta = _cache_phys_worker(j)
            disk_meta[stem] = meta
    print(f"[CACHE] Physics 5-ch ready for {len(disk_meta)} images in {time.time()-t0:.1f}s", flush=True)

    tgt_jobs = [
        (s, str(tgts_dir / f"{s.set_id}.npy"), str(tgts_dir / f"{s.set_id}.json"))
        for s in train_sets
    ]
    t1 = time.time()
    t_beat = t1
    if n_procs > 1 and len(tgt_jobs) > 4:
        with Pool(processes=n_procs) as pool:
            for idx_tg, _ in enumerate(pool.imap_unordered(_cache_target_worker, tgt_jobs, chunksize=4), 1):
                if time.time() - t_beat >= 8.0:
                    print(f"[CACHE-TGTS] {idx_tg}/{len(tgt_jobs)} sets processed", flush=True)
                    t_beat = time.time()
    else:
        for j in tgt_jobs:
            _cache_target_worker(j)
    print(f"[CACHE] Topological 5-ch targets ready for {len(tgt_jobs)} sets in {time.time()-t1:.1f}s", flush=True)

    meta_file = cache_dir / "disk_meta.json"
    if meta_file.is_file():
        old = json.loads(meta_file.read_text())
        old.update(disk_meta)
        disk_meta = old
    atomic_write_json(meta_file, disk_meta)
    return disk_meta


# ---- Datasets (Continuous 0-360 deg Equivariant Crop & Full-Disk) ----

def _unpack_targets_u8(packed_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return masks_3ch (3, h, w) float32 in [0,1] and flow_2ch (2, h, w) float32 in [-1,1]."""
    masks = packed_u8[:3].astype(np.float32) * (1.0 / 255.0)
    flow = (packed_u8[3:5].astype(np.float32) - 127.0) * (1.0 / 127.0)
    return masks, np.clip(flow, -1.0, 1.0)


def _extract_padded_patch(arr_chw: np.ndarray, cy: int, cx: int, side: int, pad_vals: list[float]) -> np.ndarray:
    c, h, w = arr_chw.shape
    half = side // 2
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + side, x0 + side
    if 0 <= y0 and y1 <= h and 0 <= x0 and x1 <= w:
        return np.array(arr_chw[:, y0:y1, x0:x1], copy=True)

    out = np.empty((c, side, side), dtype=arr_chw.dtype)
    for idx in range(c):
        out[idx].fill(pad_vals[idx])
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)
    dy0, dx0 = sy0 - y0, sx0 - x0
    if sy1 > sy0 and sx1 > sx0:
        out[:, dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = arr_chw[:, sy0:sy1, sx0:sx1]
    return out


class TopoCropDataset(Dataset):
    """Native 1.0x high-resolution 768x768 crops with continuous 0-360 deg rotation & FiLM dropout."""

    def __init__(
        self,
        sets: list[AnnotationSet],
        cache_dir: Path,
        disk_meta: dict[str, dict[str, float]],
        crop_size: int = 768,
        crops_per_set: int = 6,
        consensus_drop: float = 0.35,
    ):
        self.sets = sets
        self.phys_dir = cache_dir / "phys"
        self.tgts_dir = cache_dir / "tgts"
        self.disk_meta = disk_meta
        self.crop_size = crop_size
        self.src_side = int(math.ceil(crop_size * 1.42)) // 2 * 2  # ~1090 -> zero black corners after rotation!
        self.crops_per_set = crops_per_set
        self.consensus_drop = consensus_drop
        self.centers: list[list[list[float]]] = []
        for s in sets:
            mp = self.tgts_dir / f"{s.set_id}.json"
            self.centers.append(json.loads(mp.read_text())["centers"] if mp.is_file() else [])

    def __len__(self) -> int:
        return len(self.sets) * self.crops_per_set

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        set_idx = index // self.crops_per_set
        aset = self.sets[set_idx]
        dm = self.disk_meta[aset.stem]
        dcx, dcy, dr = dm["cx"], dm["cy"], dm["r"]

        phys_mmap = np.load(self.phys_dir / f"{aset.stem}.npy", mmap_mode="r")
        tgt_mmap = np.load(self.tgts_dir / f"{aset.set_id}.npy", mmap_mode="r")

        ctrs = self.centers[set_idx]
        if ctrs and random.random() < 0.76:
            fx, fy = random.choice(ctrs)
            cx = int(round(fx + random.uniform(-190.0, 190.0)))
            cy = int(round(fy + random.uniform(-190.0, 190.0)))
        else:
            ang_r = random.uniform(0.0, 2.0 * math.pi)
            rad_r = dr * 0.92 * math.sqrt(random.random())
            cx = int(round(dcx + rad_r * math.cos(ang_r)))
            cy = int(round(dcy + rad_r * math.sin(ang_r)))

        cx = max(256, min(W - 256, cx))
        cy = max(256, min(H - 256, cy))

        phys_patch = _extract_padded_patch(phys_mmap, cy, cx, self.src_side, [-0.4, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
        tgt_patch_u8 = _extract_padded_patch(tgt_mmap, cy, cx, self.src_side, [0, 0, 0, 127, 127])
        masks_patch, flow_patch = _unpack_targets_u8(tgt_patch_u8)

        # Continuous 0-360 deg rotation (75%) or exact D4 (25%) + random flips
        angle = random.uniform(0.0, 360.0) if random.random() < 0.75 else random.choice((0.0, 90.0, 180.0, 270.0))
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        stack_8ch = np.concatenate([phys_patch, masks_patch], axis=0)
        warped_8ch, warped_flow = rotate_flip_tensor_and_flow(
            stack_8ch,
            flow_patch,
            angle_deg=angle,
            hflip=hflip,
            vflip=vflip,
            border_values=[-0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        m0 = (self.src_side - self.crop_size) // 2
        m1 = m0 + self.crop_size
        x_crop = warped_8ch[:5, m0:m1, m0:m1].copy()
        m_crop = warped_8ch[5:, m0:m1, m0:m1].copy()
        f_crop = warped_flow[:, m0:m1, m0:m1].copy()

        m_crop[0] = (m_crop[0] >= 0.5).astype(np.float32)
        m_crop[1:] = np.clip(m_crop[1:], 0.0, 1.0)

        # Mild photometric contrast/gain augmentation on intensity & contrast channels
        if random.random() < 0.8:
            gain = random.uniform(0.90, 1.10)
            bias = random.uniform(-0.08, 0.08)
            x_crop[:3] = x_crop[:3] * gain + bias
            x_crop[3] = x_crop[3] * random.uniform(0.90, 1.10)

        annot_id = 0 if random.random() < self.consensus_drop else aset.annotator_idx
        return (
            torch.from_numpy(x_crop),
            torch.from_numpy(m_crop),
            torch.from_numpy(f_crop),
            torch.tensor(annot_id, dtype=torch.long),
        )


class TopoFullDiskDataset(Dataset):
    """0.5x full-disk 1024x1024 views teaching global 800-px magnetic neutral line continuity."""

    def __init__(
        self,
        sets: list[AnnotationSet],
        cache_dir: Path,
        disk_meta: dict[str, dict[str, float]],
        full_size: int = 1024,
        consensus_drop: float = 0.35,
    ):
        self.sets = sets
        self.phys_dir = cache_dir / "phys"
        self.tgts_dir = cache_dir / "tgts"
        self.disk_meta = disk_meta
        self.full_size = full_size
        self.consensus_drop = consensus_drop

    def __len__(self) -> int:
        return len(self.sets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aset = self.sets[index]
        dm = self.disk_meta[aset.stem]
        sz = self.full_size
        scale = sz / float(W)

        phys_full = np.array(np.load(self.phys_dir / f"{aset.stem}.npy", mmap_mode="r"), dtype=np.float32)
        tgt_u8 = np.array(np.load(self.tgts_dir / f"{aset.set_id}.npy", mmap_mode="r"))
        masks_full, flow_full = _unpack_targets_u8(tgt_u8)

        phys_ds = np.stack([cv2.resize(phys_full[c], (sz, sz), interpolation=cv2.INTER_AREA) for c in range(5)], axis=0)
        # Preserve thin barbs/spines/sep peaks during downsampling via block max-pool on masks!
        ky, kx = H // sz, W // sz
        masks_ds = masks_full.reshape(3, sz, ky, sz, kx).max(axis=(2, 4))
        flow_ds = np.stack([cv2.resize(flow_full[c], (sz, sz), interpolation=cv2.INTER_LINEAR) for c in range(2)], axis=0)

        angle = random.uniform(0.0, 360.0) if random.random() < 0.75 else random.choice((0.0, 90.0, 180.0, 270.0))
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        center = (dm["cx"] * scale, dm["cy"] * scale)

        stack_8ch = np.concatenate([phys_ds, masks_ds], axis=0)
        warped_8ch, warped_flow = rotate_flip_tensor_and_flow(
            stack_8ch,
            flow_ds,
            angle_deg=angle,
            hflip=hflip,
            vflip=vflip,
            center=center,
            border_values=[-0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        x_d = warped_8ch[:5].copy()
        m_d = warped_8ch[5:].copy()
        m_d[0] = (m_d[0] >= 0.5).astype(np.float32)
        m_d[1:] = np.clip(m_d[1:], 0.0, 1.0)

        if random.random() < 0.8:
            x_d[:3] = x_d[:3] * random.uniform(0.90, 1.10) + random.uniform(-0.08, 0.08)

        annot_id = 0 if random.random() < self.consensus_drop else aset.annotator_idx
        return (
            torch.from_numpy(x_d),
            torch.from_numpy(m_d),
            torch.from_numpy(warped_flow.copy()),
            torch.tensor(annot_id, dtype=torch.long),
        )


def compute_gpu_fg_weight(tgt_body: torch.Tensor, tgt_sep: torch.Tensor) -> torch.Tensor:
    """Fast GPU morphological thin-structure & barb weight (3x boost on <=6px barbs/edges + sep)."""
    tb4 = tgt_body.unsqueeze(1)
    eroded = -F.max_pool2d(-tb4, kernel_size=7, stride=1, padding=3)
    thin_or_rim = (tb4 - eroded).squeeze(1).clamp(0.0, 1.0)
    return 1.0 + 1.8 * thin_or_rim + 2.5 * tgt_sep


# ---- Multi-Scale + Rotational + Gaussian Sliding-Window Inference ----

def _make_gaussian_tile_weight(tile: int) -> np.ndarray:
    u = np.linspace(-1.0, 1.0, tile, dtype=np.float32)
    yy, xx = np.meshgrid(u, u, indexing="ij")
    w = np.exp(-1.25 * (xx * xx + yy * yy))
    return np.clip(w, 0.12, 1.0).astype(np.float32)


def _gpu_rotate_flip_forward(
    x_bchw: torch.Tensor,
    angle_deg: float,
    hflip: bool,
    vflip: bool,
    center: tuple[float, float],
    border_vals: list[float] | None = None,
) -> torch.Tensor:
    """Exact GPU equivalent of rotate_flip_tensor_and_flow (bilinear + constant border)."""
    _, c, h_img, w_img = x_bchw.shape
    if abs(angle_deg) > 1e-5:
        if border_vals is not None:
            b_vec = torch.tensor(border_vals, dtype=torch.float32, device=x_bchw.device).view(1, c, 1, 1)
            x_in = x_bchw - b_vec
        else:
            b_vec = None
            x_in = x_bchw

        rad = math.radians(float(angle_deg))
        ca, sa = math.cos(rad), math.sin(rad)
        sx = 0.5 * float(w_img - 1)
        sy = 0.5 * float(h_img - 1)
        uc = (float(center[0]) / sx) - 1.0
        vc = (float(center[1]) / sy) - 1.0

        a00 = ca
        a01 = -sa * (sy / sx)
        a02 = uc - ca * uc + sa * (sy / sx) * vc
        a10 = sa * (sx / sy)
        a11 = ca
        a12 = vc - sa * (sx / sy) * uc - ca * vc

        theta = torch.tensor([[[a00, a01, a02], [a10, a11, a12]]], dtype=torch.float32, device=x_bchw.device)
        if x_in.shape[0] > 1:
            theta = theta.expand(x_in.shape[0], -1, -1)
        grid = F.affine_grid(theta, size=list(x_in.shape), align_corners=True)
        out = F.grid_sample(x_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        if b_vec is not None:
            out = out + b_vec
    else:
        out = x_bchw.clone()

    if hflip:
        out = torch.flip(out, dims=[3])
    if vflip:
        out = torch.flip(out, dims=[2])
    return out


def _gpu_invert_pred_and_flow(
    probs_1chw: torch.Tensor,
    flow_12hw: torch.Tensor,
    angle_deg: float,
    hflip: bool,
    vflip: bool,
    center: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact GPU equivalent of invert_rotate_flip_predictions."""
    p = probs_1chw
    f = flow_12hw.clone()
    if vflip:
        p = torch.flip(p, dims=[2])
        f = torch.flip(f, dims=[2])
        f[:, 1] = -f[:, 1]
    if hflip:
        p = torch.flip(p, dims=[3])
        f = torch.flip(f, dims=[3])
        f[:, 0] = -f[:, 0]
    if abs(angle_deg) > 1e-5:
        pf = torch.cat([p, f], dim=1)
        pf_rot = _gpu_rotate_flip_forward(pf, -float(angle_deg), False, False, center, border_vals=None)
        p = pf_rot[:, :3]
        fx, fy = pf_rot[:, 3], pf_rot[:, 4]
        rad_inv = math.radians(-float(angle_deg))
        ca, sa = math.cos(rad_inv), math.sin(rad_inv)
        f = torch.stack([ca * fx + sa * fy, -sa * fx + ca * fy], dim=1)
    return p[0], f[0]


@torch.no_grad()
def predict_single_image(
    model: TopoFlowNet,
    phys_5ch: np.ndarray,
    disk_info: dict[str, float],
    device: torch.device,
    annot_tokens: list[tuple[int, float]],
    fast_val: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict (body_prob, spine_prob, sep_prob, flow_xy, valid_mask) at 2048x2048.

    In full mode (`fast_val=False`), fuses THREE complementary equivariant regimes entirely on GPU:
      1. Native 2048x2048 Full-Disk 6-View Rotational+D4 TTA (0, 90, 180, 270 + +/-45 deg!)
      2. Native 768x768 Overlapping Gaussian-Weighted Sliding Window (stride 426, 4x4=16 tiles, 2-flip TTA)
      3. Macro 1024x1024 Full-Disk Context Pass (4-view TTA)
    """
    use_amp = device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    fit = DiskFit(cx=disk_info["cx"], cy=disk_info["cy"], r=disk_info["r"], n_inliers=720, rms=0.5)
    valid = disk_mask(fit, margin_px=8.0)
    center_2048 = (fit.cx, fit.cy)

    def _eval_heads_weighted(feat_1x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b_sz = feat_1x.shape[0]
        p_acc = feat_1x.new_zeros((b_sz, 3, feat_1x.shape[2], feat_1x.shape[3]), dtype=torch.float32)
        f_acc = feat_1x.new_zeros((b_sz, 2, feat_1x.shape[2], feat_1x.shape[3]), dtype=torch.float32)
        w_sum = 0.0
        for tok, tw in annot_tokens:
            a_ids = torch.full((b_sz,), tok, dtype=torch.long, device=device)
            l3, f2 = model.decode_heads(feat_1x, a_ids)
            p_acc += tw * torch.sigmoid(l3.float())
            f_acc += tw * f2.float()
            w_sum += tw
        return p_acc / w_sum, f_acc / w_sum

    x_gpu = torch.from_numpy(phys_5ch).unsqueeze(0).to(device, dtype=torch.float32)

    if fast_val:
        with amp_ctx:
            if device.type != "cuda":
                x_512 = F.interpolate(x_gpu, size=(512, 512), mode="area")
                pm, fm = _eval_heads_weighted(model.forward_features_1x(x_512))
                p_tot = F.interpolate(pm, size=(H, W), mode="bilinear", align_corners=False)[0].cpu().numpy()
                f_tot = F.interpolate(fm, size=(H, W), mode="bilinear", align_corners=False)[0].cpu().numpy()
                p_tot[:, ~valid] = 0.0
                f_tot[:, ~valid] = 0.0
                return p_tot[0], p_tot[1], p_tot[2], f_tot, valid

            p0, f0 = _eval_heads_weighted(model.forward_features_1x(x_gpu))
            x_flip = torch.flip(x_gpu, dims=[3])
            p1, f1 = _eval_heads_weighted(model.forward_features_1x(x_flip))
            p_2048 = 0.5 * (p0[0] + torch.flip(p1[0], dims=[2]))
            f1_inv = torch.flip(f1[0], dims=[2])
            f1_inv[0] = -f1_inv[0]
            f_2048 = 0.5 * (f0[0] + f1_inv)

            x_1024 = F.interpolate(x_gpu, size=(1024, 1024), mode="area")
            pm, fm = _eval_heads_weighted(model.forward_features_1x(x_1024))
            pm_up = F.interpolate(pm, size=(H, W), mode="bilinear", align_corners=False)[0]
            fm_up = F.interpolate(fm, size=(H, W), mode="bilinear", align_corners=False)[0]

        p_tot = (0.75 * p_2048 + 0.25 * pm_up).cpu().numpy()
        f_tot = (0.75 * f_2048 + 0.25 * fm_up).cpu().numpy()
        p_tot[:, ~valid] = 0.0
        f_tot[:, ~valid] = 0.0
        return p_tot[0], p_tot[1], p_tot[2], f_tot, valid

    # --- FULL EQUIVARIANT MULTI-SCALE + SLIDING-WINDOW TTA ON DEVICE ---
    # 1. Full-2048 6-view Rotational + Flip TTA (batched by 2 on GPU!)
    views_2048 = [
        (0.0, False, False),
        (90.0, False, False),
        (180.0, False, False),
        (270.0, False, False),
        (45.0, True, False),
        (-45.0, False, True),
    ]
    border_5 = [-0.4, 0.0, 0.0, 0.0, 0.0]
    p_full_acc = torch.zeros((3, H, W), dtype=torch.float32, device=device)
    f_full_acc = torch.zeros((2, H, W), dtype=torch.float32, device=device)

    step_2048 = 2 if device.type == "cuda" else 1
    for v0 in range(0, len(views_2048), step_2048):
        sub_views = views_2048[v0 : v0 + step_2048]
        warped_batch = torch.cat(
            [_gpu_rotate_flip_forward(x_gpu, ang, hf, vf, center_2048, border_5) for ang, hf, vf in sub_views],
            dim=0,
        )
        with amp_ctx:
            pv_b, fv_b = _eval_heads_weighted(model.forward_features_1x(warped_batch))
        for k_v, (ang, hf, vf) in enumerate(sub_views):
            pi, fi = _gpu_invert_pred_and_flow(pv_b[k_v : k_v + 1], fv_b[k_v : k_v + 1], ang, hf, vf, center_2048)
            p_full_acc += pi
            f_full_acc += fi
    p_full_acc *= 1.0 / len(views_2048)
    f_full_acc *= 1.0 / len(views_2048)

    # 2. Native 768x768 Overlapping Gaussian Sliding-Window Inference (4x4 = 16 tiles, 2-view flip TTA)
    tile = 768
    coords = [0, 426, 853, 1280]
    gw_gpu = torch.from_numpy(_make_gaussian_tile_weight(tile)).to(device=device, dtype=torch.float32)
    p_sw_acc = torch.zeros((3, H, W), dtype=torch.float32, device=device)
    f_sw_acc = torch.zeros((2, H, W), dtype=torch.float32, device=device)
    w_sw_acc = torch.zeros((H, W), dtype=torch.float32, device=device)

    positions: list[tuple[int, int]] = []
    tiles_gpu: list[torch.Tensor] = []
    for y0 in coords:
        for x0 in coords:
            if not np.any(valid[y0 : y0 + tile, x0 : x0 + tile]):
                continue
            tiles_gpu.append(x_gpu[0, :, y0 : y0 + tile, x0 : x0 + tile])
            positions.append((y0, x0))

    if tiles_gpu:
        batch_tiles = torch.stack(tiles_gpu, dim=0)
        with amp_ctx:
            for b0 in range(0, len(positions), 8):
                bt = batch_tiles[b0 : b0 + 8]
                pt1, ft1 = _eval_heads_weighted(model.forward_features_1x(bt))
                bt_f = torch.flip(bt, dims=[3])
                pt2, ft2 = _eval_heads_weighted(model.forward_features_1x(bt_f))
                pt2 = torch.flip(pt2, dims=[3])
                ft2 = torch.flip(ft2, dims=[3])
                ft2[:, 0] = -ft2[:, 0]
                pt_m = 0.5 * (pt1 + pt2)
                ft_m = 0.5 * (ft1 + ft2)
                for k_i, (y0, x0) in enumerate(positions[b0 : b0 + 8]):
                    p_sw_acc[:, y0 : y0 + tile, x0 : x0 + tile] += pt_m[k_i] * gw_gpu
                    f_sw_acc[:, y0 : y0 + tile, x0 : x0 + tile] += ft_m[k_i] * gw_gpu
                    w_sw_acc[y0 : y0 + tile, x0 : x0 + tile] += gw_gpu

        covered = w_sw_acc > 1e-5
        p_sw_acc[:, covered] /= w_sw_acc[covered]
        f_sw_acc[:, covered] /= w_sw_acc[covered]

    # 3. Macro 1024x1024 Full-Disk Context Pass (4-view TTA in 1 GPU batch)
    x_1024 = F.interpolate(x_gpu, size=(1024, 1024), mode="area")
    views_1024 = [(0.0, False, False), (90.0, True, False), (180.0, False, False), (270.0, False, True)]
    c_1024 = (fit.cx * 0.5, fit.cy * 0.5)
    bt1024 = torch.cat(
        [_gpu_rotate_flip_forward(x_1024, ang, hf, vf, c_1024, border_5) for ang, hf, vf in views_1024],
        dim=0,
    )
    with amp_ctx:
        pm_all, fm_all = _eval_heads_weighted(model.forward_features_1x(bt1024))

    p_mac_acc = torch.zeros((3, 1024, 1024), dtype=torch.float32, device=device)
    f_mac_acc = torch.zeros((2, 1024, 1024), dtype=torch.float32, device=device)
    for idx_v, (ang, hf, vf) in enumerate(views_1024):
        pi, fi = _gpu_invert_pred_and_flow(pm_all[idx_v : idx_v + 1], fm_all[idx_v : idx_v + 1], ang, hf, vf, c_1024)
        p_mac_acc += pi
        f_mac_acc += fi
    p_mac_acc *= 1.0 / len(views_1024)
    f_mac_acc *= 1.0 / len(views_1024)

    p_mac_up = F.interpolate(p_mac_acc.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    f_mac_up = F.interpolate(f_mac_acc.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]

    # Fuse: 45% Full-2048 RotTTA + 35% Sliding-768 Gaussian + 20% Macro-1024
    p_blend = (0.45 * p_full_acc + 0.35 * p_sw_acc + 0.20 * p_mac_up).cpu().numpy()
    f_blend = (0.45 * f_full_acc + 0.35 * f_sw_acc + 0.20 * f_mac_up).cpu().numpy()
    p_blend[:, ~valid] = 0.0
    f_blend[:, ~valid] = 0.0
    return p_blend[0], p_blend[1], p_blend[2], f_blend, valid


def save_prediction_npz(
    out_path: Path,
    body_prob: np.ndarray,
    spine_prob: np.ndarray,
    sep_prob: np.ndarray,
    flow_xy: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        body=np.clip(np.round(body_prob * 255.0), 0, 255).astype(np.uint8),
        spine=np.clip(np.round(spine_prob * 255.0), 0, 255).astype(np.uint8),
        sep=np.clip(np.round(sep_prob * 255.0), 0, 255).astype(np.uint8),
        vx=np.clip(np.round(flow_xy[0] * 127.0), -127, 127).astype(np.int8),
        vy=np.clip(np.round(flow_xy[1] * 127.0), -127, 127).astype(np.int8),
    )
    tmp.replace(out_path)


def load_prediction_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path) as d:
        body = d["body"].astype(np.float32) * (1.0 / 255.0)
        spine = d["spine"].astype(np.float32) * (1.0 / 255.0)
        sep = d["sep"].astype(np.float32) * (1.0 / 255.0)
        vx = d["vx"].astype(np.float32) * (1.0 / 127.0)
        vy = d["vy"].astype(np.float32) * (1.0 / 127.0)
    return body, spine, sep, np.stack([vx, vy], axis=0)


def rles_to_pred_df(preds_rle: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for stem in sorted(preds_rle):
        for k, rle in enumerate(preds_rle[stem], 1):
            if rle:
                rows.append((f"{stem}_{k}", rle))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def decode_multi_hypothesis_mbr_image(
    body: np.ndarray,
    spine: np.ndarray,
    sep: np.ndarray,
    flow: np.ndarray,
    valid: np.ndarray,
    t_body_base: float = 0.40,
    score_gate: float = 0.72,
    tau_pq: float = 0.47,
    fp_cost_scale: float = 1.0,
    keep_masks: bool = True,
) -> list:
    """Decode an image via Multi-Hypothesis Candidate Pool (Split + Balanced + Bridged) + PQ-MBR MWIS!"""
    # Regime 1: Split-sensitive (separates tight neighboring filaments cleanly)
    c_split = decode_topoflow(
        body, spine, sep, flow, valid,
        t_body=t_body_base + 0.04, t_seed=0.52, alpha_sep=0.90, spine_bridge_thr=0.72,
        min_area=180, score_gate=score_gate - 0.05, source_tag="split", keep_masks=keep_masks,
    )
    # Regime 2: Balanced Topological Spine-Flow
    c_bal = decode_topoflow(
        body, spine, sep, flow, valid,
        t_body=t_body_base, t_seed=0.48, alpha_sep=0.78, spine_bridge_thr=0.56,
        min_area=180, score_gate=score_gate - 0.05, source_tag="bal", keep_masks=keep_masks,
    )
    # Regime 3: Bridge-sensitive (heals optically faint gaps along strong human-like spines)
    c_bridge = decode_topoflow(
        body, spine, sep, flow, valid,
        t_body=max(0.28, t_body_base - 0.05), t_seed=0.46, alpha_sep=0.68, spine_bridge_thr=0.44,
        min_area=200, score_gate=score_gate - 0.03, source_tag="bridge", keep_masks=keep_masks,
    )
    # High-confidence reference parses for MBR marginal utility estimation
    r_split = [c.rle_counts for c in c_split if c.score >= score_gate]
    r_bal = [c.rle_counts for c in c_bal if c.score >= score_gate]
    r_bridge = [c.rle_counts for c in c_bridge if c.score >= score_gate + 0.02]

    pool = c_bal + c_split + c_bridge
    refs = [(r_bal, 0.50), (r_split, 0.28), (r_bridge, 0.22)]
    return pq_mbr_solve(pool, refs, tau_pq=tau_pq, fp_cost_scale=fp_cost_scale)


def _quantize_maps_for_npz(
    body_prob: np.ndarray,
    spine_prob: np.ndarray,
    sep_prob: np.ndarray,
    flow_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.clip(np.round(body_prob * 255.0), 0, 255).astype(np.uint8),
        np.clip(np.round(spine_prob * 255.0), 0, 255).astype(np.uint8),
        np.clip(np.round(sep_prob * 255.0), 0, 255).astype(np.uint8),
        np.clip(np.round(flow_xy[0] * 127.0), -127, 127).astype(np.int8),
        np.clip(np.round(flow_xy[1] * 127.0), -127, 127).astype(np.int8),
    )


def _calib_extract_worker(job: tuple) -> tuple[str, dict]:
    stem, bp_u8, sp_u8, sep_u8, vx_i8, vy_i8, disk_info, grid_tb, min_sg, is_smoke = job
    bp = bp_u8.astype(np.float32) * (1.0 / 255.0)
    sp = sp_u8.astype(np.float32) * (1.0 / 255.0)
    sepp = sep_u8.astype(np.float32) * (1.0 / 255.0)
    flp = np.stack([vx_i8.astype(np.float32) * (1.0 / 127.0), vy_i8.astype(np.float32) * (1.0 / 127.0)], axis=0)
    fit = DiskFit(cx=disk_info["cx"], cy=disk_info["cy"], r=disk_info["r"], n_inliers=720, rms=0.5)
    vm = disk_mask(fit, margin_px=8.0)

    min_area = 20 if is_smoke else 180
    min_area_br = 20 if is_smoke else 200
    g_lo = max(0.0, float(min_sg) - 0.05)

    by_tb: dict[float, dict[str, list]] = {}
    for tb in grid_tb:
        c_single = decode_topoflow(
            bp, sp, sepp, flp, vm, t_body=tb, t_seed=0.48, min_area=min_area, score_gate=float(min_sg), keep_masks=False
        )
        c_split = decode_topoflow(
            bp, sp, sepp, flp, vm, t_body=tb + 0.04, t_seed=0.52, alpha_sep=0.90, spine_bridge_thr=0.72,
            min_area=min_area, score_gate=g_lo, source_tag="split", keep_masks=False,
        )
        c_bal = decode_topoflow(
            bp, sp, sepp, flp, vm, t_body=tb, t_seed=0.48, alpha_sep=0.78, spine_bridge_thr=0.56,
            min_area=min_area, score_gate=g_lo, source_tag="bal", keep_masks=False,
        )
        c_bridge = decode_topoflow(
            bp, sp, sepp, flp, vm, t_body=max(0.28, tb - 0.05), t_seed=0.46, alpha_sep=0.68, spine_bridge_thr=0.44,
            min_area=min_area_br, score_gate=max(0.0, float(min_sg) - 0.03), source_tag="bridge", keep_masks=False,
        )
        by_tb[float(tb)] = {"single": c_single, "split": c_split, "bal": c_bal, "bridge": c_bridge}
    return stem, by_tb


def _save_and_decode_worker(job: tuple) -> tuple[str, list[str]]:
    stem, out_npz_str, bp_u8, sp_u8, sep_u8, vx_i8, vy_i8, disk_info, best_cal, is_smoke = job
    out_p = Path(out_npz_str)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_name(out_p.name + ".tmp.npz")
    np.savez_compressed(tmp, body=bp_u8, spine=sp_u8, sep=sep_u8, vx=vx_i8, vy=vy_i8)
    tmp.replace(out_p)

    bp = bp_u8.astype(np.float32) * (1.0 / 255.0)
    sp = sp_u8.astype(np.float32) * (1.0 / 255.0)
    sepp = sep_u8.astype(np.float32) * (1.0 / 255.0)
    flp = np.stack([vx_i8.astype(np.float32) * (1.0 / 127.0), vy_i8.astype(np.float32) * (1.0 / 127.0)], axis=0)
    fit = DiskFit(cx=disk_info["cx"], cy=disk_info["cy"], r=disk_info["r"], n_inliers=720, rms=0.5)
    vm = disk_mask(fit, margin_px=8.0)

    if best_cal["mode"] == "mbr":
        cs = decode_multi_hypothesis_mbr_image(
            bp, sp, sepp, flp, vm,
            t_body_base=best_cal["t_body"],
            score_gate=best_cal["score_gate"],
            tau_pq=0.47,
            fp_cost_scale=best_cal["fp_cost_scale"],
            keep_masks=False,
        )
    else:
        cs = decode_topoflow(
            bp, sp, sepp, flp, vm,
            t_body=best_cal["t_body"],
            t_seed=0.48,
            min_area=20 if is_smoke else 180,
            score_gate=best_cal["score_gate"],
            keep_masks=False,
        )
    return stem, [c.rle_counts for c in cs if c.rle_counts]


def select_representative_annot_tokens(train_sets: list[AnnotationSet]) -> list[tuple[int, float]]:
    """Select Consensus Token (0) + top-4 most active experienced MAGFiLO annotators from train_sets."""
    counts: dict[int, int] = {}
    for s in train_sets:
        counts[s.annotator_idx] = counts.get(s.annotator_idx, 0) + 1
    top4 = [idx for idx, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    tokens = [(0, 0.60)]
    if top4:
        w_each = 0.40 / len(top4)
        for idx in top4:
            tokens.append((int(idx), w_each))
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "1_data/raw/MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--assets", type=Path, default=ROOT / "1_data/processed")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outer-fold", type=int, default=0)
    ap.add_argument("--encoder", type=str, default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--batch-crop", type=int, default=12)
    ap.add_argument("--batch-full", type=int, default=8)
    ap.add_argument("--crops-per-set", type=int, default=6)
    ap.add_argument("--lr-enc", type=float, default=1.5e-4)
    ap.add_argument("--lr-dec", type=float, default=6.0e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--infer-only", action="store_true", help="Skip training and run fast GPU TTA + parallel decode from existing best.pt")
    args = ap.parse_args()

    t_start = time.time()
    set_seed(args.seed + max(args.outer_fold, 0))
    device = torch.device(args.device)
    if device.type == "cuda":
        # Avoid 75s per-shape cuDNN 9 exhaustive autotune stalls on multi-resolution batches
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")

    args.out.mkdir(parents=True, exist_ok=True)
    train_json = args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_img_dir = args.data_root / "train/train_images"
    test_img_dir = args.data_root / "test/test_images"

    folds_df = pd.read_csv(args.assets / "folds_cvA_v1.csv")
    gt_df = pd.read_csv(args.assets / "gt_train_v1.csv")
    gt_df["stem"] = gt_df["filament_id"].str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0]
    geom_path = args.assets / "disk_geometry_v1.csv"
    geom_df = pd.read_csv(geom_path) if geom_path.is_file() else None

    all_sets, prefix_to_idx = load_magfilo_annotations(train_json)
    train_stems, inner_stems, outer_stems = make_fold_splits(folds_df, args.outer_fold, seed=args.seed)
    all_test_stems = sorted(p.stem for p in test_img_dir.glob("*.jpeg"))

    if args.smoke:
        train_stems = train_stems[:2]
        inner_stems = inner_stems[:1]
        outer_stems = outer_stems[:1]
        test_stems = all_test_stems[:2]
        epochs = 1
        eval_every = 1
        crops_per_set = 1
        batch_crop = 1
        batch_full = 1
        crop_sz = 256
        full_sz = 512
        enc_name = "convnext_tiny"
        pretrained = False
    else:
        test_stems = all_test_stems
        epochs = args.epochs
        eval_every = args.eval_every
        crops_per_set = args.crops_per_set
        batch_crop = args.batch_crop
        batch_full = args.batch_full
        crop_sz = 768
        full_sz = 1024
        enc_name = args.encoder
        pretrained = True

    train_stem_set = set(train_stems)
    train_sets = [s for s in all_sets if s.stem in train_stem_set]
    needed_train_stems = sorted(set(train_stems) | set(inner_stems) | set(outer_stems))

    print(
        f"[E120] fold={args.outer_fold} smoke={args.smoke} train_imgs={len(train_stems)} ({len(train_sets)} sets) "
        f"inner_val={len(inner_stems)} outer_oof={len(outer_stems)} test={len(test_stems)} device={device}",
        flush=True,
    )
    atomic_write_json(
        args.out / "splits.json",
        {
            "outer_fold": args.outer_fold,
            "smoke": args.smoke,
            "n_train_stems": len(train_stems),
            "n_train_sets": len(train_sets),
            "n_inner_stems": len(inner_stems),
            "n_outer_stems": len(outer_stems),
            "n_test_stems": len(test_stems),
            "inner_stems": inner_stems,
            "outer_stems": outer_stems,
        },
    )

    disk_meta = prepare_nvme_cache(
        args.cache_dir,
        train_img_dir,
        test_img_dir,
        geom_df,
        [] if args.infer_only else train_sets,
        sorted(set(inner_stems) | set(outer_stems)) if args.infer_only else needed_train_stems,
        test_stems,
        n_procs=1 if args.smoke else max(2, min(8, os.cpu_count() or 4)),
    )

    annot_tokens = select_representative_annot_tokens(train_sets)
    gt_inner = gt_df[gt_df["stem"].isin(set(inner_stems))][["filament_id", "segmentation_rle"]].copy()
    phys_dir = args.cache_dir / "phys"

    ema_model = TopoFlowNet(
        encoder_name=enc_name,
        pretrained=False,
        in_channels=5,
        num_annotators=len(prefix_to_idx) + 2,
    ).to(device)
    if device.type == "cuda":
        ema_model = ema_model.to(memory_format=torch.channels_last)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    if not args.infer_only:
        ds_crop = TopoCropDataset(train_sets, args.cache_dir, disk_meta, crop_size=crop_sz, crops_per_set=crops_per_set)
        ds_full = TopoFullDiskDataset(train_sets, args.cache_dir, disk_meta, full_size=full_sz)

        dl_crop = DataLoader(
            ds_crop,
            batch_size=batch_crop,
            shuffle=True,
            num_workers=0 if args.smoke else args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=not args.smoke,
            persistent_workers=(not args.smoke and args.workers > 0),
        )
        dl_full = DataLoader(
            ds_full,
            batch_size=batch_full,
            shuffle=True,
            num_workers=0 if args.smoke else max(1, args.workers // 2),
            pin_memory=(device.type == "cuda"),
            drop_last=not args.smoke,
            persistent_workers=(not args.smoke and args.workers > 0),
        )

        model = TopoFlowNet(
            encoder_name=enc_name,
            pretrained=pretrained,
            in_channels=5,
            num_annotators=len(prefix_to_idx) + 2,
        ).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        ema_model.load_state_dict(model.state_dict())

        enc_params = list(model.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": args.lr_enc},
                {"params": dec_params, "lr": args.lr_dec},
            ],
            weight_decay=1e-2,
        )
        criterion = TopoFlowLoss().to(device)

        steps_per_epoch = len(dl_crop) + len(dl_full)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = max(1, int(0.08 * total_steps))

        def adjust_lr(step_idx: int) -> float:
            if step_idx < warmup_steps:
                mult = float(step_idx + 1) / float(warmup_steps)
            else:
                prog = float(step_idx - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
                mult = 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
            optimizer.param_groups[0]["lr"] = args.lr_enc * mult
            optimizer.param_groups[1]["lr"] = args.lr_dec * mult
            return mult

        @torch.no_grad()
        def update_ema(step_idx: int) -> None:
            decay = min(0.996, (1.0 + step_idx) / (12.0 + step_idx))
            for p_ema, p_src in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)

        use_amp = device.type == "cuda"
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

        global_step = 0
        best_inner_pq = -1.0
        best_epoch = 0
        log_rows: list[dict] = []

        for epoch in range(1, epochs + 1):
            model.train()
            ep_t0 = time.time()
            it_crop = iter(dl_crop)
            it_full = iter(dl_full)
            ratio = max(1, len(dl_crop) // max(1, len(dl_full)))

            sum_metrics = {"loss": 0.0, "body": 0.0, "spine": 0.0, "sep": 0.0, "flow": 0.0}
            n_batches = 0
            last_beat = time.time()

            while True:
                batches_this_round = []
                for _ in range(ratio):
                    try:
                        batches_this_round.append(next(it_crop))
                    except StopIteration:
                        break
                try:
                    batches_this_round.append(next(it_full))
                except StopIteration:
                    pass
                if not batches_this_round:
                    break

                for xb, mb, fb, ab in batches_this_round:
                    lr_mult = adjust_lr(global_step)
                    xb = xb.to(device, non_blocking=True)
                    if device.type == "cuda":
                        xb = xb.to(memory_format=torch.channels_last)
                    mb = mb.to(device, non_blocking=True)
                    fb = fb.to(device, non_blocking=True)
                    ab = ab.to(device, non_blocking=True)

                    wb = compute_gpu_fg_weight(mb[:, 0], mb[:, 2])
                    optimizer.zero_grad(set_to_none=True)
                    with amp_ctx:
                        logits_3, flow_2 = model(xb, ab)
                        loss, m_dict = criterion(logits_3, flow_2, mb[:, 0], mb[:, 1], mb[:, 2], fb, wb)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss encountered at epoch={epoch} step={global_step}: {m_dict}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    update_ema(global_step)
                    global_step += 1
                    n_batches += 1
                    for k in sum_metrics:
                        sum_metrics[k] += m_dict[k]

                    now_t = time.time()
                    if now_t - last_beat >= 10.0:
                        cur_its = n_batches / max(now_t - ep_t0, 1e-3)
                        cur_l = sum_metrics["loss"] / n_batches
                        print(
                            f"[STEP ep={epoch:02d}/{epochs:02d} b={n_batches}/{steps_per_epoch}] "
                            f"loss={cur_l:.4f} (cur={m_dict['loss']:.4f}) it/s={cur_its:.2f}",
                            flush=True,
                        )
                        last_beat = now_t

                    if args.smoke and n_batches >= 2:
                        break
                if args.smoke and n_batches >= 2:
                    break

            ep_sec = time.time() - ep_t0
            it_s = n_batches / max(ep_sec, 1e-3)
            avg_m = {k: sum_metrics[k] / max(1, n_batches) for k in sum_metrics}

            inner_pq = float("nan")
            if epoch == epochs or (epoch >= max(4, epochs // 3) and epoch % eval_every == 0) or args.smoke:
                ema_model.eval()
                val_rles_by_gate: dict[float, dict[str, list[str]]] = {g: {} for g in (0.65, 0.72, 0.78)}
                v_beat = time.time()
                for v_idx, stem in enumerate(inner_stems, 1):
                    ph = np.array(np.load(phys_dir / f"{stem}.npy", mmap_mode="r"), dtype=np.float32)
                    b_p, s_p, sep_p, fl_p, v_m = predict_single_image(
                        ema_model, ph, disk_meta[stem], device, annot_tokens, fast_val=True
                    )
                    cands = decode_topoflow(
                        b_p, s_p, sep_p, fl_p, v_m,
                        t_body=0.40, t_seed=0.48, min_area=180 if not args.smoke else 20, score_gate=0.0, keep_masks=False,
                    )
                    for g in val_rles_by_gate:
                        val_rles_by_gate[g][stem] = [c.rle_counts for c in cands if c.score >= g]
                    if time.time() - v_beat >= 10.0:
                        print(f"[VAL ep={epoch:02d}] {v_idx}/{len(inner_stems)} inner images inferred", flush=True)
                        v_beat = time.time()

                best_ep_pq = -1.0
                for g, r_dict in val_rles_by_gate.items():
                    res = evaluate(gt_inner, rles_to_pred_df(r_dict), distributions=False)
                    if res.pq > best_ep_pq:
                        best_ep_pq = res.pq
                inner_pq = best_ep_pq
                if inner_pq >= best_inner_pq:
                    best_inner_pq = inner_pq
                    best_epoch = epoch
                    atomic_save_torch(
                        args.out / "best.pt",
                        {
                            "epoch": epoch,
                            "inner_pq": best_inner_pq,
                            "encoder": enc_name,
                            "num_annotators": len(prefix_to_idx) + 2,
                            "annot_tokens": annot_tokens,
                            "state_dict": ema_model.state_dict(),
                        },
                    )

            atomic_save_torch(
                args.out / "last.pt",
                {
                    "epoch": epoch,
                    "inner_pq": inner_pq,
                    "encoder": enc_name,
                    "num_annotators": len(prefix_to_idx) + 2,
                    "annot_tokens": annot_tokens,
                    "state_dict": ema_model.state_dict(),
                },
            )

            row = {
                "epoch": epoch,
                "steps": global_step,
                "sec": round(ep_sec, 1),
                "it_s": round(it_s, 2),
                "loss": round(avg_m["loss"], 4),
                "body": round(avg_m["body"], 4),
                "spine": round(avg_m["spine"], 4),
                "sep": round(avg_m["sep"], 4),
                "flow": round(avg_m["flow"], 4),
                "inner_pq": round(inner_pq, 5) if math.isfinite(inner_pq) else "",
                "best_inner_pq": round(best_inner_pq, 5),
            }
            log_rows.append(row)
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(log_rows)

            print(
                f"[EP {epoch:02d}/{epochs:02d}] loss={row['loss']:.4f} (b={row['body']:.3f} sp={row['spine']:.3f} "
                f"sep={row['sep']:.3f} fl={row['flow']:.3f}) | {ep_sec:.1f}s ({it_s:.2f} it/s) | "
                f"inner_pq={row['inner_pq']} best={best_inner_pq:.4f}@ep{best_epoch}",
                flush=True,
            )
        del model, optimizer, dl_crop, dl_full
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Load Best EMA Checkpoint & Run Fast GPU TTA + Parallel Multiprocess Decode ----
    best_ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    if "annot_tokens" in best_ckpt and best_ckpt["annot_tokens"]:
        annot_tokens = [(int(t), float(w)) for t, w in best_ckpt["annot_tokens"]]
    ema_model.load_state_dict(best_ckpt["state_dict"])
    ema_model.eval()
    print(f"[INFER] Loaded best.pt from epoch={best_ckpt['epoch']} (fast inner_pq={best_ckpt['inner_pq']:.4f})", flush=True)

    n_post_procs = 1 if args.smoke else max(2, min(12, os.cpu_count() or 4))

    # 1. Calibrate decoding & MBR thresholds strictly on inner_stems using full GPU TTA + parallel candidate extraction!
    sel_path = args.out / "selection.json"
    if args.infer_only and sel_path.is_file() and not args.smoke:
        best_cal = json.loads(sel_path.read_text(encoding="utf-8"))["calibration"]
        print(f"[CALIB] Reusing existing selection.json calibration: {best_cal}", flush=True)
    else:
        best_cal = {"pq": -1.0, "mode": "mbr", "t_body": 0.40, "score_gate": 0.78, "fp_cost_scale": 1.0}
        grid_tb = (0.38, 0.42) if not args.smoke else (0.40,)
        grid_sg = (0.72, 0.78, 0.83, 0.86) if not args.smoke else (0.50,)
        grid_fpc = (0.75, 1.00, 1.25) if not args.smoke else (1.00,)
        min_sg = min(grid_sg)

        t_inf0 = time.time()
        inf_beat = time.time()
        inner_cands: dict[str, dict[float, dict[str, list]]] = {}

        if n_post_procs <= 1:
            for idx_i, stem in enumerate(inner_stems, 1):
                ph = np.array(np.load(phys_dir / f"{stem}.npy", mmap_mode="r"), dtype=np.float32)
                bp, sp, sepp, flp, _ = predict_single_image(
                    ema_model, ph, disk_meta[stem], device, annot_tokens, fast_val=args.smoke
                )
                q5 = _quantize_maps_for_npz(bp, sp, sepp, flp)
                _, by_tb = _calib_extract_worker((stem, *q5, disk_meta[stem], grid_tb, min_sg, args.smoke))
                inner_cands[stem] = by_tb
                if time.time() - inf_beat >= 10.0:
                    print(f"[CALIB-TTA] {idx_i}/{len(inner_stems)} inner images inferred", flush=True)
                    inf_beat = time.time()
        else:
            with Pool(processes=n_post_procs) as pool:
                pending = []
                for idx_i, stem in enumerate(inner_stems, 1):
                    ph = np.array(np.load(phys_dir / f"{stem}.npy", mmap_mode="r"), dtype=np.float32)
                    bp, sp, sepp, flp, _ = predict_single_image(
                        ema_model, ph, disk_meta[stem], device, annot_tokens, fast_val=args.smoke
                    )
                    q5 = _quantize_maps_for_npz(bp, sp, sepp, flp)
                    pending.append(pool.apply_async(_calib_extract_worker, ((stem, *q5, disk_meta[stem], grid_tb, min_sg, args.smoke),)))
                    while len(pending) >= n_post_procs * 2:
                        st_done, by_tb_done = pending.pop(0).get()
                        inner_cands[st_done] = by_tb_done
                    if time.time() - inf_beat >= 10.0:
                        print(f"[CALIB-TTA] {idx_i}/{len(inner_stems)} inner images inferred", flush=True)
                        inf_beat = time.time()
                for fut in pending:
                    st_done, by_tb_done = fut.get()
                    inner_cands[st_done] = by_tb_done

        # Fast RLE-only calibration sweep (~1.5s total instead of 25 min!)
        for tb in grid_tb:
            tb_f = float(tb)
            for sg in grid_sg:
                sg_f = float(sg)
                p_single_rle = {
                    stem: [c.rle_counts for c in d[tb_f]["single"] if c.score >= sg_f]
                    for stem, d in inner_cands.items()
                }
                r_s = evaluate(gt_inner, rles_to_pred_df(p_single_rle), distributions=False)
                if r_s.pq > best_cal["pq"]:
                    best_cal = {"pq": float(r_s.pq), "mode": "single", "t_body": tb_f, "score_gate": sg_f, "fp_cost_scale": 1.0}

                for fpc in grid_fpc:
                    fpc_f = float(fpc)
                    p_mbr_rle = {}
                    for stem, d in inner_cands.items():
                        c_split = [c for c in d[tb_f]["split"] if c.score >= sg_f - 0.05]
                        c_bal = [c for c in d[tb_f]["bal"] if c.score >= sg_f - 0.05]
                        c_bridge = [c for c in d[tb_f]["bridge"] if c.score >= sg_f - 0.03]
                        r_split = [c.rle_counts for c in c_split if c.score >= sg_f]
                        r_bal = [c.rle_counts for c in c_bal if c.score >= sg_f]
                        r_bridge = [c.rle_counts for c in c_bridge if c.score >= sg_f + 0.02]
                        solved = pq_mbr_solve(
                            c_bal + c_split + c_bridge,
                            [(r_bal, 0.50), (r_split, 0.28), (r_bridge, 0.22)],
                            tau_pq=0.47,
                            fp_cost_scale=fpc_f,
                        )
                        p_mbr_rle[stem] = [c.rle_counts for c in solved if c.rle_counts]
                    r_m = evaluate(gt_inner, rles_to_pred_df(p_mbr_rle), distributions=False)
                    if r_m.pq > best_cal["pq"]:
                        best_cal = {"pq": float(r_m.pq), "mode": "mbr", "t_body": tb_f, "score_gate": sg_f, "fp_cost_scale": fpc_f}

        atomic_write_json(sel_path, {"best_epoch": int(best_ckpt["epoch"]), "calibration": best_cal})
        print(f"[CALIB] Inner-val ({len(inner_stems)} imgs) calibrated in {time.time()-t_inf0:.1f}s: {best_cal}", flush=True)
        inner_cands.clear()

    def _run_pipelined_split(split_stems: list[str], out_dir: Path, tag: str) -> dict[str, list[str]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_rle: dict[str, list[str]] = {}
        beat = time.time()
        if n_post_procs <= 1:
            for idx_s, stem in enumerate(split_stems, 1):
                ph = np.array(np.load(phys_dir / f"{stem}.npy", mmap_mode="r"), dtype=np.float32)
                bp, sp, sepp, flp, _ = predict_single_image(
                    ema_model, ph, disk_meta[stem], device, annot_tokens, fast_val=args.smoke
                )
                q5 = _quantize_maps_for_npz(bp, sp, sepp, flp)
                _, rles = _save_and_decode_worker((stem, str(out_dir / f"{stem}.npz"), *q5, disk_meta[stem], best_cal, args.smoke))
                preds_rle[stem] = rles
                if time.time() - beat >= 10.0:
                    print(f"[{tag}] {idx_s}/{len(split_stems)} images inferred", flush=True)
                    beat = time.time()
        else:
            with Pool(processes=n_post_procs) as pool:
                pending = []
                for idx_s, stem in enumerate(split_stems, 1):
                    ph = np.array(np.load(phys_dir / f"{stem}.npy", mmap_mode="r"), dtype=np.float32)
                    bp, sp, sepp, flp, _ = predict_single_image(
                        ema_model, ph, disk_meta[stem], device, annot_tokens, fast_val=args.smoke
                    )
                    q5 = _quantize_maps_for_npz(bp, sp, sepp, flp)
                    pending.append(
                        pool.apply_async(
                            _save_and_decode_worker,
                            ((stem, str(out_dir / f"{stem}.npz"), *q5, disk_meta[stem], best_cal, args.smoke),),
                        )
                    )
                    while len(pending) >= n_post_procs * 2:
                        st_done, rles_done = pending.pop(0).get()
                        preds_rle[st_done] = rles_done
                    if time.time() - beat >= 10.0:
                        print(f"[{tag}] {idx_s}/{len(split_stems)} images inferred", flush=True)
                        beat = time.time()
                for fut in pending:
                    st_done, rles_done = fut.get()
                    preds_rle[st_done] = rles_done
        return preds_rle

    # 2. Outer OOF Inference & Evaluation (strictly untouched!)
    t_oof = time.time()
    oof_rles = _run_pipelined_split(outer_stems, args.out / "oof_maps", "OOF-TTA")
    oof_df = rles_to_pred_df(oof_rles)
    oof_df.to_csv(args.out / "oof_predictions.csv", index=False)
    outer_summary = {}
    if outer_stems:
        gt_outer = gt_df[gt_df["stem"].isin(set(outer_stems))][["filament_id", "segmentation_rle"]].copy()
        res_outer = evaluate(gt_outer, oof_df, distributions=True)
        outer_summary = res_outer.summary()
        outer_summary["PQ_exact"] = float(res_outer.pq)
        print(f"[OUTER OOF] Fold {args.outer_fold} ({len(outer_stems)} imgs, {time.time()-t_oof:.1f}s): {outer_summary}", flush=True)

    # 3. Test Set Inference (180 images!)
    t_test = time.time()
    test_rles = _run_pipelined_split(test_stems, args.out / "test_maps", "TEST-TTA")
    sub_path = args.out / f"sub_fold{args.outer_fold}.csv"
    sub_df = rles_to_pred_df(test_rles)
    validate_submission(sub_df, list(test_stems if args.smoke else all_test_stems))
    write_submission(sub_df, sub_path)
    test_sec = time.time() - t_test
    print(
        f"[TEST] Inferred {len(test_stems)} test images in {test_sec:.1f}s ({test_sec/max(1,len(test_stems)):.2f}s/img), "
        f"instances={sum(len(v) for v in test_rles.values())}",
        flush=True,
    )

    completed_ep = epochs
    if args.infer_only and (args.out / "train_log.csv").is_file():
        try:
            completed_ep = int(pd.read_csv(args.out / "train_log.csv")["epoch"].max())
        except Exception:
            completed_ep = epochs

    report = {
        "experiment": "E-120 TopoFlowNet",
        "smoke": args.smoke,
        "outer_fold": args.outer_fold,
        "completed_epochs": completed_ep,
        "best_epoch": int(best_ckpt["epoch"]),
        "inner_calibration": best_cal,
        "outer_oof_summary": outer_summary,
        "test_images_inferred": len(test_stems),
        "test_sec_per_image": round(test_sec / max(1, len(test_stems)), 3),
        "total_wall_seconds": round(time.time() - t_start, 1),
    }
    atomic_write_json(args.out / "report.json", report)
    print(f"[DONE] E120_COMPLETE fold={args.outer_fold} wall={report['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
