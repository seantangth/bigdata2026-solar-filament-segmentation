"""E-121a Trainer & Exact-D4 Native-Resolution Inferencer for Snake-TopoFlowNet (512 Specialist).

Complements E-120 (Global + Continuous-Rotation TopoFlowNet) to form a 100% Pure Dual-Scale
TopoFlowNet Ensemble (zero legacy smp.Unet, zero GBDT pickle, zero post-hoc dilation):
  1. Identical CV-A outer folds {0..4} (141-142 untouched outer OOF images per fold).
  2. Physical-Image Balanced Exposure (10 native 512x512 crops per physical image per epoch,
     fixing the 2.5x single-annotator under-exposure diagnosed on E-120 OOF).
  3. TopoFlowNet(use_snake=True): GroupNorm Dynamic Snake Convolutions on dec16/dec8/dec4 +
     anisotropic BarbStripResidual on dec2 + native stride-1/1 & 1/2 fine stem.
  4. Symmetric Boundary/Barb Band Weighting: weights both inner thin barbs and outer adjacent
     spicule pixels equally (2.25x), preserving Bayes-optimal p*=0.50 without rim fattening.
  5. Exact Lossless 8-View Dihedral D4 Full-2048 + Native 512x512 Gaussian Tile GPU Inference.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "3_src"))
sys.path.insert(0, str(ROOT / "3_src/scripts"))

from sfsc import H, W
from sfsc.geometry import DiskFit, disk_mask
from sfsc.metrics import evaluate
from sfsc.submission import validate_submission, write_submission
from sfsc.topoflow_net import TopoFlowLoss, TopoFlowNet
from sfsc.topoflow_targets import (
    AnnotationSet,
    decode_topoflow,
    load_magfilo_annotations,
    pq_mbr_solve,
    rotate_flip_tensor_and_flow,
)
from train_e120_topoflow import (
    _calib_extract_worker,
    _extract_padded_patch,
    _make_gaussian_tile_weight,
    _quantize_maps_for_npz,
    _save_and_decode_worker,
    _unpack_targets_u8,
    atomic_save_torch,
    atomic_write_json,
    make_fold_splits,
    prepare_nvme_cache,
    rles_to_pred_df,
    set_seed,
)


def exact_d4_tensor_and_flow(
    arr_chw: np.ndarray,
    flow_2hw: np.ndarray,
    k_rot90: int,
    hflip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Lossless exact-grid dihedral D4 transform (k * 90 deg CCW + optional horizontal flip).

    Matches `rotate_flip_tensor_and_flow(..., angle_deg=90*k, hflip=hflip, vflip=False)` bit-for-bit
    on square patches with ZERO bilinear interpolation blur on 1-2 px barbs!
    """
    k = int(k_rot90) % 4
    if k == 0:
        out = arr_chw.copy()
        vx, vy = flow_2hw[0].copy(), flow_2hw[1].copy()
    elif k == 1:
        out = np.rot90(arr_chw, k=1, axes=(1, 2)).copy()
        vx_r = np.rot90(flow_2hw[0], k=1, axes=(0, 1))
        vy_r = np.rot90(flow_2hw[1], k=1, axes=(0, 1))
        vx, vy = vy_r.copy(), (-vx_r).copy()
    elif k == 2:
        out = np.rot90(arr_chw, k=2, axes=(1, 2)).copy()
        vx = (-np.rot90(flow_2hw[0], k=2, axes=(0, 1))).copy()
        vy = (-np.rot90(flow_2hw[1], k=2, axes=(0, 1))).copy()
    else:
        out = np.rot90(arr_chw, k=3, axes=(1, 2)).copy()
        vx_r = np.rot90(flow_2hw[0], k=3, axes=(0, 1))
        vy_r = np.rot90(flow_2hw[1], k=3, axes=(0, 1))
        vx, vy = (-vy_r).copy(), vx_r.copy()

    if hflip:
        out = np.ascontiguousarray(out[:, :, ::-1])
        vx = np.ascontiguousarray(-vx[:, ::-1])
        vy = np.ascontiguousarray(vy[:, ::-1])
    return out, np.stack([vx, vy], axis=0)


def gpu_d4_forward(x_bchw: torch.Tensor, k_rot90: int, hflip: bool) -> torch.Tensor:
    k = int(k_rot90) % 4
    out = torch.rot90(x_bchw, k=k, dims=[2, 3]) if k else x_bchw
    if hflip:
        out = torch.flip(out, dims=[3])
    return out.contiguous()


def gpu_d4_invert(probs_3hw: torch.Tensor, flow_2hw: torch.Tensor, k_rot90: int, hflip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    p = probs_3hw
    vx, vy = flow_2hw[0], flow_2hw[1]
    if hflip:
        p = torch.flip(p, dims=[2])
        vx = -torch.flip(vx, dims=[1])
        vy = torch.flip(vy, dims=[1])
    k = int(k_rot90) % 4
    if k == 1:
        p = torch.rot90(p, k=3, dims=[1, 2])
        vx_u = torch.rot90(vx, k=3, dims=[0, 1])
        vy_u = torch.rot90(vy, k=3, dims=[0, 1])
        vx, vy = -vy_u, vx_u
    elif k == 2:
        p = torch.rot90(p, k=2, dims=[1, 2])
        vx = -torch.rot90(vx, k=2, dims=[0, 1])
        vy = -torch.rot90(vy, k=2, dims=[0, 1])
    elif k == 3:
        p = torch.rot90(p, k=1, dims=[1, 2])
        vx_u = torch.rot90(vx, k=1, dims=[0, 1])
        vy_u = torch.rot90(vy, k=1, dims=[0, 1])
        vx, vy = vy_u, -vx_u
    return p, torch.stack([vx, vy], dim=0)


class TopoNative512Dataset(Dataset):
    """Physical-image balanced native 512x512 crop dataset with lossless D4 + gentle continuous rot."""

    def __init__(
        self,
        train_stems: list[str],
        sets: list[AnnotationSet],
        cache_dir: Path,
        disk_meta: dict[str, dict[str, float]],
        crop_size: int = 512,
        crops_per_image: int = 10,
        consensus_drop: float = 0.40,
    ):
        self.stems = sorted(train_stems)
        self.phys_dir = cache_dir / "phys"
        self.tgts_dir = cache_dir / "tgts"
        self.disk_meta = disk_meta
        self.crop_size = crop_size
        self.rot_side = int(math.ceil(crop_size * 1.32)) // 2 * 2
        self.crops_per_image = crops_per_image
        self.consensus_drop = consensus_drop

        self.sets_by_stem: dict[str, list[tuple[AnnotationSet, list[list[float]]]]] = {}
        for s in sets:
            mp = self.tgts_dir / f"{s.set_id}.json"
            ctrs = json.loads(mp.read_text())["centers"] if mp.is_file() else []
            self.sets_by_stem.setdefault(s.stem, []).append((s, ctrs))

    def __len__(self) -> int:
        return len(self.stems) * self.crops_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stems[index // self.crops_per_image]
        aset, ctrs = random.choice(self.sets_by_stem[stem])
        dm = self.disk_meta[stem]
        dcx, dcy, dr = dm["cx"], dm["cy"], dm["r"]

        phys_mmap = np.load(self.phys_dir / f"{stem}.npy", mmap_mode="r")
        tgt_mmap = np.load(self.tgts_dir / f"{aset.set_id}.npy", mmap_mode="r")

        if ctrs and random.random() < 0.80:
            fx, fy = random.choice(ctrs)
            cx = int(round(fx + random.uniform(-155.0, 155.0)))
            cy = int(round(fy + random.uniform(-155.0, 155.0)))
        else:
            ang_r = random.uniform(0.0, 2.0 * math.pi)
            rad_r = dr * 0.96 * math.sqrt(random.random())
            cx = int(round(dcx + rad_r * math.cos(ang_r)))
            cy = int(round(dcy + rad_r * math.sin(ang_r)))

        half_c = self.crop_size // 2
        cx = max(half_c, min(W - half_c, cx))
        cy = max(half_c, min(H - half_c, cy))

        # 85% exact lossless D4 (zero interpolation blur on 1-2 px barbs!), 15% gentle +-22 deg rotation
        if random.random() < 0.85:
            phys_patch = _extract_padded_patch(phys_mmap, cy, cx, self.crop_size, [-0.4, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
            tgt_u8 = _extract_padded_patch(tgt_mmap, cy, cx, self.crop_size, [0, 0, 0, 127, 127])
            masks_p, flow_p = _unpack_targets_u8(tgt_u8)
            stack_8 = np.concatenate([phys_patch, masks_p], axis=0)
            k_rot = random.randint(0, 3)
            hflip = random.random() < 0.5
            warped_8, f_crop = exact_d4_tensor_and_flow(stack_8, flow_p, k_rot, hflip)
            x_crop = warped_8[:5]
            m_crop = warped_8[5:]
        else:
            phys_patch = _extract_padded_patch(phys_mmap, cy, cx, self.rot_side, [-0.4, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
            tgt_u8 = _extract_padded_patch(tgt_mmap, cy, cx, self.rot_side, [0, 0, 0, 127, 127])
            masks_p, flow_p = _unpack_targets_u8(tgt_u8)
            stack_8 = np.concatenate([phys_patch, masks_p], axis=0)
            angle = random.choice((0.0, 90.0, 180.0, 270.0)) + random.uniform(-22.0, 22.0)
            warped_8, warped_f = rotate_flip_tensor_and_flow(
                stack_8,
                flow_p,
                angle_deg=angle,
                hflip=(random.random() < 0.5),
                vflip=False,
                border_values=[-0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            m0 = (self.rot_side - self.crop_size) // 2
            m1 = m0 + self.crop_size
            x_crop = warped_8[:5, m0:m1, m0:m1].copy()
            m_crop = warped_8[5:, m0:m1, m0:m1].copy()
            f_crop = warped_f[:, m0:m1, m0:m1].copy()

        m_crop[0] = (m_crop[0] >= 0.5).astype(np.float32)
        m_crop[1:] = np.clip(m_crop[1:], 0.0, 1.0)

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


def compute_symmetric_barb_weight(tgt_body: torch.Tensor, tgt_sep: torch.Tensor) -> torch.Tensor:
    """Symmetric inner/outer boundary + barb band weighting (zero Bayes logit fattening bias!)."""
    tb4 = tgt_body.unsqueeze(1)
    dilated = F.max_pool2d(tb4, kernel_size=7, stride=1, padding=3)
    eroded = -F.max_pool2d(-tb4, kernel_size=7, stride=1, padding=3)
    sym_band = (dilated - eroded).squeeze(1).clamp(0.0, 1.0)
    return 1.0 + 1.25 * sym_band + 2.5 * tgt_sep


def select_balanced_annot_tokens(train_sets: list[AnnotationSet]) -> list[tuple[int, float]]:
    """80% Consensus Token (0) + 20% across Top-6 annotators for balanced Single411 & Multi296 calibration."""
    counts: dict[int, int] = {}
    for s in train_sets:
        counts[s.annotator_idx] = counts.get(s.annotator_idx, 0) + 1
    top6 = [idx for idx, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]]
    tokens = [(0, 0.80)]
    if top6:
        w_each = 0.20 / len(top6)
        for idx in top6:
            tokens.append((int(idx), w_each))
    return tokens


@torch.no_grad()
def predict_single_image_snake512(
    model: TopoFlowNet,
    phys_5ch: np.ndarray,
    disk_info: dict[str, float],
    device: torch.device,
    annot_tokens: list[tuple[int, float]],
    fast_val: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact-grid 8-view Dihedral D4 Full-2048 + Native 512x512 Overlapping Tile GPU Inference."""
    use_amp = device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    fit = DiskFit(cx=disk_info["cx"], cy=disk_info["cy"], r=disk_info["r"], n_inliers=720, rms=0.5)
    valid = disk_mask(fit, margin_px=8.0)

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
            xf = torch.flip(x_gpu, dims=[3])
            p1, f1 = _eval_heads_weighted(model.forward_features_1x(xf))
            p1i, f1i = gpu_d4_invert(p1[0], f1[0], 0, True)
            p_tot = (0.5 * (p0[0] + p1i)).cpu().numpy()
            f_tot = (0.5 * (f0[0] + f1i)).cpu().numpy()
            p_tot[:, ~valid] = 0.0
            f_tot[:, ~valid] = 0.0
            return p_tot[0], p_tot[1], p_tot[2], f_tot, valid

    # 1. Exact Lossless 8-View Dihedral D4 on Full 2048x2048 Disk (batched by 2 on GPU!)
    d4_views = [(k, hf) for k in (0, 1, 2, 3) for hf in (False, True)]
    p_d4_acc = torch.zeros((3, H, W), dtype=torch.float32, device=device)
    f_d4_acc = torch.zeros((2, H, W), dtype=torch.float32, device=device)
    step_2048 = 2 if device.type == "cuda" else 1
    for v0 in range(0, len(d4_views), step_2048):
        sub_v = d4_views[v0 : v0 + step_2048]
        wb = torch.cat([gpu_d4_forward(x_gpu, k, hf) for k, hf in sub_v], dim=0)
        with amp_ctx:
            pb, fb = _eval_heads_weighted(model.forward_features_1x(wb))
        for idx_v, (k, hf) in enumerate(sub_v):
            pi, fi = gpu_d4_invert(pb[idx_v], fb[idx_v], k, hf)
            p_d4_acc += pi
            f_d4_acc += fi
    p_d4_acc *= 1.0 / len(d4_views)
    f_d4_acc *= 1.0 / len(d4_views)

    # 2. Native 512x512 Overlapping Gaussian Sliding Window (exact 512 training scale, 2-flip TTA)
    tile = 512
    coords = [0, 384, 768, 1152, 1536]
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
            for b0 in range(0, len(positions), 12):
                bt = batch_tiles[b0 : b0 + 12]
                pt1, ft1 = _eval_heads_weighted(model.forward_features_1x(bt))
                bt_f = torch.flip(bt, dims=[3])
                pt2, ft2 = _eval_heads_weighted(model.forward_features_1x(bt_f))
                pt2 = torch.flip(pt2, dims=[3])
                ft2 = torch.flip(ft2, dims=[3])
                ft2[:, 0] = -ft2[:, 0]
                pt_m = 0.5 * (pt1 + pt2)
                ft_m = 0.5 * (ft1 + ft2)
                for k_i, (y0, x0) in enumerate(positions[b0 : b0 + 12]):
                    p_sw_acc[:, y0 : y0 + tile, x0 : x0 + tile] += pt_m[k_i] * gw_gpu
                    f_sw_acc[:, y0 : y0 + tile, x0 : x0 + tile] += ft_m[k_i] * gw_gpu
                    w_sw_acc[y0 : y0 + tile, x0 : x0 + tile] += gw_gpu

        covered = w_sw_acc > 1e-5
        p_sw_acc[:, covered] /= w_sw_acc[covered]
        f_sw_acc[:, covered] /= w_sw_acc[covered]

    p_blend = (0.65 * p_d4_acc + 0.35 * p_sw_acc).cpu().numpy()
    f_blend = (0.65 * f_d4_acc + 0.35 * f_sw_acc).cpu().numpy()
    p_blend[:, ~valid] = 0.0
    f_blend[:, ~valid] = 0.0
    return p_blend[0], p_blend[1], p_blend[2], f_blend, valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "1_data/raw/MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--assets", type=Path, default=ROOT / "1_data/processed")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outer-fold", type=int, default=0)
    ap.add_argument("--encoder", type=str, default="convnext_base.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--batch-crop", type=int, default=16)
    ap.add_argument("--crops-per-image", type=int, default=10)
    ap.add_argument("--lr-enc", type=float, default=1.4e-4)
    ap.add_argument("--lr-dec", type=float, default=5.5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3026)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--infer-only", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    set_seed(args.seed + max(args.outer_fold, 0))
    device = torch.device(args.device)
    if device.type == "cuda":
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
    train_stems, inner_stems, outer_stems = make_fold_splits(
        folds_df, args.outer_fold, seed=args.seed, inner_frac=0.10
    )
    all_test_stems = sorted(p.stem for p in test_img_dir.glob("*.jpeg"))

    if args.smoke:
        train_stems = train_stems[:2]
        inner_stems = inner_stems[:1]
        outer_stems = outer_stems[:1]
        test_stems = all_test_stems[:2]
        epochs = 1
        eval_every = 1
        crops_per_img = 2
        batch_crop = 2
        crop_sz = 256
        enc_name = "convnext_tiny"
        pretrained = False
    else:
        test_stems = all_test_stems
        epochs = args.epochs
        eval_every = args.eval_every
        crops_per_img = args.crops_per_image
        batch_crop = args.batch_crop
        crop_sz = 512
        enc_name = args.encoder
        pretrained = True

    train_stem_set = set(train_stems)
    train_sets = [s for s in all_sets if s.stem in train_stem_set]
    needed_train_stems = sorted(set(train_stems) | set(inner_stems) | set(outer_stems))

    print(
        f"[E121-SNAKE512] fold={args.outer_fold} smoke={args.smoke} train_imgs={len(train_stems)} ({len(train_sets)} sets) "
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

    annot_tokens = select_balanced_annot_tokens(train_sets)
    gt_inner = gt_df[gt_df["stem"].isin(set(inner_stems))][["filament_id", "segmentation_rle"]].copy()
    phys_dir = args.cache_dir / "phys"

    ema_model = TopoFlowNet(
        encoder_name=enc_name,
        pretrained=False,
        in_channels=5,
        num_annotators=len(prefix_to_idx) + 2,
        use_snake=True,
    ).to(device)
    if device.type == "cuda":
        ema_model = ema_model.to(memory_format=torch.channels_last)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    if not args.infer_only:
        ds_crop = TopoNative512Dataset(
            train_stems, train_sets, args.cache_dir, disk_meta,
            crop_size=crop_sz, crops_per_image=crops_per_img,
        )
        dl_crop = DataLoader(
            ds_crop,
            batch_size=batch_crop,
            shuffle=True,
            num_workers=0 if args.smoke else args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=not args.smoke,
            persistent_workers=(not args.smoke and args.workers > 0),
        )

        model = TopoFlowNet(
            encoder_name=enc_name,
            pretrained=pretrained,
            in_channels=5,
            num_annotators=len(prefix_to_idx) + 2,
            use_snake=True,
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

        steps_per_epoch = len(dl_crop)
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
            sum_metrics = {"loss": 0.0, "body": 0.0, "spine": 0.0, "sep": 0.0, "flow": 0.0}
            n_batches = 0
            last_beat = time.time()

            for xb, mb, fb, ab in dl_crop:
                adjust_lr(global_step)
                xb = xb.to(device, non_blocking=True)
                if device.type == "cuda":
                    xb = xb.to(memory_format=torch.channels_last)
                mb = mb.to(device, non_blocking=True)
                fb = fb.to(device, non_blocking=True)
                ab = ab.to(device, non_blocking=True)

                wb = compute_symmetric_barb_weight(mb[:, 0], mb[:, 2])
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    logits_3, flow_2 = model(xb, ab)
                    loss, m_dict = criterion(logits_3, flow_2, mb[:, 0], mb[:, 1], mb[:, 2], fb, wb)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} step={global_step}: {m_dict}")

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

            ep_sec = time.time() - ep_t0
            it_s = n_batches / max(ep_sec, 1e-3)
            avg_m = {k: sum_metrics[k] / max(1, n_batches) for k in sum_metrics}

            inner_pq = float("nan")
            if epoch == epochs or (epoch >= max(4, epochs // 3) and epoch % eval_every == 0) or args.smoke:
                ema_model.eval()
                val_rles_by_gate: dict[float, dict[str, list[str]]] = {g: {} for g in (0.68, 0.75, 0.80, 0.84)}
                v_beat = time.time()
                for v_idx, stem in enumerate(inner_stems, 1):
                    ph = np.array(np.load(phys_dir / f"{stem}.npy", mmap_mode="r"), dtype=np.float32)
                    b_p, s_p, sep_p, fl_p, v_m = predict_single_image_snake512(
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
                            "use_snake": True,
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
                    "use_snake": True,
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
        del model, optimizer, dl_crop
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    if "annot_tokens" in best_ckpt and best_ckpt["annot_tokens"]:
        annot_tokens = [(int(t), float(w)) for t, w in best_ckpt["annot_tokens"]]
    ema_model.load_state_dict(best_ckpt["state_dict"])
    ema_model.eval()
    print(f"[INFER] Loaded best.pt from epoch={best_ckpt['epoch']} (fast inner_pq={best_ckpt['inner_pq']:.4f})", flush=True)

    n_post_procs = 1 if args.smoke else max(2, min(12, os.cpu_count() or 4))

    sel_path = args.out / "selection.json"
    if args.infer_only and sel_path.is_file() and not args.smoke:
        best_cal = json.loads(sel_path.read_text(encoding="utf-8"))["calibration"]
        print(f"[CALIB] Reusing existing selection.json calibration: {best_cal}", flush=True)
    else:
        best_cal = {"pq": -1.0, "mode": "mbr", "t_body": 0.40, "score_gate": 0.80, "fp_cost_scale": 1.0}
        grid_tb = (0.38, 0.42) if not args.smoke else (0.40,)
        grid_sg = (0.74, 0.78, 0.82, 0.85) if not args.smoke else (0.50,)
        grid_fpc = (0.75, 1.00, 1.25) if not args.smoke else (1.00,)
        min_sg = min(grid_sg)

        t_inf0 = time.time()
        inf_beat = time.time()
        inner_cands: dict[str, dict[float, dict[str, list]]] = {}

        if n_post_procs <= 1:
            for idx_i, stem in enumerate(inner_stems, 1):
                ph = np.array(np.load(phys_dir / f"{stem}.npy", mmap_mode="r"), dtype=np.float32)
                bp, sp, sepp, flp, _ = predict_single_image_snake512(
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
                    bp, sp, sepp, flp, _ = predict_single_image_snake512(
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
                bp, sp, sepp, flp, _ = predict_single_image_snake512(
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
                    bp, sp, sepp, flp, _ = predict_single_image_snake512(
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

    t_test = time.time()
    test_rles = _run_pipelined_split(test_stems, args.out / "test_maps", "TEST-TTA")
    sub_path = args.out / f"sub_snake512_fold{args.outer_fold}.csv"
    sub_df = rles_to_pred_df(test_rles)
    validate_submission(sub_df, list(test_stems if args.smoke else all_test_stems))
    write_submission(sub_df, sub_path)
    test_sec = time.time() - t_test
    print(
        f"[TEST] Inferred {len(test_stems)} test images in {test_sec:.1f}s ({test_sec/max(1,len(test_stems)):.2f}s/img), "
        f"instances={sum(len(v) for v in test_rles.values())}",
        flush=True,
    )

    report = {
        "experiment": "E-121a Snake-TopoFlowNet-512",
        "smoke": args.smoke,
        "outer_fold": args.outer_fold,
        "completed_epochs": epochs,
        "best_epoch": int(best_ckpt["epoch"]),
        "inner_calibration": best_cal,
        "outer_oof_summary": outer_summary,
        "test_images_inferred": len(test_stems),
        "test_sec_per_image": round(test_sec / max(1, len(test_stems)), 3),
        "total_wall_seconds": round(time.time() - t_start, 1),
    }
    atomic_write_json(args.out / "report.json", report)
    print(f"[DONE] E121_COMPLETE fold={args.outer_fold} wall={report['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
