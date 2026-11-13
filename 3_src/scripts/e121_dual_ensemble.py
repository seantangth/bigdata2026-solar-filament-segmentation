"""E-121a 100% Pure Dual-Scale TopoFlowNet Evaluator, Nested Cross-Fitted PQ-MBR Solver & Submission Builder.

Fuses:
  - 5-Fold E-120 Global + Continuous-Rotation Equivariant TopoFlowNet (stride-1/1 fine stem)
  - 5-Fold E-121a Native-512 Lossless-D4 Snake-TopoFlowNet (GroupNorm Dynamic Snake + BarbStrip + Balanced Exposure)

Evaluates 4 Pure TopoFlowNet Arms (ZERO legacy smp.Unet, ZERO LightGBM pickle, ZERO post-hoc morphological dilation):
  1. `pure_e120_5m`       : 5-Fold Global/Equivariant TopoFlowNet alone
  2. `pure_e121_5m`       : 5-Fold Native-512 Snake-TopoFlowNet alone
  3. `pure_dual_field_10m`: 10-Model Dual-Scale Calibrated Topological Field Fusion (Body + Spine + Sep + Euler Flow)
  4. `pure_dual_mbr_10m`  : 10-Model Dual-Scale Multi-Hypothesis Exact Lagrangian MWIS / PQ-MBR Solver
Plus optional diagnostic comparison vs `A21L7` (0.46390) and optional `cross_all_mbr` reference.

Includes strict nested 5-fold out-of-fold hyperparameter cross-fitting (zero OOF leakage),
2,000-rep image-level paired bootstrap CI, Single411 vs Multi296 breakdown, Organizers' Rubric
(mIoU, mDice, %IoU>0.5, 1:N, N:1), Multiscale IoU, and Paired Common-TP Barb Branch Density.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "3_src"))
sys.path.insert(0, str(ROOT / "3_src/scripts"))

from sfsc import H, W
from sfsc.geometry import DiskFit, disk_mask
from sfsc.metrics import evaluate
from sfsc.submission import validate_submission, write_submission
from sfsc.topoflow_targets import CandidateInstance, decode_topoflow, pq_mbr_solve
from train_e120_topoflow import load_prediction_npz
from e120_ensemble_mbr import (
    _decode_tri_regimes,
    logit_eps,
    paired_image_bootstrap,
    rles_by_stem_to_df,
    sha256_file,
    sigmoid_f32,
)
import barbs_diag as BD
from miou_metric import miou_full
import rubric_metrics as RM


def _average_npz_list(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b_acc = np.zeros((H, W), dtype=np.float32)
    s_acc = np.zeros((H, W), dtype=np.float32)
    sep_acc = np.zeros((H, W), dtype=np.float32)
    fl_acc = np.zeros((2, H, W), dtype=np.float32)
    for p_str in paths:
        bp, sp, sepp, flp = load_prediction_npz(Path(p_str))
        b_acc += bp
        s_acc += sp
        sep_acc += sepp
        fl_acc += flp
    inv_n = 1.0 / max(1, len(paths))
    b_m = b_acc * inv_n
    s_m = s_acc * inv_n
    sep_m = sep_acc * inv_n
    fl_m = fl_acc * inv_n
    nrm = np.sqrt(fl_m[0] * fl_m[0] + fl_m[1] * fl_m[1])
    act = nrm > 1e-4
    fl_m[0, act] /= nrm[act]
    fl_m[1, act] /= nrm[act]
    return b_m, s_m, sep_m, fl_m


def _fuse_dual_fields(
    f120: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    f121: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid: np.ndarray,
    w_121: float,
    temp_120: float = 1.05,
    temp_121: float = 1.00,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b0, s0, sep0, fl0 = f120
    b1, s1, sep1, fl1 = f121
    w0 = 1.0 - w_121

    z_b = w_121 * (logit_eps(b1) / temp_121) + w0 * (logit_eps(b0) / temp_120)
    body = sigmoid_f32(z_b)
    spine = (w_121 * s1 + w0 * s0).astype(np.float32)
    sep = (w_121 * sep1 + w0 * sep0).astype(np.float32)
    flow = (w_121 * fl1 + w0 * fl0).astype(np.float32)
    nrm = np.sqrt(flow[0] * flow[0] + flow[1] * flow[1])
    act = nrm > 1e-4
    flow[0, act] /= nrm[act]
    flow[1, act] /= nrm[act]

    body[~valid] = 0.0
    spine[~valid] = 0.0
    sep[~valid] = 0.0
    flow[:, ~valid] = 0.0
    return body, spine, sep, flow


def _extract_stem_candidates_worker(job: tuple) -> tuple[str, dict]:
    """Decode compact RLE-only candidate pools for E-120, E-121a, and Dual-Scale field blends."""
    (
        stem,
        e120_paths,
        e121_paths,
        disk_tuple,
        cal_120,
        cal_121,
        dual_grid_w121,
        dual_grid_tbody,
        min_dual_gate,
    ) = job

    fit = DiskFit(cx=disk_tuple[0], cy=disk_tuple[1], r=disk_tuple[2], n_inliers=720, rms=0.5)
    valid = disk_mask(fit, margin_px=8.0)

    f120 = _average_npz_list(e120_paths)
    f121 = _average_npz_list(e121_paths)

    # 1. Pure E-120 (5-model family)
    g120 = float(cal_120["gate"])
    c120_sp, c120_ba, c120_br = _decode_tri_regimes(
        *f120, valid, t_body_base=float(cal_120["t_body"]), score_gate=g120, prefix="e120"
    )
    solved_120 = pq_mbr_solve(
        c120_ba + c120_sp + c120_br,
        [
            ([c.rle_counts for c in c120_ba if c.score >= g120], 0.50),
            ([c.rle_counts for c in c120_sp if c.score >= g120], 0.28),
            ([c.rle_counts for c in c120_br if c.score >= g120 + 0.02], 0.22),
        ],
        tau_pq=0.47,
        fp_cost_scale=float(cal_120["fpc"]),
    )

    # 2. Pure E-121a (5-model family)
    g121 = float(cal_121["gate"])
    c121_sp, c121_ba, c121_br = _decode_tri_regimes(
        *f121, valid, t_body_base=float(cal_121["t_body"]), score_gate=g121, prefix="e121"
    )
    solved_121 = pq_mbr_solve(
        c121_ba + c121_sp + c121_br,
        [
            ([c.rle_counts for c in c121_ba if c.score >= g121], 0.50),
            ([c.rle_counts for c in c121_sp if c.score >= g121], 0.28),
            ([c.rle_counts for c in c121_br if c.score >= g121 + 0.02], 0.22),
        ],
        tau_pq=0.47,
        fp_cost_scale=float(cal_121["fpc"]),
    )

    # 3. Dual-Scale Field Blends across compact calibration grid
    dual_regimes: dict[str, tuple[list[CandidateInstance], list[CandidateInstance], list[CandidateInstance]]] = {}
    for w121 in dual_grid_w121:
        fd = _fuse_dual_fields(f120, f121, valid, w_121=float(w121))
        for tb in dual_grid_tbody:
            key = f"{w121:.2f}_{tb:.2f}"
            dual_regimes[key] = _decode_tri_regimes(
                *fd, valid, t_body_base=float(tb), score_gate=float(min_dual_gate), prefix=f"dual_{key}"
            )

    return stem, {
        "e120_solved": solved_120,
        "e120_split": c120_sp,
        "e121_solved": solved_121,
        "e121_ba": c121_ba,
        "e121_split": c121_sp,
        "e121_bridge": c121_br,
        "dual_regimes": dual_regimes,
    }


def _solve_121_from_cache(entry: dict, g121: float, fpc121: float) -> list[CandidateInstance]:
    cba = [c for c in entry["e121_ba"] if c.score >= g121 - 0.05]
    csp = [c for c in entry["e121_split"] if c.score >= g121 - 0.05]
    cbr = [c for c in entry["e121_bridge"] if c.score >= g121 - 0.03]
    return pq_mbr_solve(
        cba + csp + cbr,
        [
            ([c.rle_counts for c in cba if c.score >= g121], 0.50),
            ([c.rle_counts for c in csp if c.score >= g121], 0.28),
            ([c.rle_counts for c in cbr if c.score >= g121 + 0.02], 0.22),
        ],
        tau_pq=0.47,
        fp_cost_scale=fpc121,
    )


def _solve_dual_stem_from_cache(
    entry: dict,
    w121: float,
    tb: float,
    gate: float,
    fpc_field: float,
    fpc_mbr: float,
    w_ref_dual: float,
    w_ref_121: float,
    w_ref_120: float,
) -> tuple[list[str], list[str]]:
    """Fast RLE-only solver returning (`pure_dual_field_10m`, `pure_dual_mbr_10m`) RLE lists."""
    key = f"{w121:.2f}_{tb:.2f}"
    c_sp_raw, c_ba_raw, c_br_raw = entry["dual_regimes"][key]
    c_sp = [c for c in c_sp_raw if c.score >= gate - 0.05]
    c_ba = [c for c in c_ba_raw if c.score >= gate - 0.05]
    c_br = [c for c in c_br_raw if c.score >= gate - 0.03]

    refs_field = [
        ([c.rle_counts for c in c_ba if c.score >= gate], 0.50),
        ([c.rle_counts for c in c_sp if c.score >= gate], 0.28),
        ([c.rle_counts for c in c_br if c.score >= gate + 0.02], 0.22),
    ]
    cands_dual_field = pq_mbr_solve(
        c_ba + c_sp + c_br,
        refs_field,
        tau_pq=0.47,
        fp_cost_scale=fpc_field,
    )
    rles_field = [c.rle_counts for c in cands_dual_field]

    pool_mbr = (
        list(cands_dual_field)
        + [c for c in c_sp if c.score >= gate - 0.04]
        + [c for c in c_br if c.score >= gate - 0.02]
        + list(entry["e121_solved"])
        + [c for c in entry["e121_split"] if c.score >= gate - 0.04]
        + list(entry["e120_solved"])
    )
    refs_mbr = [
        (rles_field, w_ref_dual),
        ([c.rle_counts for c in entry["e121_solved"]], w_ref_121),
        ([c.rle_counts for c in entry["e120_solved"]], w_ref_120),
    ]
    cands_dual_mbr = pq_mbr_solve(
        pool_mbr,
        refs_mbr,
        tau_pq=0.47,
        fp_cost_scale=fpc_mbr,
    )
    return rles_field, [c.rle_counts for c in cands_dual_mbr]


def _paired_barb_worker(job: tuple) -> list[dict]:
    stem, gt_rows, pred_rles_by_arm = job
    pred_masks = {
        name: [mask_util.decode({"size": [H, W], "counts": r.encode("ascii")}).astype(bool) for r in rles]
        for name, rles in pred_rles_by_arm.items()
    }
    arm_names = list(pred_masks.keys())
    rows = []
    for annot_rles in gt_rows:
        for rle in annot_rles:
            gm = mask_util.decode({"size": [H, W], "counts": rle.encode("ascii")}).astype(bool)
            matched = {}
            for name in arm_names:
                for pm in pred_masks[name]:
                    inter = int((gm & pm).sum())
                    if inter == 0:
                        continue
                    iou = inter / float(int(gm.sum()) + int(pm.sum()) - inter)
                    if iou > 0.5:
                        matched[name] = (iou, pm)
                        break
            if len(matched) == len(arm_names):
                ys, xs = np.nonzero(gm)
                y0, y1 = max(0, int(ys.min()) - 12), min(H, int(ys.max()) + 13)
                x0, x1 = max(0, int(xs.min()) - 12), min(W, int(xs.max()) + 13)
                G = gm[y0:y1, x0:x1]
                sg, thin_g = BD.shape(G)
                rec: dict[str, float | int] = {
                    "gt_branch": float(sg["branch_per100"]),
                    "gt_thin": int(thin_g.sum()),
                    "gt_area": int(G.sum()),
                }
                for name, (iou, pm) in matched.items():
                    P = pm[y0:y1, x0:x1]
                    sp, _ = BD.shape(P)
                    miss_thin = int((G & ~P & thin_g).sum())
                    rec[f"{name}_iou"] = float(iou)
                    rec[f"{name}_miou"] = float(miou_full(gm, pm))
                    rec[f"{name}_branch"] = float(sp["branch_per100"])
                    rec[f"{name}_miss_thin"] = miss_thin
                rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e120-root", type=Path, default=ROOT / "5_outputs/exp120_topoflow")
    ap.add_argument("--e121-root", type=Path, default=ROOT / "5_outputs/exp121_snake512")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--assets", type=Path, default=ROOT / "1_data/processed")
    ap.add_argument("--a21l7-oof-csv", type=Path, default=ROOT / "5_outputs/exp102_diagnostic/baseline/predictions.csv")
    ap.add_argument("--test-images", type=Path, default=ROOT / "1_data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    ap.add_argument("--out", type=Path, default=ROOT / "5_outputs/exp121_snake512/final_eval")
    ap.add_argument("--copy-submissions-dir", type=Path, default=ROOT / "5_outputs/submissions")
    ap.add_argument("--build-test", action="store_true")
    ap.add_argument("--run-barb-audit", action="store_true")
    ap.add_argument("--max-stems-per-fold", type=int, default=0)
    ap.add_argument("--procs", type=int, default=max(2, min(8, os.cpu_count() or 4)))
    args = ap.parse_args()

    t_all = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    folds_df = pd.read_csv(args.assets / "folds_cvA_v1.csv")
    gt_df = pd.read_csv(args.assets / "gt_train_v1.csv")
    gt_df["stem"] = gt_df["filament_id"].str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0]
    gt_df["annot_key"] = gt_df["filament_id"].str.split("_", n=1).str[0]
    multi_stems = set(
        gt_df.groupby("stem")["filament_id"]
        .apply(lambda s: s.str.split("-", n=1).str[0].nunique() > 1)
        .loc[lambda x: x]
        .index
    )

    geom_df = pd.read_csv(args.assets / "disk_geometry_v1.csv")
    geom_map = {str(r.stem): (float(r.cx), float(r.cy), float(r.r)) for r in geom_df.itertuples(index=False)}

    if args.a21l7_oof_csv.is_file():
        base_oof_df = pd.read_csv(args.a21l7_oof_csv)
    else:
        # The A21L7 U-Net baseline OOF CSV is an optional, development-only comparison (not part of the public repo).
        # Without it, every "vs_a21l7" / "a21l7_pq" field is computed against an empty prediction set and should be ignored.
        print(f"[INFO] optional A21L7 baseline OOF CSV not found ({args.a21l7_oof_csv}); "
              "'vs_a21l7' fields compare against an empty prediction set.", flush=True)
        base_oof_df = pd.DataFrame({"filament_id": pd.Series(dtype=str), "segmentation_rle": pd.Series(dtype=str)})
    base_oof_df["stem"] = base_oof_df["filament_id"].str.rsplit("_", n=1).str[0]

    def _resolve_npz(root: Path, fid: int, split_folder: str, stem: str) -> Path:
        for cand in (
            root / f"fold{fid}/{split_folder}/{stem}.npz",
            root / f"fold{fid}/run/{split_folder}/{stem}.npz",
        ):
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Missing {split_folder}/{stem}.npz under {root}/fold{fid}")

    def _median_cal(root: Path, fids: list[int], defaults: tuple[float, float, float]) -> dict[str, float]:
        tbs, sgs, fpcs = [], [], []
        for fid in fids:
            for cand in (root / f"fold{fid}/selection.json", root / f"fold{fid}/run/selection.json"):
                if cand.is_file():
                    c = json.loads(cand.read_text())["calibration"]
                    tbs.append(float(c["t_body"]))
                    sgs.append(float(c["score_gate"]))
                    fpcs.append(float(c["fp_cost_scale"]))
                    break
        return {
            "t_body": float(np.median(tbs)) if tbs else defaults[0],
            "gate": float(np.median(sgs)) if sgs else defaults[1],
            "fpc": float(np.median(fpcs)) if fpcs else defaults[2],
        }

    cal_120 = _median_cal(args.e120_root, fold_ids, (0.42, 0.86, 1.00))
    cal_121_raw = _median_cal(args.e121_root, fold_ids, (0.42, 0.85, 1.00))
    # Extract E-121a candidates from min gate 0.85 so crop-sampling prior shift (gates 0.85..0.92) can be cross-fitted cleanly
    cal_121 = {"t_body": cal_121_raw["t_body"], "gate": min(0.85, cal_121_raw["gate"]), "fpc": cal_121_raw["fpc"]}
    print(f"[CALIB] Inner-val medians -> E120={cal_120} | E121_raw={cal_121_raw}", flush=True)

    dual_grid_w121 = (0.38, 0.50, 0.62)
    dual_grid_tbody = (0.40, 0.43)
    dual_grid_gate = (0.83, 0.86, 0.88, 0.90)
    dual_grid_fpc = (1.00, 1.25, 1.50)
    ref_weight_grid = (
        (0.50, 0.30, 0.20),
        (0.55, 0.20, 0.25),
        (0.65, 0.15, 0.20),
    )

    stems_by_fold: dict[int, list[str]] = {}
    eval_stems: list[str] = []
    oof_jobs = []
    for fid in fold_ids:
        f_stems = sorted(folds_df.loc[folds_df["fold"] == fid, "stem"].tolist())
        if args.max_stems_per_fold > 0:
            f_stems = f_stems[: args.max_stems_per_fold]
        stems_by_fold[fid] = f_stems
        for s in f_stems:
            p120 = str(_resolve_npz(args.e120_root, fid, "oof_maps", s))
            p121 = str(_resolve_npz(args.e121_root, fid, "oof_maps", s))
            eval_stems.append(s)
            oof_jobs.append((
                s, [p120], [p121], geom_map[s], cal_120, cal_121,
                dual_grid_w121, dual_grid_tbody, min(dual_grid_gate),
            ))

    print(f"[EXTRACT] Decoding multi-regime candidate pools across {len(oof_jobs)} OOF images...", flush=True)
    t_ext = time.time()
    oof_cache: dict[str, dict] = {}
    with Pool(processes=args.procs) as pool:
        for idx_done, (stem, entry) in enumerate(pool.imap_unordered(_extract_stem_candidates_worker, oof_jobs, chunksize=4), 1):
            oof_cache[stem] = entry
            if idx_done % 100 == 0 or idx_done == len(oof_jobs):
                print(f"  [{idx_done}/{len(oof_jobs)}] extracted ({time.time()-t_ext:.1f}s)", flush=True)

    # Score every dual hyperparameter tuple per fold once -> zero-leakage nested 5-fold cross-fitting in seconds!
    candidate_cfgs = []
    for w121 in dual_grid_w121:
        for tb in dual_grid_tbody:
            for gate in dual_grid_gate:
                for fpc_mbr in dual_grid_fpc:
                    for wr_d, wr_121, wr_120 in ref_weight_grid:
                        candidate_cfgs.append({
                            "w121": w121,
                            "tb": tb,
                            "gate": gate,
                            "fpc_field": 1.00,
                            "fpc_mbr": fpc_mbr,
                            "w_ref_dual": wr_d,
                            "w_ref_121": wr_121,
                            "w_ref_120": wr_120,
                        })

    gt_by_fold = {
        fid: gt_df[gt_df["stem"].isin(set(stems_by_fold[fid]))][["filament_id", "segmentation_rle"]].copy()
        for fid in fold_ids
    }

    from sfsc.metrics import pq_from_counts

    # Nested 5-fold calibration of E-121a prior-shifted gate & fpc (unclipping the single-fold 0.85 grid cap, zero OOF leakage)
    e121_cal_cfgs = [
        {"gate": g121, "fpc": fpc121}
        for g121 in (0.85, 0.88, 0.90, 0.92)
        for fpc121 in (1.00, 1.25, 1.50)
    ]
    e121_cands_by_cfg: list[dict[str, list[CandidateInstance]]] = []
    e121_counts: dict[int, dict[int, tuple[float, float, float, float]]] = {i: {} for i in range(len(e121_cal_cfgs))}
    for ci, ec in enumerate(e121_cal_cfgs):
        cmap = {s: _solve_121_from_cache(oof_cache[s], ec["gate"], ec["fpc"]) for s in eval_stems}
        e121_cands_by_cfg.append(cmap)
        for fid in fold_ids:
            fstems = stems_by_fold[fid]
            res_121 = evaluate(gt_by_fold[fid], rles_by_stem_to_df({k: [c.rle_counts for c in cmap[k]] for k in fstems}), distributions=False)
            e121_counts[ci][fid] = (float(res_121.tp), float(res_121.fp), float(res_121.fn), float(res_121.rows["sum_iou"].sum()))

    for held_fid in fold_ids:
        tr_fids = [f for f in fold_ids if f != held_fid] or [held_fid]
        best_ci = max(
            range(len(e121_cal_cfgs)),
            key=lambda ci: pq_from_counts(
                sum(e121_counts[ci][f][0] for f in tr_fids),
                sum(e121_counts[ci][f][1] for f in tr_fids),
                sum(e121_counts[ci][f][2] for f in tr_fids),
                sum(e121_counts[ci][f][3] for f in tr_fids),
            )[0],
        )
        for s in stems_by_fold[held_fid]:
            oof_cache[s]["e121_solved"] = e121_cands_by_cfg[best_ci][s]

    best_121_full_ci = max(
        range(len(e121_cal_cfgs)),
        key=lambda ci: pq_from_counts(
            sum(e121_counts[ci][f][0] for f in fold_ids),
            sum(e121_counts[ci][f][1] for f in fold_ids),
            sum(e121_counts[ci][f][2] for f in fold_ids),
            sum(e121_counts[ci][f][3] for f in fold_ids),
        )[0],
    )
    cal_121 = {
        "t_body": cal_121_raw["t_body"],
        "gate": float(e121_cal_cfgs[best_121_full_ci]["gate"]),
        "fpc": float(e121_cal_cfgs[best_121_full_ci]["fpc"]),
    }
    print(f"[CALIB-E121] Nested prior-shift calibrated E-121a consensus -> {cal_121}", flush=True)

    print(f"[CROSS-FIT] Evaluating {len(candidate_cfgs)} compact configs with strict nested out-of-fold selection...", flush=True)
    fold_counts_field: dict[int, dict[int, tuple[float, float, float, float]]] = {i: {} for i in range(len(candidate_cfgs))}
    fold_counts_mbr: dict[int, dict[int, tuple[float, float, float, float]]] = {i: {} for i in range(len(candidate_cfgs))}
    rles_by_cfg_field: list[dict[str, list[str]]] = []
    rles_by_cfg_mbr: list[dict[str, list[str]]] = []

    for c_idx, c_cfg in enumerate(candidate_cfgs):
        map_f: dict[str, list[str]] = {}
        map_m: dict[str, list[str]] = {}
        for s in eval_stems:
            rf, rm = _solve_dual_stem_from_cache(
                oof_cache[s],
                w121=c_cfg["w121"],
                tb=c_cfg["tb"],
                gate=c_cfg["gate"],
                fpc_field=c_cfg["fpc_field"],
                fpc_mbr=c_cfg["fpc_mbr"],
                w_ref_dual=c_cfg["w_ref_dual"],
                w_ref_121=c_cfg["w_ref_121"],
                w_ref_120=c_cfg["w_ref_120"],
            )
            map_f[s] = rf
            map_m[s] = rm
        rles_by_cfg_field.append(map_f)
        rles_by_cfg_mbr.append(map_m)
        for fid in fold_ids:
            fstems = stems_by_fold[fid]
            res_f = evaluate(gt_by_fold[fid], rles_by_stem_to_df({k: map_f[k] for k in fstems}), distributions=False)
            res_m = evaluate(gt_by_fold[fid], rles_by_stem_to_df({k: map_m[k] for k in fstems}), distributions=False)
            fold_counts_field[c_idx][fid] = (float(res_f.tp), float(res_f.fp), float(res_f.fn), float(res_f.rows["sum_iou"].sum()))
            fold_counts_mbr[c_idx][fid] = (float(res_m.tp), float(res_m.fp), float(res_m.fn), float(res_m.rows["sum_iou"].sum()))

    from sfsc.metrics import pq_from_counts

    def _nested_select(
        fold_counts: dict[int, dict[int, tuple[float, float, float, float]]],
        rles_per_cfg: list[dict[str, list[str]]],
    ) -> tuple[dict[str, list[str]], dict[int, dict], dict]:
        nested_oof: dict[str, list[str]] = {}
        selected_per_fold: dict[int, dict] = {}
        for held_fid in fold_ids:
            train_fids = [f for f in fold_ids if f != held_fid] or [held_fid]
            best_i = 0
            best_pq = -1.0
            for c_idx in range(len(candidate_cfgs)):
                tp = sum(fold_counts[c_idx][f][0] for f in train_fids)
                fp = sum(fold_counts[c_idx][f][1] for f in train_fids)
                fn = sum(fold_counts[c_idx][f][2] for f in train_fids)
                si = sum(fold_counts[c_idx][f][3] for f in train_fids)
                pq_tr, _, _ = pq_from_counts(tp, fp, fn, si)
                if pq_tr > best_pq:
                    best_pq = pq_tr
                    best_i = c_idx
            selected_per_fold[held_fid] = {"cfg_idx": best_i, "train_4fold_pq": round(best_pq, 6), **candidate_cfgs[best_i]}
            for s in stems_by_fold[held_fid]:
                nested_oof[s] = rles_per_cfg[best_i][s]

        # Full-5-fold consensus config for 180 Test images
        best_full_i = 0
        best_full_pq = -1.0
        for c_idx in range(len(candidate_cfgs)):
            tp = sum(fold_counts[c_idx][f][0] for f in fold_ids)
            fp = sum(fold_counts[c_idx][f][1] for f in fold_ids)
            fn = sum(fold_counts[c_idx][f][2] for f in fold_ids)
            si = sum(fold_counts[c_idx][f][3] for f in fold_ids)
            pq_all, _, _ = pq_from_counts(tp, fp, fn, si)
            if pq_all > best_full_pq:
                best_full_pq = pq_all
                best_full_i = c_idx
        return nested_oof, selected_per_fold, {"cfg_idx": best_full_i, "oof_5fold_pq": round(best_full_pq, 6), **candidate_cfgs[best_full_i]}

    nested_field_rles, sel_field_folds, consensus_field_cfg = _nested_select(fold_counts_field, rles_by_cfg_field)
    nested_mbr_rles, sel_mbr_folds, consensus_mbr_cfg = _nested_select(fold_counts_mbr, rles_by_cfg_mbr)

    arm_oof_rles: dict[str, dict[str, list[str]]] = {
        "pure_e120_5m": {s: [c.rle_counts for c in oof_cache[s]["e120_solved"]] for s in eval_stems},
        "pure_e121_5m": {s: [c.rle_counts for c in oof_cache[s]["e121_solved"]] for s in eval_stems},
        "pure_dual_field_10m": nested_field_rles,
        "pure_dual_mbr_10m": nested_mbr_rles,
    }

    gt_eval = gt_df[gt_df["stem"].isin(set(eval_stems))][["filament_id", "segmentation_rle"]].copy()
    base_eval = base_oof_df[base_oof_df["stem"].isin(set(eval_stems))][["filament_id", "segmentation_rle"]].copy()
    pure120_df = rles_by_stem_to_df(arm_oof_rles["pure_e120_5m"])

    single_eval_stems = set(eval_stems) - multi_stems
    multi_eval_stems = set(eval_stems) & multi_stems
    gt_single = gt_df[gt_df["stem"].isin(single_eval_stems)][["filament_id", "segmentation_rle"]].copy()
    gt_multi = gt_df[gt_df["stem"].isin(multi_eval_stems)][["filament_id", "segmentation_rle"]].copy()

    summary_report: dict[str, object] = {
        "evaluated_folds": fold_ids,
        "n_eval_images": len(eval_stems),
        "cal_e120": cal_120,
        "cal_e121": cal_121,
        "nested_selection_dual_field": sel_field_folds,
        "consensus_test_cfg_dual_field": consensus_field_cfg,
        "nested_selection_dual_mbr": sel_mbr_folds,
        "consensus_test_cfg_dual_mbr": consensus_mbr_cfg,
        "arms": {},
    }

    print(f"\n=== 100% PURE TOPOFLOWNET EVALUATION ON {len(eval_stems)} OOF IMAGES (NESTED CROSS-FITTED) ===", flush=True)
    for arm_name in ("pure_e120_5m", "pure_e121_5m", "pure_dual_field_10m", "pure_dual_mbr_10m"):
        s_map = arm_oof_rles[arm_name]
        cand_df = rles_by_stem_to_df(s_map)
        cand_df.to_csv(args.out / f"oof_{arm_name}.csv", index=False)

        boot_a21 = paired_image_bootstrap(gt_eval, base_eval, cand_df, n_boot=2000, seed=2026)
        boot_120 = paired_image_bootstrap(gt_eval, pure120_df, cand_df, n_boot=2000, seed=2026)

        ps_df = rles_by_stem_to_df({k: v for k, v in s_map.items() if k in single_eval_stems})
        pm_df = rles_by_stem_to_df({k: v for k, v in s_map.items() if k in multi_eval_stems})
        res_single = evaluate(gt_single, ps_df, distributions=False) if len(gt_single) else None
        res_multi = evaluate(gt_multi, pm_df, distributions=False) if len(gt_multi) else None

        res_all_dist = evaluate(gt_eval, cand_df, distributions=True)
        rub_sum = RM.stats(res_all_dist)

        per_fold_pq = {}
        for fid in fold_ids:
            fst = set(stems_by_fold[fid])
            bs_f = base_oof_df[base_oof_df["stem"].isin(fst)][["filament_id", "segmentation_rle"]]
            cd_f = rles_by_stem_to_df({k: s_map[k] for k in stems_by_fold[fid]})
            rb_f = evaluate(gt_by_fold[fid], bs_f, distributions=False)
            rc_f = evaluate(gt_by_fold[fid], cd_f, distributions=False)
            per_fold_pq[f"fold{fid}"] = {
                "a21l7_pq": round(float(rb_f.pq), 5),
                "cand_pq": round(float(rc_f.pq), 5),
                "delta_vs_a21l7": round(float(rc_f.pq - rb_f.pq), 5),
            }

        arm_rec = {
            "vs_a21l7": boot_a21,
            "vs_pure_e120": {
                "delta_pq": boot_120["delta_pq"],
                "ci95": boot_120["ci95"],
                "p_pos": boot_120["p_pos"],
            },
            "single_411_pq": round(float(res_single.pq), 5) if res_single else None,
            "multi_296_pq": round(float(res_multi.pq), 5) if res_multi else None,
            "rubric": {k: round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v for k, v in rub_sum.items()},
            "per_fold": per_fold_pq,
            "oof_total_instances": int(len(cand_df)),
            "oof_mean_instances_per_image": round(float(len(cand_df) / max(1, len(eval_stems))), 3),
        }
        summary_report["arms"][arm_name] = arm_rec
        print(
            f"  [{arm_name:20s}] PQ={boot_a21['cand']['PQ_exact']:.5f} "
            f"| vs A21L7 Δ={boot_a21['delta_pq']:+.5f} CI={boot_a21['ci95']} p+={boot_a21['p_pos']:.3f} "
            f"| vs E120 Δ={boot_120['delta_pq']:+.5f} CI={boot_120['ci95']} "
            f"| Single={arm_rec['single_411_pq']} Multi={arm_rec['multi_296_pq']} "
            f"| mDice={rub_sum['mean_dice']:.4f} 1:N={boot_a21['cand']['gt_1toN']} N:1={boot_a21['cand']['pred_Nto1']}",
            flush=True,
        )

    if args.run_barb_audit:
        print("\n[BARB AUDIT] Running paired Common-TP Barb & Multiscale-IoU audit across 120 Multi-Annotator images...", flush=True)
        a21_rle_map = {
            s: g["segmentation_rle"].tolist()
            for s, g in base_oof_df[base_oof_df["stem"].isin(multi_eval_stems)].groupby("stem", sort=False)
        }
        sample_multi = sorted(multi_eval_stems)[:120]
        barb_jobs = []
        for s in sample_multi:
            gt_s = gt_df[gt_df["stem"] == s]
            annot_groups = [g["segmentation_rle"].tolist() for _, g in gt_s.groupby("annot_key", sort=False)]
            pred_by_arm = {
                "A21L7": a21_rle_map.get(s, []),
                "pure_e120_5m": arm_oof_rles["pure_e120_5m"].get(s, []),
                "pure_e121_5m": arm_oof_rles["pure_e121_5m"].get(s, []),
                "pure_dual_mbr_10m": arm_oof_rles["pure_dual_mbr_10m"].get(s, []),
            }
            barb_jobs.append((s, annot_groups, pred_by_arm))

        with Pool(processes=args.procs) as pool:
            barb_rows = [r for sub in pool.map(_paired_barb_worker, barb_jobs) for r in sub]
        bdf = pd.DataFrame(barb_rows)
        barb_summary: dict[str, object] = {
            "n_common_tp": int(len(bdf)),
            "gt_branch_per100_median": round(float(bdf["gt_branch"].median()), 3) if len(bdf) else None,
            "gt_branch_per100_mean": round(float(bdf["gt_branch"].mean()), 3) if len(bdf) else None,
            "models": {},
        }
        if len(bdf):
            for name in ("A21L7", "pure_e120_5m", "pure_e121_5m", "pure_dual_mbr_10m"):
                tr = 1.0 - float(bdf[f"{name}_miss_thin"].sum()) / max(1.0, float(bdf["gt_thin"].sum()))
                m_rec = {
                    "mean_iou": round(float(bdf[f"{name}_iou"].mean()), 4),
                    "mean_miou": round(float(bdf[f"{name}_miou"].mean()), 4),
                    "median_miou": round(float(bdf[f"{name}_miou"].median()), 4),
                    "barb_branch_per100_median": round(float(bdf[f"{name}_branch"].median()), 3),
                    "barb_branch_per100_mean": round(float(bdf[f"{name}_branch"].mean()), 3),
                    "thin_recall": round(tr, 4),
                }
                barb_summary["models"][name] = m_rec
                print(
                    f"  [BARB {name:18s}] CommonTP={len(bdf)} | MIoU={m_rec['mean_miou']:.4f} "
                    f"| Barb med={m_rec['barb_branch_per100_median']:.3f}/100px | ThinRecall={m_rec['thin_recall']:.4f}",
                    flush=True,
                )
        summary_report["paired_barb_audit"] = barb_summary

    if args.build_test:
        print("\n[TEST] Building 100% Pure Dual-Scale TopoFlowNet submissions across 180 Test images...", flush=True)
        t_test0 = time.time()
        test_stems = sorted(p.stem for p in args.test_images.glob("*.jpeg"))
        needed_w = sorted({float(consensus_field_cfg["w121"]), float(consensus_mbr_cfg["w121"])})
        needed_tb = sorted({float(consensus_field_cfg["tb"]), float(consensus_mbr_cfg["tb"])})
        min_test_gate = min(float(consensus_field_cfg["gate"]), float(consensus_mbr_cfg["gate"]))

        test_jobs = []
        for ts in test_stems:
            p120_list = [str(_resolve_npz(args.e120_root, fid, "test_maps", ts)) for fid in fold_ids]
            p121_list = [str(_resolve_npz(args.e121_root, fid, "test_maps", ts)) for fid in fold_ids]
            test_jobs.append((
                ts, p120_list, p121_list, geom_map[ts], cal_120, cal_121,
                needed_w, needed_tb, min_test_gate,
            ))

        test_cache: dict[str, dict] = {}
        with Pool(processes=args.procs) as pool:
            for stem, entry in pool.imap_unordered(_extract_stem_candidates_worker, test_jobs, chunksize=4):
                test_cache[stem] = entry

        arm_test_rles: dict[str, dict[str, list[str]]] = {
            "pure_e121_5m": {},
            "pure_dual_field_10m": {},
            "pure_dual_mbr_10m": {},
        }
        for ts in test_stems:
            arm_test_rles["pure_e121_5m"][ts] = [c.rle_counts for c in test_cache[ts]["e121_solved"]]
            rf_c, _ = _solve_dual_stem_from_cache(
                test_cache[ts],
                w121=float(consensus_field_cfg["w121"]),
                tb=float(consensus_field_cfg["tb"]),
                gate=float(consensus_field_cfg["gate"]),
                fpc_field=float(consensus_field_cfg["fpc_field"]),
                fpc_mbr=float(consensus_field_cfg["fpc_mbr"]),
                w_ref_dual=float(consensus_field_cfg["w_ref_dual"]),
                w_ref_121=float(consensus_field_cfg["w_ref_121"]),
                w_ref_120=float(consensus_field_cfg["w_ref_120"]),
            )
            _, rm_c = _solve_dual_stem_from_cache(
                test_cache[ts],
                w121=float(consensus_mbr_cfg["w121"]),
                tb=float(consensus_mbr_cfg["tb"]),
                gate=float(consensus_mbr_cfg["gate"]),
                fpc_field=float(consensus_mbr_cfg["fpc_field"]),
                fpc_mbr=float(consensus_mbr_cfg["fpc_mbr"]),
                w_ref_dual=float(consensus_mbr_cfg["w_ref_dual"]),
                w_ref_121=float(consensus_mbr_cfg["w_ref_121"]),
                w_ref_120=float(consensus_mbr_cfg["w_ref_120"]),
            )
            arm_test_rles["pure_dual_field_10m"][ts] = rf_c
            arm_test_rles["pure_dual_mbr_10m"][ts] = rm_c

        summary_report["test_seconds"] = round(time.time() - t_test0, 1)
        summary_report["test_submissions"] = {}
        if args.copy_submissions_dir:
            args.copy_submissions_dir.mkdir(parents=True, exist_ok=True)

        for arm_name, s_map in sorted(arm_test_rles.items()):
            sub_csv = args.out / f"sub_e121_{arm_name}_{len(fold_ids)}f.csv"
            df_sub = rles_by_stem_to_df(s_map)
            val_rep = validate_submission(df_sub, list(test_stems))
            write_submission(df_sub, sub_csv)
            sub_sha = sha256_file(sub_csv)
            sub_meta = {
                "path": str(sub_csv),
                "sha256": sub_sha,
                "n_instances": int(len(df_sub)),
                "mean_instances_per_image": round(float(len(df_sub) / max(1, len(test_stems))), 3),
                "n_nonempty_images": int(val_rep["n_images_with_rows"]),
                "missing_images": val_rep["missing_images"],
                "validate": val_rep,
                "oof_pq_exact": summary_report["arms"].get(arm_name, {}).get("vs_a21l7", {}).get("cand", {}).get("PQ_exact"),
            }
            summary_report["test_submissions"][arm_name] = sub_meta
            if args.copy_submissions_dir and len(fold_ids) == 5:
                dst_csv = args.copy_submissions_dir / sub_csv.name
                shutil.copy2(sub_csv, dst_csv)
                (args.copy_submissions_dir / f"{sub_csv.stem}.json").write_text(
                    json.dumps(sub_meta, indent=2) + "\n", encoding="utf-8"
                )
            print(
                f"  [TEST SUB VERIFIED] {sub_csv.name}: {len(df_sub)} disjoint instances "
                f"({sub_meta['mean_instances_per_image']:.2f}/img) across "
                f"{sub_meta['n_nonempty_images']}/180 test images | sha256={sub_sha[:16]}...",
                flush=True,
            )

    summary_report["total_wall_seconds"] = round(time.time() - t_all, 1)
    (args.out / "evaluation_summary.json").write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Saved evaluation_summary.json to {args.out / 'evaluation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
