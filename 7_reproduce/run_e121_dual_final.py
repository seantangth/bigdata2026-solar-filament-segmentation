"""100% Pure Dual-Scale TopoFlowNet Reproducible Pipeline (10 Checkpoints -> Submission + Optional Evaluation).

Reproduces the final Kaggle submission `sub_e121_pure_dual_mbr_10m_5f.csv` (ref 58267599) from `final_e121_dual.json`.

Supports:
  1. End-to-end GPU inference from the 10 pure TopoFlowNet checkpoints (`--weights 4_models`):
       - 5x `exp120_topoflow_f{0..4}/best.pt` (Global + Continuous-Rotation Equivariant TopoFlowNet)
       - 5x `exp121_snake512_f{0..4}/best.pt` (Native-512 Lossless-D4 Snake-TopoFlowNet)
     Reads ONLY single-image H-alpha `.jpeg` inputs (`extract_physics_channels` + single-image disk fit).
  2. `--dry-run`: Fast smoke verification (first 3 images, Fold-0 Dual-Scale pair) on GPU or CPU.
  3. Deterministic CPU reproduction from pre-saved `.npz` topological maps (`--e120-maps` + `--e121-maps`).
  4. Optional `--gt-csv` evaluation on any annotated dataset (computes official PQ/SQ/RQ + Rubric metrics).
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from multiprocessing import Pool
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "3_src"), str(ROOT / "3_src/scripts")]

from sfsc.geometry import fit_disk  # noqa: E402
from sfsc.metrics import evaluate  # noqa: E402
from sfsc.submission import validate_submission, write_submission  # noqa: E402
from sfsc.topoflow_net import TopoFlowNet  # noqa: E402
from sfsc.topoflow_physics import extract_physics_channels  # noqa: E402
from train_e120_topoflow import predict_single_image, save_prediction_npz  # noqa: E402
from train_e121_snake512 import predict_single_image_snake512  # noqa: E402
from e120_ensemble_mbr import rles_by_stem_to_df  # noqa: E402
from e121_dual_ensemble import _extract_stem_candidates_worker, _solve_dual_stem_from_cache  # noqa: E402
import rubric_metrics as RM  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_TZ_LOG = timezone(timedelta(hours=8))


def _fmt_dur(sec: float) -> str:
    s = int(max(sec, 0.0))
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


class _ProgressTracker:
    def __init__(self, total: int, stage_names: list[str], every_sec: float = 45.0) -> None:
        self.total = max(int(total), 1)
        self.stage_names = stage_names
        self.every = every_sec
        self.stage_idx = 0
        self.done_n = 0
        self.t0 = time.time()
        self.last = 0.0

    def set_stage(self, idx: int) -> None:
        self.stage_idx = idx

    def update(self, n: int = 1, force: bool = False) -> None:
        self.done_n += n
        now = time.time()
        if not force and (now - self.last) < self.every:
            return
        self.last = now
        elapsed = max(now - self.t0, 1e-3)
        st_name = self.stage_names[self.stage_idx] if self.stage_idx < len(self.stage_names) else "run"
        parts = [
            "[PROGRESS]",
            datetime.now(_TZ_LOG).strftime("%Y-%m-%d %H:%M:%S"),
            f"stage={st_name}({self.stage_idx + 1}/{len(self.stage_names)})",
            f"done={self.done_n}/{self.total} ({100.0 * self.done_n / self.total:.1f}%)",
        ]
        if self.done_n > 0:
            rate = self.done_n / elapsed
            remain = max(0.0, (self.total - self.done_n) / rate)
            fin = datetime.now(_TZ_LOG) + timedelta(seconds=remain)
            parts.extend([
                f"rate={rate:.3g}img/s",
                f"elapsed={_fmt_dur(elapsed)}",
                f"eta={_fmt_dur(remain)}",
                f"finish={fin.strftime('%H:%M')}",
            ])
        else:
            parts.extend([f"elapsed={_fmt_dur(elapsed)}", "eta=?", "finish=?"])
        print(" ".join(parts), flush=True)

    def done(self) -> None:
        self.done_n = self.total
        print(
            f"[PROGRESS] {datetime.now(_TZ_LOG):%Y-%m-%d %H:%M:%S} DONE elapsed={_fmt_dur(time.time() - self.t0)}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=HERE / "final_e121_dual.json")
    ap.add_argument("--test-images", type=Path, required=True, help="Directory of input .jpeg images")
    ap.add_argument("--work", type=Path, required=True, help="Clean output directory")
    ap.add_argument("--weights", type=Path, default=ROOT / "4_models", help="Directory holding <model_name>/best.pt")
    ap.add_argument("--e120-maps", type=Path, default=None, help="Optional pre-inferred E-120 maps root")
    ap.add_argument("--e121-maps", type=Path, default=None, help="Optional pre-inferred E-121a maps root")
    ap.add_argument("--geometry", type=Path, default=None, help="Optional precomputed disk_geometry.csv")
    ap.add_argument("--gt-csv", type=Path, default=None, help="Optional ground-truth CSV to score predictions against")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=max(2, min(8, os.cpu_count() or 4)))
    ap.add_argument("--dry-run", action="store_true", help="Fast smoke test on 3 images using Fold-0 Dual-Scale pair")
    ap.add_argument("--verify", action="store_true", help="Verify exact SHA256 against final_e121_dual.json (from maps)")
    args = ap.parse_args()

    t_all = time.time()
    cfg_doc = json.loads(args.config.read_text(encoding="utf-8"))
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)

    all_stems = sorted(p.stem for p in args.test_images.glob("*.jpeg"))
    assert len(all_stems) > 0, f"No .jpeg images found in {args.test_images}"
    if args.dry_run:
        test_stems = all_stems[: min(3, len(all_stems))]
        fold_ids = [0]
    else:
        if "n_test_images" in cfg_doc and args.gt_csv is None:
            assert len(all_stems) == int(cfg_doc["n_test_images"]), (
                f"Expected {cfg_doc['n_test_images']} images, found {len(all_stems)}"
            )
        test_stems = all_stems
        fold_ids = list(cfg_doc["folds"])

    manifest: dict[str, object] = {
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "dry_run": bool(args.dry_run),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": args.device,
        "n_images": len(test_stems),
        "folds": fold_ids,
        "stages": {},
    }

    # Stage 1: Single-image disk geometry & 5-ch solar physics frontend
    t0 = time.time()
    geom_map: dict[str, tuple[float, float, float]] = {}
    phys_cache: dict[str, np.ndarray] = {}
    need_gpu_infer = (args.e120_maps is None) or (args.e121_maps is None)
    pre_geom = pd.read_csv(args.geometry).set_index("stem") if (args.geometry and args.geometry.is_file()) else None

    stage_names = ["physics"]
    n_infer_models = (len(fold_ids) if args.e120_maps is None else 0) + (len(fold_ids) if args.e121_maps is None else 0)
    if args.e120_maps is None:
        stage_names.extend([f"e120_f{fid}" for fid in fold_ids])
    if args.e121_maps is None:
        stage_names.extend([f"e121_f{fid}" for fid in fold_ids])
    stage_names.append("decode_mbr")
    total_units = len(test_stems) * (1 + n_infer_models + 1)
    prog = _ProgressTracker(total=total_units, stage_names=stage_names, every_sec=45.0)
    prog.set_stage(0)

    def _load_one_phys(st: str) -> tuple[str, np.ndarray | None, tuple[float, float, float]]:
        if need_gpu_infer:
            img_arr = np.array(Image.open(args.test_images / f"{st}.jpeg"))
            ch_arr, d_fit, _ = extract_physics_channels(img_arr)
            return st, ch_arr, (float(d_fit.cx), float(d_fit.cy), float(d_fit.r))
        if pre_geom is not None and st in pre_geom.index:
            g_tup = tuple(float(pre_geom.loc[st, k]) for k in ("cx", "cy", "r"))  # type: ignore[assignment]
            return st, None, g_tup  # type: ignore[return-value]
        img_arr = np.array(Image.open(args.test_images / f"{st}.jpeg"))
        d_fit = fit_disk(img_arr)
        return st, None, (float(d_fit.cx), float(d_fit.cy), float(d_fit.r))

    with ThreadPoolExecutor(max_workers=max(1, min(8, args.workers))) as phys_pool:
        for st, ch_opt, g_tup in phys_pool.map(_load_one_phys, test_stems):
            if ch_opt is not None:
                phys_cache[st] = ch_opt
            geom_map[st] = g_tup
            prog.update(1)
    prog.update(0, force=True)

    manifest["stages"]["geometry_physics"] = {
        "seconds": round(time.time() - t0, 2),
        "n_images": len(test_stems),
    }
    print(f"[STAGE 1] Single-image physics & disk fit for {len(test_stems)} images in {time.time()-t0:.1f}s", flush=True)

    # Stage 2: Pure Dual-Scale TopoFlowNet Inference (E-120 Global/Equiv + E-121a Native-512 Snake)
    device = torch.device(args.device)
    sec_per_model: dict[str, float] = {}
    t_inf0 = time.time()
    cur_stage = 1

    def _resolve_ckpt(family_prefix: str, fid: int) -> Path:
        for cand in (
            args.weights / f"{family_prefix}_f{fid}/best.pt",
            args.weights / f"fold{fid}/best.pt",
            args.weights / f"fold{fid}/run/best.pt",
        ):
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Missing checkpoint for {family_prefix} fold {fid} under {args.weights}")

    save_workers = max(2, min(4, args.workers))
    with ThreadPoolExecutor(max_workers=save_workers) as io_pool:
        pending_io: list[Future[None]] = []

        def _flush_io(max_q: int = 0) -> None:
            while len(pending_io) > max_q:
                pending_io.pop(0).result()

        if args.e120_maps is not None:
            e120_root = args.e120_maps
        else:
            e120_root = args.work / "maps_e120"
            for fid in fold_ids:
                prog.set_stage(cur_stage)
                cur_stage += 1
                m_t0 = time.time()
                m_name = f"exp120_topoflow_f{fid}"
                ckpt_p = _resolve_ckpt("exp120_topoflow", fid)
                ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
                model = TopoFlowNet(
                    encoder_name=ckpt["encoder"],
                    pretrained=False,
                    in_channels=5,
                    num_annotators=int(ckpt["num_annotators"]),
                    use_snake=bool(ckpt.get("use_snake", False)),
                ).to(device).eval()
                model.load_state_dict(ckpt["state_dict"])
                tokens = [(int(t), float(w)) for t, w in ckpt["annot_tokens"]]
                out_fdir = e120_root / f"fold{fid}/test_maps"
                for stem in test_stems:
                    cx, cy, r = geom_map[stem]
                    bp, sp, sepp, flp, _ = predict_single_image(
                        model, phys_cache[stem], {"cx": cx, "cy": cy, "r": r}, device, tokens, fast_val=args.dry_run
                    )
                    _flush_io(save_workers * 2)
                    pending_io.append(io_pool.submit(save_prediction_npz, out_fdir / f"{stem}.npz", bp, sp, sepp, flp))
                    prog.update(1)
                _flush_io(0)
                sec_per_model[m_name] = round(time.time() - m_t0, 2)
                print(f"  [INFER] {m_name} finished {len(test_stems)} images in {sec_per_model[m_name]:.1f}s", flush=True)
                prog.update(0, force=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if args.e121_maps is not None:
            e121_root = args.e121_maps
        else:
            e121_root = args.work / "maps_e121"
            for fid in fold_ids:
                prog.set_stage(cur_stage)
                cur_stage += 1
                m_t0 = time.time()
                m_name = f"exp121_snake512_f{fid}"
                ckpt_p = _resolve_ckpt("exp121_snake512", fid)
                ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
                model = TopoFlowNet(
                    encoder_name=ckpt["encoder"],
                    pretrained=False,
                    in_channels=5,
                    num_annotators=int(ckpt["num_annotators"]),
                    use_snake=bool(ckpt.get("use_snake", True)),
                ).to(device).eval()
                model.load_state_dict(ckpt["state_dict"])
                tokens = [(int(t), float(w)) for t, w in ckpt["annot_tokens"]]
                out_fdir = e121_root / f"fold{fid}/test_maps"
                for stem in test_stems:
                    cx, cy, r = geom_map[stem]
                    bp, sp, sepp, flp, _ = predict_single_image_snake512(
                        model, phys_cache[stem], {"cx": cx, "cy": cy, "r": r}, device, tokens, fast_val=args.dry_run
                    )
                    _flush_io(save_workers * 2)
                    pending_io.append(io_pool.submit(save_prediction_npz, out_fdir / f"{stem}.npz", bp, sp, sepp, flp))
                    prog.update(1)
                _flush_io(0)
                sec_per_model[m_name] = round(time.time() - m_t0, 2)
                print(f"  [INFER] {m_name} finished {len(test_stems)} images in {sec_per_model[m_name]:.1f}s", flush=True)
                prog.update(0, force=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    manifest["stages"]["infer"] = {
        "seconds": round(time.time() - t_inf0, 2),
        "seconds_per_model": sec_per_model,
    }

    # Stage 3: Geodesic Euler Spine-Flow Decoding + Exact Conflict-Graph Lagrangian MWIS / PQ-MBR
    prog.set_stage(cur_stage)
    t_dec0 = time.time()
    scfg = cfg_doc["solver_config"]
    cal_120 = scfg["cal_e120"]
    cal_121 = scfg["cal_e121"]
    mbr_cfg = scfg["dual_mbr"]

    def _find_map(root: Path, fid: int, stem: str) -> str:
        for cand in (
            root / f"fold{fid}/test_maps/{stem}.npz",
            root / f"fold{fid}/run/test_maps/{stem}.npz",
            root / f"fold{fid}/oof_maps/{stem}.npz",
            root / f"fold{fid}/run/oof_maps/{stem}.npz",
        ):
            if cand.is_file():
                return str(cand)
        raise FileNotFoundError(f"Missing map for fold {fid} stem {stem} under {root}")

    jobs = []
    for ts in test_stems:
        p120 = [_find_map(e120_root, fid, ts) for fid in fold_ids]
        p121 = [_find_map(e121_root, fid, ts) for fid in fold_ids]
        jobs.append((
            ts, p120, p121, geom_map[ts], cal_120, cal_121,
            [float(mbr_cfg["w121"])], [float(mbr_cfg["tb"])], float(mbr_cfg["gate"]),
        ))

    n_workers = 1 if args.dry_run else args.workers
    decoded_cache: dict[str, dict] = {}
    if n_workers <= 1:
        for jb in jobs:
            st, ent = _extract_stem_candidates_worker(jb)
            decoded_cache[st] = ent
            prog.update(1)
    else:
        with Pool(processes=n_workers) as pool:
            for st, ent in pool.imap_unordered(_extract_stem_candidates_worker, jobs, chunksize=4):
                decoded_cache[st] = ent
                prog.update(1)

    final_rles: dict[str, list[str]] = {}
    field_rles: dict[str, list[str]] = {}
    for ts in test_stems:
        rf, rm = _solve_dual_stem_from_cache(
            decoded_cache[ts],
            w121=float(mbr_cfg["w121"]),
            tb=float(mbr_cfg["tb"]),
            gate=float(mbr_cfg["gate"]),
            fpc_field=float(mbr_cfg.get("fpc_field", 1.0)),
            fpc_mbr=float(mbr_cfg["fpc_mbr"]),
            w_ref_dual=float(mbr_cfg["w_ref_dual"]),
            w_ref_121=float(mbr_cfg["w_ref_121"]),
            w_ref_120=float(mbr_cfg["w_ref_120"]),
        )
        field_rles[ts] = rf
        final_rles[ts] = rm

    sub_dir = args.work / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    df_sub = rles_by_stem_to_df(final_rles)
    val_rep = validate_submission(df_sub, test_stems)
    out_csv = sub_dir / cfg_doc.get("output_filename", "sub_e121_pure_dual_mbr_5f.csv")
    write_submission(df_sub, out_csv)
    sub_sha = sha256_file(out_csv)

    df_field = rles_by_stem_to_df(field_rles)
    validate_submission(df_field, test_stems)
    write_submission(df_field, sub_dir / "sub_e121_pure_dual_field_5f.csv")

    manifest["stages"]["decode_mbr"] = {
        "seconds": round(time.time() - t_dec0, 2),
        "output_csv": str(out_csv),
        "sha256": sub_sha,
        "validate": val_rep,
    }
    print(
        f"[STAGE 3] Decoded {val_rep['n_rows']} disjoint instances across {val_rep['n_images_with_rows']}/{len(test_stems)} "
        f"images in {manifest['stages']['decode_mbr']['seconds']:.1f}s | sha256={sub_sha[:16]}...",
        flush=True,
    )

    if args.gt_csv is not None and args.gt_csv.is_file():
        t_ev0 = time.time()
        gt_df = pd.read_csv(args.gt_csv)
        gt_df["stem"] = gt_df["filament_id"].str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0]
        gt_sub = gt_df[gt_df["stem"].isin(set(test_stems))][["filament_id", "segmentation_rle"]].copy()
        res_pq = evaluate(gt_sub, df_sub, distributions=True)
        rub_sum = RM.stats(res_pq)
        eval_summary = {
            "pq_summary": {**res_pq.summary(), "PQ_exact": round(float(res_pq.pq), 6)},
            "rubric": {k: round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v for k, v in rub_sum.items()},
            "seconds": round(time.time() - t_ev0, 2),
        }
        manifest["stages"]["evaluate"] = eval_summary
        print(f"[STAGE 4 EVAL] PQ={res_pq.pq:.5f} SQ={res_pq.sq:.4f} RQ={res_pq.rq:.4f} | Rubric mDice={rub_sum['mean_dice']:.4f}", flush=True)

    manifest["total_seconds"] = round(time.time() - t_all, 2)
    manifest["seconds_per_image"] = round((time.time() - t_all) / max(1, len(test_stems)), 3)
    (args.work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.verify and not args.dry_run and "expected" in cfg_doc:
        exp_sha = cfg_doc["expected"].get("sha256")
        if exp_sha:
            assert sub_sha == exp_sha, f"SHA256 mismatch: got {sub_sha}, expected {exp_sha}"
            print("[VERIFY] Exact submission SHA256 verified 100%.", flush=True)

    prog.done()


if __name__ == "__main__":
    main()
