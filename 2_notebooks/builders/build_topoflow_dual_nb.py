"""Builds 2_notebooks/SFSC_06_topoflow_dual_pipeline.ipynb (spec: 2_notebooks/specs/SFSC_06_topoflow_dual_pipeline.spec.md).

Self-contained end-to-end demonstration notebook aligned with all Organizer November requirements:
  1. Automatic checkpoint download + MD5 verification (`7_reproduce/download_weights.py`).
  2. Single-image 5-channel Solar Physics Pre-processing (`sfsc.topoflow_physics.extract_physics_channels`).
  3. Toggleable Train-from-Scratch (`SFSC_NB_TRAIN_FROM_SCRATCH=1`) vs. Pretrained 10-Model Dual-Scale TopoFlowNet.
  4. Topological Spine-Flow + Geodesic Euler Integration + Exact Conflict-Graph Lagrangian MWIS/MBR decoding.
  5. Official Panoptic Quality + Rubric distribution evaluation on any annotated set (plus 707-image OOF reproduction).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

NAME = "SFSC_06_topoflow_dual_pipeline.ipynb"
cells = []


def _cell_id(idx: int, src: str) -> str:
    return hashlib.sha1(f"{idx}:{src}".encode("utf-8")).hexdigest()[:12]


def md(s: str) -> None:
    cells.append({
        "id": _cell_id(len(cells), s),
        "cell_type": "markdown",
        "metadata": {},
        "source": s.splitlines(keepends=True),
    })


def code(s: str) -> None:
    cells.append({
        "id": _cell_id(len(cells), s),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": s.splitlines(keepends=True),
    })

md("""# SFSC 2026 — End-to-End Dual-Scale TopoFlowNet Pipeline (Physics → Train/Load → Spine-Flow MWIS → Evaluation)

**Purpose:** Demonstrate our complete 10-model **Pure Dual-Scale TopoFlowNet** system end to end in a single self-contained notebook:
1. **Physics Pre-processing (`Stage 1`):** Single-image solar disk circle fit + 5-channel chromospheric physics frontend ($I_{\\mathrm{norm}}$, quiet-Sun radial profile absorption $\\Delta I_{\\mathrm{QS}}(r)$, mirror-padded local absorption $\\Delta I_{\\mathrm{loc}}$, multi-scale Hessian dark-valley ridge curvature, heliocentric foreshortening cosine $\\mu = \\sqrt{1 - (r/R_{\\odot})^2}$).
2. **Training or Pretrained Loading (`Stage 2`):** Toggle `TRAIN_FROM_SCRATCH` (`False` by default to auto-download and verify the 10 released checkpoints via `7_reproduce/download_weights.py`; set `True` to train `TopoFlowNet` directly from raw MAGFiLO annotations).
3. **Dual-Scale Topological Inference & Exact MWIS Decoding (`Stage 3`):** Combines 5 folds of **Global/Equivariant `TopoFlowNet`** (`E-120`, continuous $360^\\circ$ vector-field equivariance) and 5 folds of **Native-512 `Snake-TopoFlowNet`** (`E-121a`, GroupNorm Dynamic Snake Convolutions + lossless dihedral $D_4$ + symmetric barb weighting), integrates the 5-channel **Topological Spine-Flow** field via Geodesic Euler steps, and selects disjoint instances via **Exact Conflict-Graph Lagrangian Maximum-Weight Independent Set (`pq_mbr_solve`)**.
4. **Qualitative & Official Rubric Evaluation (`Stages 4–5`):** Multi-panel visualization of topological fields and fine chromospheric barbs on a test image, plus official **Panoptic Quality (PQ)** and **Rubric** (`IoU`, `Dice`, `1:n` fragmentation, `n:1` merging) evaluation on any annotated dataset.

- **Runtime:** CUDA GPU recommended for full 180-image inference (measured `115.3 min` end-to-end across all 10 models × 180 images on 1× NVIDIA A100 40 GB: `3.24 s/img` per `E-120` fold, `4.26 s/img` per `E-121a` fold, `0.73 s/img` CPU Euler+MWIS); `DRY_RUN=1` (default, 3 images × Fold-0 Dual-Scale pair) runs in `25 s` on 1× A100 GPU or `~32 s` on CPU.
- **Prerequisites:** `pip install -r 7_reproduce/requirements.txt` and Kaggle competition images under `1_data/raw/MAGFiLO_1.0_Kaggle_2026/`.
- **Switches (Cell 2 or environment variables):**
  - `SFSC_NB_DRY_RUN=1` (default `1` for quick 3-image verification; set `0` for full 180-image submission)
  - `SFSC_NB_AUTO_DOWNLOAD=1` (default `1`: automatically fetches missing checkpoints from GitHub Release)
  - `SFSC_NB_TRAIN_FROM_SCRATCH=0` (set `1` to run a live training pass from raw annotations)
  - `SFSC_NB_EVAL_GT_CSV` (optional path to ground-truth CSV to score predictions on any labelled dataset)
""")

code(r"""# === Cell 1: Setup & Imports ===
import hashlib
import importlib.metadata as im
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image


def _find_root(p: Path = Path.cwd().resolve()) -> Path:
    for q in [p, *p.parents]:
        if (q / "7_reproduce/run_e121_dual_final.py").exists():
            return q
    raise FileNotFoundError("Open this notebook inside the repository root (7_reproduce/run_e121_dual_final.py not found).")


ROOT = _find_root()
sys.path[:0] = [str(ROOT / "3_src"), str(ROOT / "3_src/scripts")]

from sfsc.geometry import disk_mask
from sfsc.metrics import evaluate
from sfsc.rle import decode_rle
from sfsc.topoflow_physics import extract_physics_channels
from train_e120_topoflow import load_prediction_npz
import rubric_metrics as RM

_cell_times = {}
_t0 = None


def cell_start(name: str) -> None:
    global _t0
    _t0 = time.time()
    print(f"▶ {name}")


def cell_end(name: str) -> None:
    dt = time.time() - _t0
    _cell_times[name] = dt
    print(f"✓ {name} — {dt:.1f}s")


print(f"Repository root: {ROOT}")
""")

code(r"""# === Cell 2: Configuration & Pipeline Switches ===
cell_start("Cell 2: Configuration")
env = os.environ.get

DRY_RUN = env("SFSC_NB_DRY_RUN", "1") == "1"                  # Set '0' for full 10-model x 180-image submission run
AUTO_DOWNLOAD = env("SFSC_NB_AUTO_DOWNLOAD", "1") == "1"      # Automatically download missing weights from GitHub Release
TRAIN_FROM_SCRATCH = env("SFSC_NB_TRAIN_FROM_SCRATCH", "0") == "1"
TRAIN_EPOCHS = int(env("SFSC_NB_TRAIN_EPOCHS", "1" if DRY_RUN else "26"))
RUN_INFER = env("SFSC_NB_RUN", "1") == "1"
CHECK_MD5 = env("SFSC_NB_MD5", "1") == "1"

DATA_ROOT = (ROOT / env("SFSC_NB_DATA", "1_data/raw/MAGFiLO_1.0_Kaggle_2026")).resolve()
ASSETS_DIR = (ROOT / env("SFSC_NB_ASSETS", "1_data/processed")).resolve()
TEST_IMAGES = ( Path(env("SFSC_NB_IMAGES")) if env("SFSC_NB_IMAGES") else (DATA_ROOT / "test/test_images") ).resolve()
EVAL_GT_CSV = Path(env("SFSC_NB_EVAL_GT_CSV")).resolve() if env("SFSC_NB_EVAL_GT_CSV") else None
OOF_EVAL_CSV = (ROOT / env("SFSC_NB_OOF_CSV", "5_outputs/exp121_snake512/final_eval/oof_pure_dual_mbr_10m.csv")).resolve()

WEIGHTS_DIR = (ROOT / env("SFSC_NB_WEIGHTS", "4_models")).resolve()
WORK_DIR = (ROOT / env("SFSC_NB_WORK", f"runs/topoflow_dual_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if DRY_RUN else ''}")).resolve()
WORKERS = int(env("SFSC_NB_WORKERS", "4"))

CFG_PATH = ROOT / "7_reproduce/final_e121_dual.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

import torch
HAS_CUDA = torch.cuda.is_available()
DEVICE = env("SFSC_NB_DEVICE", "cuda" if HAS_CUDA else "cpu")
GPU_DESC = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)" if HAS_CUDA else "CPU"

NEEDED_MODELS = (
    ["exp120_topoflow_f0", "exp121_snake512_f0"]
    if DRY_RUN
    else [m["name"] for m in CFG["models"]]
)
STEM_DEMO = env("SFSC_NB_STEM", "")

print(f"Device             : {DEVICE} ({GPU_DESC})")
print(f"DRY_RUN            : {DRY_RUN} (models needed: {len(NEEDED_MODELS)})")
print(f"AUTO_DOWNLOAD      : {AUTO_DOWNLOAD}")
print(f"TRAIN_FROM_SCRATCH : {TRAIN_FROM_SCRATCH}")
print(f"Images directory   : {TEST_IMAGES}")
print(f"Weights directory  : {WEIGHTS_DIR}")
print(f"Work directory     : {WORK_DIR}")
cell_end("Cell 2: Configuration")
""")

code(r"""# === Cell 3: Environment Verification ===
cell_start("Cell 3: Environment Verification")
req_lines = [
    ln.strip().split("==")
    for ln in (ROOT / "7_reproduce/requirements.txt").read_text(encoding="utf-8").splitlines()
    if "==" in ln and not ln.lstrip().startswith("#")
]
pkg_rows = []
for pkg, req_ver in req_lines:
    try:
        inst_ver = im.version(pkg)
    except im.PackageNotFoundError:
        inst_ver = None
    pkg_rows.append({
        "package": pkg,
        "pinned": req_ver,
        "installed": inst_ver,
        "exact_match": bool(inst_ver) and inst_ver.split("+")[0] == req_ver,
    })
print(pd.DataFrame(pkg_rows).to_string(index=False))
cell_end("Cell 3: Environment Verification")
""")

code(r"""# === Cell 4: Inputs Check & Automatic Checkpoint Download ===
cell_start("Cell 4: Inputs & Auto-Download")
assert TEST_IMAGES.is_dir(), f"Image directory not found: {TEST_IMAGES}"
n_imgs = len(list(TEST_IMAGES.glob("*.jpeg")))
assert n_imgs > 0, f"No .jpeg files found in {TEST_IMAGES}"

wman_path = ROOT / "7_reproduce/weights_manifest.json"
wman = json.loads(wman_path.read_text(encoding="utf-8")) if wman_path.is_file() else {"checkpoints": []}
man_by_name = {c["name"]: c for c in wman.get("checkpoints", [])}

missing_ckpts = [
    m for m in NEEDED_MODELS
    if not (WEIGHTS_DIR / m / "best.pt").is_file()
]

if missing_ckpts and not TRAIN_FROM_SCRATCH:
    if AUTO_DOWNLOAD and all(m in man_by_name for m in missing_ckpts):
        print(f"Auto-downloading {len(missing_ckpts)} missing checkpoints into {WEIGHTS_DIR} ...")
        dl_cmd = [
            sys.executable, "-u", str(ROOT / "7_reproduce/download_weights.py"),
            "--out", str(WEIGHTS_DIR),
            "--only", ",".join(NEEDED_MODELS),
        ]
        subprocess.run(dl_cmd, cwd=ROOT, check=True)
    else:
        raise FileNotFoundError(
            f"Missing checkpoints {missing_ckpts} under {WEIGHTS_DIR}. "
            "Enable SFSC_NB_AUTO_DOWNLOAD=1 or provide --weights."
        )

if CHECK_MD5 and not TRAIN_FROM_SCRATCH:
    for m in NEEDED_MODELS:
        p = WEIGHTS_DIR / m / "best.pt"
        if m in man_by_name and p.is_file():
            c = man_by_name[m]
            assert p.stat().st_size == c["size"], f"Size mismatch on {p}"
            h = hashlib.md5()
            with p.open("rb") as f:
                for blk in iter(lambda: f.read(1 << 23), b""):
                    h.update(blk)
            assert h.hexdigest() == c["md5"], f"MD5 mismatch on {p}"

print(f"Verified {n_imgs} input images and {len(NEEDED_MODELS)} required TopoFlowNet checkpoints.")
cell_end("Cell 4: Inputs & Auto-Download")
""")

code(r"""# === Cell 5: Stage 1 — Single-Image 5-Channel Solar Physics Pre-processing ===
cell_start("Cell 5: Stage 1 Physics Frontend")
import matplotlib.pyplot as plt

sample_stems = sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg"))
demo_stem = STEM_DEMO if (STEM_DEMO and (TEST_IMAGES / f"{STEM_DEMO}.jpeg").is_file()) else sample_stems[0]
raw_img = np.asarray(Image.open(TEST_IMAGES / f"{demo_stem}.jpeg"))
phys_5ch, disk_fit, valid_disk = extract_physics_channels(raw_img)

fig_p, axes_p = plt.subplots(1, 6, figsize=(18, 3.4), layout="constrained")
titles_p = [
    f"Raw Hα ({demo_stem})",
    "Ch0: Disk Intensity I_norm",
    "Ch1: Quiet-Sun Flattened ΔI_QS",
    "Ch2: Mirror Local Contrast ΔI_loc",
    "Ch3: Hessian Dark-Valley Ridge",
    "Ch4: Foreshortening Cosine μ",
]
panels_p = [raw_img, phys_5ch[0], phys_5ch[1], phys_5ch[2], phys_5ch[3], phys_5ch[4]]
cmaps_p = ["gray", "magma", "coolwarm", "coolwarm", "inferno", "viridis"]
for ax, arr, ttl, cm in zip(axes_p, panels_p, titles_p, cmaps_p):
    ax.imshow(arr, cmap=cm)
    ax.set_title(ttl, fontsize=9)
    ax.set_axis_off()
WORK_DIR.mkdir(parents=True, exist_ok=True)
(WORK_DIR / "figures").mkdir(parents=True, exist_ok=True)
fig_p.savefig(WORK_DIR / "figures" / f"stage1_physics_{demo_stem}.png", dpi=100)
plt.show()
print(f"Disk fit: center=({disk_fit.cx:.1f}, {disk_fit.cy:.1f}), radius={disk_fit.r:.1f} px | Physics tensor={phys_5ch.shape} {phys_5ch.dtype}")
cell_end("Cell 5: Stage 1 Physics Frontend")
""")

code(r"""# === Cell 6: Stage 2 — Train from Scratch (Optional Toggle) or Use Pretrained Checkpoints ===
cell_start("Cell 6: Stage 2 Training / Checkpoint Toggle")
if TRAIN_FROM_SCRATCH:
    if not (ASSETS_DIR / "folds_cvA_v1.csv").is_file():
        print("Building processed fold/geometry CSVs from raw Kaggle download ...")
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "7_reproduce/make_processed.py"), "--data", str(DATA_ROOT), "--out", str(ASSETS_DIR)],
            cwd=ROOT, check=True,
        )
    scratch_out = WORK_DIR / "trained_from_scratch/exp121_snake512_f0"
    scratch_cache = WORK_DIR / "scratch_cache"
    train_cmd = [
        sys.executable, "-u", str(ROOT / "3_src/scripts/train_e121_snake512.py"),
        "--data-root", str(DATA_ROOT),
        "--assets", str(ASSETS_DIR),
        "--cache-dir", str(scratch_cache),
        "--out", str(scratch_out),
        "--outer-fold", "0",
        "--epochs", str(TRAIN_EPOCHS),
        "--device", DEVICE,
    ]
    if DRY_RUN:
        train_cmd.append("--smoke")
    print("Running live Snake-TopoFlowNet training:", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=ROOT, check=True)
    print(f"Live training completed -> {scratch_out / 'best.pt'}")
else:
    print(f"TRAIN_FROM_SCRATCH=False -> using verified pretrained Dual-Scale TopoFlowNet weights in {WEIGHTS_DIR}.")
cell_end("Cell 6: Stage 2 Training / Checkpoint Toggle")
""")

code(r"""# === Cell 7: Stage 3 — Dual-Scale TopoFlowNet Inference & Exact Lagrangian MWIS Decoding ===
cell_start("Cell 7: Stage 3 Dual-Scale Pipeline")
if RUN_INFER:
    pipe_cmd = [
        sys.executable, "-u", str(ROOT / "7_reproduce/run_e121_dual_final.py"),
        "--config", str(CFG_PATH),
        "--test-images", str(TEST_IMAGES),
        "--weights", str(WEIGHTS_DIR),
        "--work", str(WORK_DIR),
        "--device", DEVICE,
        "--workers", str(WORKERS),
    ]
    if DRY_RUN:
        pipe_cmd.append("--dry-run")
    if EVAL_GT_CSV is not None and EVAL_GT_CSV.is_file():
        pipe_cmd += ["--gt-csv", str(EVAL_GT_CSV)]
    print("Executing:", " ".join(pipe_cmd), flush=True)
    proc = subprocess.Popen(pipe_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log_lines = []
    for line in proc.stdout:
        log_lines.append(line)
        print(line, end="", flush=True)
    rc = proc.wait()
    (WORK_DIR / "pipeline_run.log").write_text("".join(log_lines), encoding="utf-8")
    assert rc == 0, f"run_e121_dual_final.py failed with exit code {rc}"

MANIFEST = json.loads((WORK_DIR / "manifest.json").read_text(encoding="utf-8"))
_raw_sub = Path(MANIFEST["stages"]["decode_mbr"]["output_csv"])
_local_sub = WORK_DIR / "submissions" / _raw_sub.name
SUB_CSV = _local_sub if _local_sub.is_file() else (_raw_sub if _raw_sub.is_absolute() else (ROOT / _raw_sub).resolve())
SUB_DF = pd.read_csv(SUB_CSV)
print(f"Generated disjoint submission: {SUB_CSV} ({len(SUB_DF)} instances across {MANIFEST['n_images']} images)")
cell_end("Cell 7: Stage 3 Dual-Scale Pipeline")
""")

code(r"""# === Cell 8: Stage 4 — Topological Spine-Flow Fields & Barb-Preserving Instance Overlay ===
cell_start("Cell 8: Stage 4 Visualization")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

e121_map_p = WORK_DIR / f"maps_e121/fold0/test_maps/{demo_stem}.npz"
e120_map_p = WORK_DIR / f"maps_e120/fold0/test_maps/{demo_stem}.npz"

if e121_map_p.is_file() and e120_map_p.is_file():
    b120, s120, sep120, fl120 = load_prediction_npz(e120_map_p)
    b121, s121, sep121, fl121 = load_prediction_npz(e121_map_p)
    w121_vis = float(CFG["solver_config"]["dual_mbr"]["w121"])
    w120_vis = 1.0 - w121_vis
    body_vis = w120_vis * b120 + w121_vis * b121
    spine_vis = w120_vis * s120 + w121_vis * s121
    sep_vis = w120_vis * sep120 + w121_vis * sep121
    flow_vis = w120_vis * fl120 + w121_vis * fl121
else:
    body_vis = np.zeros_like(raw_img, dtype=np.float32)
    spine_vis = np.zeros_like(raw_img, dtype=np.float32)
    sep_vis = np.zeros_like(raw_img, dtype=np.float32)
    flow_vis = np.zeros((2, *raw_img.shape[:2]), dtype=np.float32)

inst_rows = SUB_DF[SUB_DF["filament_id"].str.rsplit("_", n=1).str[0] == demo_stem]
inst_lab = np.zeros(raw_img.shape[:2], dtype=np.int32)
for k, rle_str in enumerate(inst_rows["segmentation_rle"], 1):
    inst_lab[decode_rle(rle_str)] = k

pal = np.random.default_rng(42).uniform(0.32, 0.98, (2048, 3))
rgba = np.zeros(inst_lab.shape + (4,), dtype=np.float32)
rgba[..., :3] = pal[inst_lab % len(pal)]
rgba[..., 3] = (inst_lab > 0) * 0.65

# Pick largest detected filament for high-resolution barb & Euler-flow zoom window
if inst_lab.max() > 0:
    areas = [(inst_lab == k).sum() for k in range(1, int(inst_lab.max()) + 1)]
    k_big = int(np.argmax(areas)) + 1
    ys, xs = np.nonzero(inst_lab == k_big)
    cy_z, cx_z = int(ys.mean()), int(xs.mean())
else:
    cy_z, cx_z = raw_img.shape[0] // 2, raw_img.shape[1] // 2
hw = 192
y0 = max(0, min(raw_img.shape[0] - 2 * hw, cy_z - hw))
x0 = max(0, min(raw_img.shape[1] - 2 * hw, cx_z - hw))
zw = (slice(y0, y0 + 2 * hw), slice(x0, x0 + 2 * hw))

fig_v, ax_v = plt.subplots(2, 3, figsize=(16, 10.5), layout="constrained")
ax_v[0, 0].imshow(raw_img, cmap="gray")
ax_v[0, 0].add_patch(Circle((disk_fit.cx, disk_fit.cy), disk_fit.r, fill=False, color="yellow", lw=0.9))
ax_v[0, 0].add_patch(Rectangle((x0, y0), 2 * hw, 2 * hw, fill=False, color="cyan", lw=1.6))
ax_v[0, 0].set_title(f"(a) Full-Disk Hα & Zoom Box ({demo_stem})")

ax_v[0, 1].imshow(body_vis, cmap="magma", vmin=0, vmax=1)
ax_v[0, 1].set_title("(b) Dual-Scale Calibrated Body Probability")

ax_v[0, 2].imshow(raw_img, cmap="gray")
ax_v[0, 2].imshow(rgba)
ax_v[0, 2].set_title(f"(c) Exact MWIS Disjoint Output ({int(inst_lab.max())} Filaments)")

ax_v[1, 0].imshow(spine_vis[zw], cmap="inferno", vmin=0, vmax=1)
ax_v[1, 0].contour(sep_vis[zw] > 0.35, colors="cyan", linewidths=1.0)
ax_v[1, 0].set_title("(d) Zoom: Medial Spine + Contact Boundary (Cyan)")

ax_v[1, 1].imshow(body_vis[zw], cmap="gray", vmin=0, vmax=1)
step_q = 10
gy, gx = np.mgrid[0 : 2 * hw : step_q, 0 : 2 * hw : step_q]
fx_sub = flow_vis[0, y0 : y0 + 2 * hw : step_q, x0 : x0 + 2 * hw : step_q]
fy_sub = flow_vis[1, y0 : y0 + 2 * hw : step_q, x0 : x0 + 2 * hw : step_q]
act_q = body_vis[y0 : y0 + 2 * hw : step_q, x0 : x0 + 2 * hw : step_q] > 0.20
ax_v[1, 1].quiver(gx[act_q], gy[act_q], fx_sub[act_q], fy_sub[act_q], color="lime", scale=22, width=0.004)
ax_v[1, 1].set_title("(e) Zoom: Geodesic Euler Spine-Flow Field (vx, vy)")

ax_v[1, 2].imshow(raw_img[zw], cmap="gray")
ax_v[1, 2].imshow(rgba[zw])
ax_v[1, 2].set_title("(f) Zoom: Native Stride-1/1 + Snake Barb Contour")

for a in ax_v.ravel():
    a.set_axis_off()

fig_out = WORK_DIR / "figures" / f"pipeline_topoflow_{demo_stem}.png"
fig_v.savefig(fig_out, dpi=110)
plt.show()
print(f"Saved multi-panel figure -> {fig_out}")
cell_end("Cell 8: Stage 4 Visualization")
""")

code(r"""# === Cell 9: Stage 5 — Official Panoptic Quality & Rubric Evaluation ===
cell_start("Cell 9: Stage 5 Official Evaluation")
if "evaluate" in MANIFEST["stages"]:
    ev = MANIFEST["stages"]["evaluate"]
    print("=== Evaluation on Provided Ground-Truth Dataset ===")
    print(json.dumps(ev, indent=2))
elif OOF_EVAL_CSV.is_file() and (ASSETS_DIR / "gt_train_v1.csv").is_file():
    print(f"=== Evaluating 5-Fold Out-of-Fold Predictions ({OOF_EVAL_CSV.name}) on All 707 Training Images ===")
    gt_all = pd.read_csv(ASSETS_DIR / "gt_train_v1.csv")[["filament_id", "segmentation_rle"]]
    oof_pred = pd.read_csv(OOF_EVAL_CSV)
    res_oof = evaluate(gt_all, oof_pred, distributions=True)
    rub_oof = RM.stats(res_oof)
    print(f"Official 707-Image OOF PQ : {res_oof.pq:.5f} (SQ={res_oof.sq:.4f}, RQ={res_oof.rq:.4f})")
    print(f"Counts                    : TP={res_oof.tp} | FP={res_oof.fp} | FN={res_oof.fn} | 1:N={res_oof.summary()['gt_1toN']} | N:1={res_oof.summary()['pred_Nto1']}")
    print(f"Rubric Distributions      : mean IoU={rub_oof['mean_iou']:.4f} | mean Dice={rub_oof['mean_dice']:.4f} | share IoU>0.5={rub_oof['share_iou_gt_0p5']:.4f}")
else:
    print("No ground-truth CSV provided (test set is unlabelled). Pass SFSC_NB_EVAL_GT_CSV=<path> to evaluate any annotated split.")
cell_end("Cell 9: Stage 5 Official Evaluation")
""")

code(r"""# === Cell 10: Summary ===
cell_start("Summary")
val_info = MANIFEST["stages"]["decode_mbr"]["validate"]
print(f'''notebook         : SFSC_06_topoflow_dual_pipeline.ipynb
DRY_RUN          : {DRY_RUN}
device           : {DEVICE} ({GPU_DESC})
status           : ok
images_processed : {MANIFEST["n_images"]}
instances_kept   : {val_info["n_rows"]} across {val_info["n_images_with_rows"]} non-empty images
pixel_disjoint   : {val_info["overlap_checked"]}
total_seconds    : {MANIFEST["total_seconds"]}s ({MANIFEST["seconds_per_image"]}s/img)
submission_csv   : {SUB_CSV}
submission_sha256: {MANIFEST["stages"]["decode_mbr"]["sha256"]}
''')
cell_end("Summary")
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "file_extension": ".py"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = Path(os.environ.get("SFSC_NB_OUT") or (Path(__file__).resolve().parents[1] / NAME))
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out_path, "cells:", len(cells))
