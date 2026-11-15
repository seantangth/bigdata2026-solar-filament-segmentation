# Solar Filament Segmentation Challenge 2026 — Reproducible Solution

Instance segmentation of solar filaments in full-disk GONG H-alpha observations (IEEE BigData Cup 2026,
[Kaggle: filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)).

One command reproduces our final **10-model Pure Dual-Scale `TopoFlowNet`** submission (`sub_e121_pure_dual_mbr_10m_5f.csv`, Public LB **0.40**) directly from the 180 test images and 10 MD5-verified checkpoints:

```bash
python 7_reproduce/run_e121_dual_final.py \
    --test-images 1_data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --weights 4_models --work runs/e121_final
```

## Results

| System / Submission | Models | Representation & Decoding | CV-A OOF PQ (707 train images) | Multi-296 $n{:}1$ / Barb Density | Public LB (≈90 test images) |
|---|---|---|---|---|---|
| **Final submission — Pure Dual-Scale `TopoFlowNet`** `sub_e121_pure_dual_mbr_10m_5f.csv` | 10 pure `TopoFlowNet`s from 5-fold CV (`5 × Global-1024 E-120` + `5 × Native-512 Snake E-121a`) | 5-ch solar physics frontend, continuous spine-flow field, damped Euler basin tracking, exact conflict-graph MWIS (zero tabular GBDT, zero morphological post-dilation) | **0.4624** strict nested cross-fit (**0.4638** 5-fold consensus) | **1.8%** / **2.61** per 100 px | **0.40** |
| Ablation — Global-1024 family alone (`E-120`) | the 5 `E-120` models of the final ensemble | Same continuous spine-flow + exact MWIS | 0.4571 | 2.0% / 1.88 per 100 px | 0.39 |
| Ablation — Native-512 Snake family alone (`E-121a`) | the 5 `E-121a` models of the final ensemble | Same continuous spine-flow + exact MWIS | 0.4553 (nested) | — | — |

Comparisons with our earlier U-Net baselines and a 20-model cross-family hybrid are reported in the technical report (Table I); their code and weights are intentionally not part of this repository, which contains only the final submission.

### Key Scientific Findings

- **PQ & CV-A Evaluation Protocol**: Official Panoptic Quality (`IoU > 0.5`, pooled across all 1,154 observer–image rows on 707 training images). `CV-A` (`1_data/processed/folds_cvA_v1.csv`) is a 5-fold `GroupKFold` grouped by image stem and stratified by `(observer count × GONG station)`. All Pure Dual-Scale `TopoFlowNet` fusion and MWIS hyperparameters are evaluated via **strict nested 5-fold cross-fitting** (fit on 4 training folds, score on the held-out fold).
- **Multi-Annotator Uncertainty**: Across all 1,196 ordered observer pairs on the 296 multi-annotator training images, human-vs.-human PQ is **0.340** (`SQ = 0.634`, `RQ = 0.537`, median best-match `IoU ≈ 0.52`, `17.8%` zero-overlap misses). Under the exact PQ differential, retaining a candidate matched by $k$ of $n$ observers with mean overlap $\bar{\imath}$ improves global PQ iff $\mathbb{E}[(k/n)\bar{\imath}] > \mathrm{PQ}/2 \approx 0.23$.
- **Dual-Scale Complementarity & Native Barb Fidelity**: Whole-disk `E-120` excels on dense active-belt multi-observer scenes (`PQ = 0.4621`), whereas native-resolution `E-121a` gains **`+0.0071` PQ** on single-observer thin peripheral filaments (`0.4558` vs. `0.4487`) and raises matched `mDice` to `0.7369`. Combining all 10 pure models advances nested OOF PQ by **`+0.0053`** (`95% CI [+0.0014, +0.0091]`, `p = 0.0035`) and Public LB from **0.39** to **0.40**. On 1,253 common-TP filaments across 120 multi-annotator images, Pure Dual-Scale `TopoFlowNet` cuts inter-filament false merges (`n:1`) by **33.9%** (`156` vs. `236` on all 707 images) and raises natural skeleton branch density by **67.7%** (`2.61` vs. `1.56` branch points / 100 px; Multiscale IoU `0.840` vs. `0.837`; thin-structure recall `0.618` vs. `0.597`, exceeding human second-observer recall `0.586`) with **zero morphological post-dilation**.

## Pure Dual-Scale `TopoFlowNet` Pipeline

```
Single GONG H-alpha image (2048×2048)
 ├─ 1. Single-image heliocentric 5-ch physics frontend                               sfsc/topoflow_physics.py
 │      solar-limb circle fit (cx, cy, R_sun) + valid disk mask (r <= R_sun + 8 px)
 │      [I_norm, radial quiet-Sun ΔI_QS(r), limb-mirrored ΔI_loc, multi-scale Hessian ridge, μ = sqrt(1 - (r/R_sun)^2)]
 ├─ 2. 10 Pure TopoFlowNet checkpoints (5 folds × 2 complementary scales)            sfsc/topoflow_net.py
 │      ├─ 5 × Global-1024 TopoFlowNet (E-120): ConvNeXt-B + parallel stride-1 fine stem + Annotator FiLM
 │      └─ 5 × Native-512 Snake-TopoFlowNet (E-121a): + GroupNorm Dynamic Snake Convs (1/16..1/4) + 1×7/7×1 BarbStrip (1/2 decoder)
 │      FiLM heads averaged over a consensus token + top training-annotator tokens at test time
 │      TTA: E-120 = 2048 six-view rot/flip + 768-px windows + 1024 four-view (0.45/0.35/0.20);
 │           E-121a = 2048 exact D4 eight-view + 512-px windows (0.65/0.35)
 ├─ 3. Continuous 5-channel topological spine-flow field                             sfsc/topoflow_targets.py
 │      Body B ∈ [0,1], Medial Spine S ∈ [0,1], Instance Separator E ∈ [0,1], Geodesic Flow v = -∇d_G(x, S_i) ∈ [-1,1]^2
 ├─ 4. Dual-scale field fusion + damped Euler streamline basin tracking              scripts/e121_dual_ensemble.py
 │      separator-gated seed cores; blobs with >=2 seeds split by 22 damped Euler steps x_{k+1} = x_k + Δs v(x_k)
 └─ 5. Exact conflict-graph MWIS / PQ-MBR solver                                     sfsc/topoflow_targets.py
        expected-PQ utility u_i = Σ_r w_r φ(max_g IoU(C_i, g)), φ(m) = m − τ/2 if m > 0.5 else −τ/2 (τ = 0.47)
        conflict graph = any pixel overlap; exact branch-and-bound per component → pixel-disjoint RLE submission
```

Every submission is verified for strict pixel disjointness, unique instance IDs, and valid COCO RLE (`3_src/sfsc/submission.py`). Inference uses strictly the single test image at a time (no temporal neighbors, no magnetograms, no external test data).

## Reproducing the Submissions

1. **Environment** (Python 3.12, CUDA GPU; tested on NVIDIA A100 40 GB and Apple M4 CPU):
   ```bash
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r 7_reproduce/requirements.txt
   ```
2. **Data**: download the official Kaggle competition archive into `1_data/raw/MAGFiLO_1.0_Kaggle_2026/{train,test}/`:
   ```bash
   kaggle competitions download -c filament-segmentation-2026 -p 1_data/raw
   unzip 1_data/raw/filament-segmentation-2026.zip -d 1_data/raw
   ```
3. **Pretrained Weights**: the 10 checkpoints (`4.03 GB`) are published on GitHub Release `weights-v1` and verified by exact byte size and MD5 against `7_reproduce/weights_manifest.json`:
   ```bash
   python 7_reproduce/download_weights.py --out 4_models --config 7_reproduce/final_e121_dual.json
   ```
4. **Run the final Pure Dual-Scale `TopoFlowNet` submission**:
   ```bash
   # Full 180-image GPU inference + Euler decoding + Exact MWIS:
   python 7_reproduce/run_e121_dual_final.py \
       --test-images 1_data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images \
       --weights 4_models --work runs/e121_final

   # Fast smoke test (3 images, Fold-0 Dual-Scale pair; runs in ~35s even on CPU):
   python 7_reproduce/run_e121_dual_final.py \
       --test-images 1_data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images \
       --weights 4_models --work runs/e121_smoke --dry-run

   # Byte-exact CPU verification from our stored topological maps (not included in this repository; SHA256 = 28ebcf822354fb49...):
   python 7_reproduce/run_e121_dual_final.py \
       --test-images 1_data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images \
       --e120-maps 5_outputs/exp120_topoflow --e121-maps 5_outputs/exp121_snake512 \
       --geometry 1_data/processed/disk_geometry_v1.csv --work runs/e121_verify --verify
   ```
5. **Self-Contained End-to-End Notebook**:
   - `2_notebooks/SFSC_06_topoflow_dual_pipeline.ipynb` walks through the complete Pure Dual-Scale `TopoFlowNet` system end to end: environment check, automated weight verification, single-image 5-channel solar physics visualization, `TRAIN_FROM_SCRATCH` toggle vs. pretrained loading, topological spine-flow inference + exact MWIS decoding, 6-panel qualitative overlay, and official PQ + Rubric evaluation (`SFSC_NB_DRY_RUN=1` runs in `25 s` on 1× A100 or `~32 s` on CPU; `SFSC_NB_DRY_RUN=0` runs all 10 models on all 180 test images, ~115 min on 1× A100).

**Determinism & Clean-Machine Verified Runtime**:
- From stored topological field maps (`--e120-maps` / `--e121-maps`), the CPU fusion + Euler + MWIS solver regenerates `sub_e121_pure_dual_mbr_10m_5f.csv` (`1,157` disjoint instances across `178/180` test images, `SHA256 = 28ebcf822354fb490f4a9412ad4492147deedac1367c44e4193bdc8c8ebd8a03`) **byte for byte** in `144.1 s` (`0.80 s/image` on 8 Apple M4 cores; `132.1 s` / `0.73 s/image` on clean Linux x86_64).
- Clean-machine end-to-end verification from raw `.jpeg` images and MD5-verified checkpoints (`Linux x86_64`, clean `Python 3.12.16` venv, `torch 2.11.0+cu128`, `1× NVIDIA A100-SXM4-40GB`) completes all 180 test images across all 10 models in **`6,920.0 s` (`115.3 min` total)**: `28.9 s` (`0.16 s/image`) for single-image solar physics & disk fit, `580.6–592.1 s` (`3.24 s/image`) per Global-1024 `E-120` fold, `764.9–772.7 s` (`4.26 s/image`) per Native-512 Snake `E-121a` fold, and `132.1 s` (`0.73 s/image`) for CPU Euler + exact MWIS decoding, matching our Kaggle submission (`58267599`) at **`PQ = 0.9975` (`SQ = 0.9979`, `RQ = 0.9996`, `1,156 / 1,157` 1-to-1 matched instances across identical `178/180` populated images, `FP = 0`, `1:n = 0`, `n:1 = 0`)**.

## Reproducing Training from Scratch

Rebuild the deterministic processed metadata CSVs (`folds_cvA_v1.csv`, `gt_train_v1.csv`, `disk_geometry_v1.csv`) from the Kaggle download:

```bash
python 7_reproduce/make_processed.py --data 1_data/raw/MAGFiLO_1.0_Kaggle_2026 --out 1_data/processed
```

Train all 10 Pure Dual-Scale `TopoFlowNet` checkpoints (`folds 0..4` for both scales; ~11 A100 GPU-hours total):

```bash
# 1. Global-1024 TopoFlowNet (E-120, folds 0..4; ~24 min/fold on 1x A100 40GB):
for F in 0 1 2 3 4; do
  python 3_src/scripts/train_e120_topoflow.py \
      --outer-fold $F --epochs 24 \
      --data-root 1_data/raw/MAGFiLO_1.0_Kaggle_2026 \
      --assets 1_data/processed \
      --cache-dir runs/cache_e120 \
      --out 5_outputs/exp120_topoflow/fold${F}/run
done

# 2. Native-512 Snake-TopoFlowNet (E-121a, folds 0..4; ~1.8 h/fold on 1x A100 40GB):
for F in 0 1 2 3 4; do
  python 3_src/scripts/train_e121_snake512.py \
      --outer-fold $F --epochs 26 \
      --data-root 1_data/raw/MAGFiLO_1.0_Kaggle_2026 \
      --assets 1_data/processed \
      --cache-dir runs/cache_e121 \
      --out 5_outputs/exp121_snake512/fold${F}/run
done

# 3. Strict nested 5-fold OOF evaluation + 180-image test submission + common-TP barb audit:
python 3_src/scripts/e121_dual_ensemble.py \
    --e120-root 5_outputs/exp120_topoflow \
    --e121-root 5_outputs/exp121_snake512 \
    --assets 1_data/processed \
    --test-images 1_data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images \
    --out 5_outputs/exp121_snake512/final_eval \
    --build-test --run-barb-audit
```

## Repository Layout

| Path | Contents |
|---|---|
| `7_reproduce/` | `run_e121_dual_final.py` & `final_e121_dual.json` (the final 10-model Pure Dual-Scale `TopoFlowNet` submission), `download_weights.py` + `weights_manifest.json` (10 checkpoints, size + MD5), `make_processed.py`, `requirements.txt` |
| `3_src/sfsc/` | Solar-limb geometry, single-image 5-ch physics frontend (`topoflow_physics.py`), `TopoFlowNet` + GroupNorm Dynamic Snake + Annotator FiLM (`topoflow_net.py`), continuous spine-flow targets + Euler decoder + exact MWIS solver (`topoflow_targets.py`), official-equivalent PQ scorer (`metrics.py`), RLE & submission validation |
| `3_src/scripts/` | Training (`train_e120_topoflow.py`, `train_e121_snake512.py`), dual-scale fusion + nested OOF evaluation (`e121_dual_ensemble.py`, shared helpers in `e120_ensemble_mbr.py`), official rubric & paired barb / Multiscale-IoU diagnostics (`rubric_metrics.py`, `barbs_diag.py`, `miou_metric.py`) and the candidate utilities they import (`tune_postprocess.py`, `fraglink.py`, `postops.py`) |
| `3_src/tests/` | Exact equivalence between `sfsc.metrics` and the organizers' self-evaluation notebook; TopoFlowNet target/decoder unit tests |
| `2_notebooks/` | Self-contained demo notebook `SFSC_06_topoflow_dual_pipeline.ipynb`, its builder and spec |

## License and Acknowledgements

Code: MIT License (`LICENSE`). Third-party libraries: `timm` (Apache-2.0), PyTorch (BSD-3-Clause). Data: MAGFiLO (Kaggle 2026 competition split; no external data or MAGFiLO Dataverse archive used in deployed models) — please cite Ahmadzadeh et al., *Scientific Data* 11, 1031 (2024), https://doi.org/10.1038/s41597-024-03876-y.
