# Spec: SFSC_06_topoflow_dual_pipeline.ipynb

## 目標
公開 repo 的「100% Pure Dual-Scale TopoFlowNet 全流程自包含示範 Notebook」，100% 對齊主辦方 11 月更新要求（Kaggle Discussion #755907 / #753665）：
1. **自動下載權重**：預設 `SFSC_NB_AUTO_DOWNLOAD=1`，若 `4_models/` 缺少所需的 `TopoFlowNet` 權重，Cell 4 自動呼叫 `7_reproduce/download_weights.py` 從 GitHub Release 下載並驗證檔案大小與 MD5（評審零手動步驟）。
2. **單一 Notebook 涵蓋完整四階段**：
   - **Pre-processing**：單張 $2048\times2048$ H$\alpha$ 日面擬合與 5 通道太陽物理前端（CLV 去臨邊昏暗、雙尺度 DoG、Hessian 管狀脊線特徵、極座標 $r/R$）與視覺化。
   - **Training（可切換從頭訓練或載入預訓練權重）**：`SFSC_NB_TRAIN_FROM_SCRATCH=0`（預設載入預訓練權重）或 `1`（直接呼叫 `3_src/scripts/train_e121_snake512.py`／`train_e120_topoflow.py` 從原始標註訓練）。
   - **Inference & Topological Post-processing**：呼叫 `7_reproduce/run_e121_dual_final.py` 執行 10 模型純雙尺度 `TopoFlowNet`（5 顆 `E-120` 全域連續旋轉等變 + 5 顆 `E-121a` 原生 512 動態蛇形卷積）推論、測地線歐拉流積分（Geodesic Euler Spine-Flow）與精確衝突圖最大權重獨立集（Exact Lagrangian MWIS / PQ-MBR）實例解碼，並繪製六宮格拓撲場與細微棘刺（Barbs）局部放大圖。
   - **Evaluation（支援任意標註集與 707 張五折 OOF 重現）**：使用官方對齊之 `sfsc.metrics.evaluate` 與 `rubric_metrics.py` 計算 `PQ / SQ / RQ` 與 Rubric 分佈指標（`mean IoU`、`mean Dice`、`IoU > 0.5` 比例、`1:n` 斷裂率、`n:1` 誤併率）。
3. **100% 單一模型家族一致性（#753665 Re.5）**：同一組 10 個五折權重同時重現技術報告中的 707 張 CV-A OOF 讀數與最終 Kaggle 提交檔，無額外 LightGBM `.pkl` 或舊版 U-Net 依賴。

## 平台與資源
- 平台：任一具備 CUDA GPU 之環境（Lambda A100 / Google Colab GPU / 本地工作站）；`DRY_RUN=1` 亦支援 CPU 快速煙霧驗證。
- 預估時間：
  - `DRY_RUN=1`（預設，3 張影像 × Fold-0 雙尺度對）：A100 GPU 約 20–30 秒；CPU 約 45–90 秒。
  - `DRY_RUN=0`（完整 180 張測試集 × 10 模型）：A100 GPU 約 22–28 分鐘。

## Cell 大綱
| Cell # | 用途 | 主要動作 |
|--------|------|---------|
| 0 (md) | 首屏執行合約 | Purpose / Architecture / Runtime / Estimated time / Switches / Prerequisites（英文） |
| 1 | Setup & Imports | 自動定位 repo 根目錄、載入 `sfsc` 物理前端與評估模組、設定計時器 |
| 2 | Config & Switches | 讀取 `7_reproduce/final_e121_dual.json`；支援 `SFSC_NB_DRY_RUN`、`SFSC_NB_AUTO_DOWNLOAD`、`SFSC_NB_TRAIN_FROM_SCRATCH`、`SFSC_NB_EVAL_GT_CSV`、`SFSC_NB_RUN_OOF_EVAL` |
| 3 | Environment Check | 比對 `7_reproduce/requirements.txt` 套件版本與 CUDA 硬體資訊 |
| 4 | Inputs & Auto-Download Weights | 檢查輸入影像；若缺少模型權重自動執行 `7_reproduce/download_weights.py` 並驗證 MD5 |
| 5 | Stage 1: Solar Physics Frontend | 單張影像執行 `extract_physics_channels` 並繪製 5 通道物理特徵圖 |
| 6 | Stage 2: Train-from-Scratch or Load Pretrained | 依 `TRAIN_FROM_SCRATCH` 切換執行訓練腳本或使用已驗證之預訓練權重 |
| 7 | Stage 3: Dual-Scale TopoFlowNet + MWIS Pipeline | 子程序呼叫 `7_reproduce/run_e121_dual_final.py` 產出無重疊 RLE 提交檔並串流進度 |
| 8 | Stage 4: Topological Field & Instance Visualization | 繪製單張測試影像之輸入、Body 機率、Medial Spine + Contact Boundary、Euler Flow 向量場與最終無重疊實例遮罩（含棘刺放大窗） |
| 9 | Stage 5: Official PQ & Rubric Evaluation | 對任意標註 CSV（或 707 張 OOF 預測）執行官方 `PQ` 與 Rubric 分佈評估 |
| 10 | Summary | 輸出執行摘要、推論秒數、互斥驗證狀態與輸出路徑 |
