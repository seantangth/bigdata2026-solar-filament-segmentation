"""Rebuild the small training/OOF assets from the Kaggle download (the final inference pipeline does not need them).

Writes, into a new or empty --out directory, the same four CSVs the training notebooks and OOF scripts read from
`1_data/processed/` (same code paths as the original generators `3_src/scripts/make_folds.py`, `make_assets_tar.py`,
`check_geometry.py`, which write to fixed paths):
  * folds_cvA_v1.csv   CV-A: 5 folds by file (all annotator sets of a file share a fold), stratified by (#annotators, station), seed 2026
  * folds_cvB_v1.csv   CV-B: 5 folds by time cluster (gap > 2 days), diagnostics only
  * gt_train_v1.csv    official-format GT (filament_id = '<annotator>-<stem>_<k>', compressed RLE counts), one row per annotated filament
  * disk_geometry_v1.csv  solar-disk circle fit (cx, cy, r, inliers, rms) for every train and test image

Usage: python 7_reproduce/make_processed.py --data 1_data/raw/MAGFiLO_1.0_Kaggle_2026 --out 1_data/processed [--workers 8]
"""
from __future__ import annotations

import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '3_src'))
from sfsc.data import build_gt_df, file_table, load_train  # noqa: E402
from sfsc.folds import make_cvA, make_cvB  # noqa: E402
from sfsc.geometry import fit_disk  # noqa: E402

JSON = 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
SPLITS = [('train', 'train/train_images'), ('test', 'test/test_images')]


def _fit(job):
    split, p = job
    f = fit_disk(np.array(Image.open(p)))
    return dict(split=split, stem=Path(p).stem, cx=f.cx, cy=f.cy, r=f.r, n_inl=f.n_inliers, rms=f.rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True, help='Kaggle dataset root (contains train/ and test/)')
    ap.add_argument('--out', type=Path, required=True, help='new or empty output directory')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(); t0 = time.time()
    assert not a.out.exists() or not any(a.out.iterdir()), f'{a.out} must be new or empty'
    a.out.mkdir(parents=True, exist_ok=True)
    ts = load_train(a.data / JSON)
    ft = file_table(ts)
    cva, cvb = make_cvA(ft), make_cvB(ft)
    cva[['file_name', 'stem', 'station', 'n_annot', 'n_fil', 'fold']].sort_values('file_name').to_csv(a.out / 'folds_cvA_v1.csv', index=False)
    cvb[['file_name', 'stem', 'station', 'n_annot', 'n_fil', 'cluster', 'fold']].to_csv(a.out / 'folds_cvB_v1.csv', index=False)
    gt = build_gt_df(ts); gt.to_csv(a.out / 'gt_train_v1.csv', index=False)
    print(f'folds: CV-A files/fold {cva.fold.value_counts().sort_index().tolist()} | GT rows {len(gt)}', flush=True)
    jobs = [(split, str(p)) for split, sub in SPLITS for p in sorted((a.data / sub).glob('*.jpeg'))]
    with Pool(a.workers) as pool:
        rows = pool.map(_fit, jobs, chunksize=4)          # map keeps the sorted (split, stem) order of the original
    pd.DataFrame(rows).to_csv(a.out / 'disk_geometry_v1.csv', index=False)
    print(f'geometry: {len(rows)} images ({sum(r["split"] == "train" for r in rows)} train / {sum(r["split"] == "test" for r in rows)} test)'
          f' | {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
