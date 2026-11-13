"""CV split builders. CV-A: GroupKFold by file (all annotator sets of a file share a fold), stratified by
(#annotators, station). CV-B: GroupKFold by time cluster (observations closer than `gap_days` share a cluster)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_cvA(ft: pd.DataFrame, n_folds: int = 5, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    ft = ft.copy()
    ft["fold"] = -1
    nxt = 0
    for _, grp in ft.groupby(["n_annot", "station"], sort=True):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        for i in idx:                       # deal round-robin, continuing the counter across strata
            ft.loc[i, "fold"] = nxt % n_folds
            nxt += 1
    return ft


def make_cvB(ft: pd.DataFrame, n_folds: int = 5, gap_days: float = 2.0, seed: int = 2026) -> pd.DataFrame:
    ft = ft.sort_values("time").copy()
    t = ft["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
    cluster = np.zeros(len(ft), dtype=int)
    for i in range(1, len(ft)):
        cluster[i] = cluster[i - 1] + (1 if (t[i] - t[i - 1]) > gap_days * 86400 else 0)
    ft["cluster"] = cluster
    sizes = ft.groupby("cluster").size().sort_values(ascending=False)
    rng = np.random.RandomState(seed)
    load = np.zeros(n_folds)
    cl_fold = {}
    for c, sz in sizes.items():            # greedy balance: largest clusters first, ties broken randomly
        f = int(np.argmin(load + rng.rand(n_folds) * 1e-3))
        cl_fold[c] = f
        load[f] += sz
    ft["fold"] = ft["cluster"].map(cl_fold)
    return ft.sort_values("file_name").reset_index(drop=True)
