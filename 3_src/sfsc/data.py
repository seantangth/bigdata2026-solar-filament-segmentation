"""Official train JSON → per-annotator instance sets, soft labels, and GT dataframe in the official scorer format."""
from __future__ import annotations

import datetime as dt
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from . import H, W

STEM_RE = re.compile(r"^(\d{14})([A-Z])h$")
STATIONS = {"B": "Big Bear", "C": "Cerro Tololo", "L": "Learmonth", "M": "Mauna Loa", "T": "El Teide", "U": "Udaipur"}


def stem_of(file_name: str) -> str:
    return Path(file_name).stem


def parse_stem(stem: str) -> tuple[dt.datetime, str]:
    """'20140609195854Bh' → (datetime(2014,6,9,19,58,54), 'B')."""
    m = STEM_RE.match(stem)
    if m is None:
        raise ValueError(f"unexpected stem {stem!r}")
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), m.group(2)


def ann_rle(ann: dict) -> dict:
    """Polygon → RLE exactly as the organisers rasterise it (pycocotools frPyObjects + merge)."""
    return mask_util.merge(mask_util.frPyObjects(ann["segmentation"], H, W))


def ann_counts(ann: dict) -> str:
    return ann_rle(ann)["counts"].decode("ascii")


def ann_mask(ann: dict) -> np.ndarray:
    return mask_util.decode(ann_rle(ann)).astype(bool)


@dataclass
class TrainSet:
    images: list
    annotations: list
    by_image: dict = field(default_factory=dict)    # annotator-image id → [annotation]
    by_file: dict = field(default_factory=dict)     # file_name → [annotator-image id]
    image_file: dict = field(default_factory=dict)  # annotator-image id → file_name

    def files(self) -> list[str]:
        return sorted(self.by_file)

    @staticmethod
    def annotator_of(image_id: str) -> str:
        return image_id.split("-", 1)[0]

    def n_annotators(self, file_name: str) -> int:
        return len(self.by_file[file_name])


def load_train(json_path) -> TrainSet:
    d = json.load(open(json_path))
    ts = TrainSet(d["images"], d["annotations"])
    by_image, by_file = defaultdict(list), defaultdict(list)
    for a in d["annotations"]:
        by_image[a["image_id"]].append(a)
    for im in d["images"]:
        by_file[im["file_name"]].append(im["id"])
        ts.image_file[im["id"]] = im["file_name"]
    ts.by_image, ts.by_file = dict(by_image), dict(by_file)
    return ts


def build_gt_df(ts: TrainSet, files=None) -> pd.DataFrame:
    """GT in the official format: filament_id='<annotator>-<stem>_<k>' (k from 1), segmentation_rle=compressed counts."""
    files = ts.files() if files is None else list(files)
    rows = []
    for f in files:
        for iid in ts.by_file[f]:
            for k, a in enumerate(ts.by_image.get(iid, []), 1):
                rows.append((f"{iid}_{k}", ann_counts(a)))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def instance_label_map(ts: TrainSet, image_id: str) -> np.ndarray:
    """int32 (H, W): 0 = background, k = k-th instance of this annotator-image (later instances overwrite earlier)."""
    lab = np.zeros((H, W), dtype=np.int32)
    for k, a in enumerate(ts.by_image.get(image_id, []), 1):
        lab[ann_mask(a)] = k
    return lab


def soft_label(ts: TrainSet, file_name: str) -> np.ndarray:
    """float32 (H, W): fraction of this file's annotator sets that mark the pixel (1 annotator → {0,1})."""
    ids = ts.by_file[file_name]
    acc = np.zeros((H, W), dtype=np.float32)
    for iid in ids:
        acc += (instance_label_map(ts, iid) > 0)
    return acc / len(ids)


def file_table(ts: TrainSet) -> pd.DataFrame:
    rows = []
    for f in ts.files():
        t, st = parse_stem(stem_of(f))
        ids = ts.by_file[f]
        rows.append(dict(file_name=f, stem=stem_of(f), time=t, station=st, n_annot=len(ids),
                         n_fil=sum(len(ts.by_image.get(i, [])) for i in ids)))
    return pd.DataFrame(rows)


def list_test_stems(test_dir) -> list[str]:
    return sorted(stem_of(p.name) for p in Path(test_dir).glob("*.jpeg"))
