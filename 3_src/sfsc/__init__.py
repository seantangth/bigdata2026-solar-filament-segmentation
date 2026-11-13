"""SFSC — Solar Filament Segmentation Challenge 2026 toolkit (Kaggle filament-segmentation-2026).

Modules
-------
rle         : COCO RLE encode/decode + pixel-disjointness assertion (Kaggle rejects overlapping predictions)
data        : official train JSON loader, per-annotator instance sets, GT dataframe in the official format
metrics     : Panoptic Quality scorer, verified line-by-line equivalent to the host's self-evaluation notebook (E-009)
folds       : CV-A (GroupKFold by file) and CV-B (time-cluster GroupKFold) split builders
geometry    : robust solar-disk circle fit → valid-region mask
classical   : local-contrast threshold baseline (format validation / scale anchor, E-011)
submission  : prediction masks → validated submission CSV
"""
H = 2048
W = 2048
