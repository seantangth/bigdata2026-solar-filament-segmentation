"""Layer 3 & Layer 5: Topological Spine-Flow Target Generation, Equivariant Transforms,
Euler Spine-Attractor Decoding, and PQ-Optimal MWIS/MBR Solver (E-120).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from . import H, W
from .rle import encode_mask


@dataclass
class AnnotatedInstance:
    ann_id: str
    category_id: int
    area: float
    bbox: tuple[float, float, float, float]
    rle_counts: bytes
    spine_xy: np.ndarray  # shape (N, 2) float32 in (x, y) pixel coords


@dataclass
class AnnotationSet:
    set_id: str           # e.g. '040301-20140609195854Bh'
    annotator_prefix: str # e.g. '040301'
    annotator_idx: int    # 1..37 (0 is reserved for Consensus Token)
    stem: str             # e.g. '20140609195854Bh'
    file_name: str
    instances: list[AnnotatedInstance]


def load_magfilo_annotations(json_path: str | Path) -> tuple[list[AnnotationSet], dict[str, int]]:
    """Load official MAGFiLO Kaggle train JSON including polygons and human-drawn spine polylines."""
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    prefixes = sorted({im["id"].split("-", 1)[0] for im in raw["images"]})
    prefix_to_idx = {p: idx + 1 for idx, p in enumerate(prefixes)}  # 1..N; 0 = consensus

    img_by_id: dict[str, dict] = {}
    anns_by_img: dict[str, list[dict]] = {}
    for im in raw["images"]:
        img_by_id[im["id"]] = im
        anns_by_img[im["id"]] = []
    for ann in raw["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    sets: list[AnnotationSet] = []
    for set_id, im in img_by_id.items():
        prefix, stem = set_id.split("-", 1)
        h, w = int(im.get("height", H)), int(im.get("width", W))
        inst_list: list[AnnotatedInstance] = []
        for ann in anns_by_img[set_id]:
            rles = mask_util.frPyObjects(ann["segmentation"], h, w)
            rle = mask_util.merge(rles)
            sp = np.asarray(ann["spine"], dtype=np.float32).reshape(-1, 2)
            bx, by, bw, bh = [float(v) for v in ann["bbox"]]
            inst_list.append(
                AnnotatedInstance(
                    ann_id=str(ann["id"]),
                    category_id=int(ann.get("category_id", 3)),
                    area=float(ann.get("area", 0.0)),
                    bbox=(bx, by, bw, bh),
                    rle_counts=rle["counts"] if isinstance(rle["counts"], bytes) else rle["counts"].encode("ascii"),
                    spine_xy=sp,
                )
            )
        sets.append(
            AnnotationSet(
                set_id=set_id,
                annotator_prefix=prefix,
                annotator_idx=prefix_to_idx[prefix],
                stem=stem,
                file_name=im["file_name"],
                instances=inst_list,
            )
        )
    return sets, prefix_to_idx


def decode_instance_mask(inst: AnnotatedInstance, h: int = H, w: int = W) -> np.ndarray:
    return mask_util.decode({"size": [h, w], "counts": inst.rle_counts}).astype(bool)


def _rasterize_spine_polyline(spine_xy: np.ndarray, h: int, w: int, thickness: int = 1) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    if spine_xy.shape[0] == 0:
        return canvas.astype(bool)
    pts = np.round(spine_xy).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    if len(pts) == 1:
        canvas[pts[0, 1], pts[0, 0]] = 1
    else:
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=thickness, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def _compute_instance_geodesic_flow(
    m_box: np.ndarray,
    s_box: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact 8-connected internal geodesic unit vector field (vx, vy) inside m_box toward s_box,
    plus thin-structure EDT inside m_box.
    """
    bh, bw = m_box.shape
    vx = np.zeros((bh, bw), dtype=np.float32)
    vy = np.zeros((bh, bw), dtype=np.float32)
    edt = ndimage.distance_transform_edt(m_box).astype(np.float32)

    fg_ys, fg_xs = np.nonzero(m_box)
    n_fg = fg_ys.size
    if n_fg == 0:
        return vx, vy, edt

    seeds_mask = m_box & s_box
    if not np.any(seeds_mask):
        max_d = float(edt.max())
        seeds_mask = m_box & (edt >= max(max_d * 0.75, max_d - 1.0))

    idx_map = np.full((bh, bw), -1, dtype=np.int32)
    idx_map[fg_ys, fg_xs] = np.arange(n_fg, dtype=np.int32)

    rows_l, cols_l, wts_l = [], [], []
    for dy, dx, dist in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142135), (1, -1, 1.4142135)):
        y0s = slice(0, bh - dy) if dy > 0 else slice(0, bh)
        y1s = slice(dy, bh) if dy > 0 else slice(0, bh)
        if dx >= 0:
            x0s, x1s = slice(0, bw - dx), slice(dx, bw)
        else:
            x0s, x1s = slice(-dx, bw), slice(0, bw + dx)

        u = idx_map[y0s, x0s]
        v = idx_map[y1s, x1s]
        valid = (u >= 0) & (v >= 0)
        if np.any(valid):
            uu = u[valid]
            vv = v[valid]
            rows_l.append(uu)
            cols_l.append(vv)
            wts_l.append(np.full(uu.size, dist, dtype=np.float32))

    if rows_l:
        r_idx = np.concatenate(rows_l)
        c_idx = np.concatenate(cols_l)
        w_val = np.concatenate(wts_l)
        graph = coo_matrix((w_val, (r_idx, c_idx)), shape=(n_fg, n_fg)).tocsr()
        seed_nodes = idx_map[seeds_mask]
        d_fg = dijkstra(graph, directed=False, indices=seed_nodes, min_only=True)
        inf_mask = ~np.isfinite(d_fg)
        if np.any(inf_mask):
            edt_seed = ndimage.distance_transform_edt(~seeds_mask)
            d_fg[inf_mask] = edt_seed[fg_ys[inf_mask], fg_xs[inf_mask]]
    else:
        d_fg = np.zeros(n_fg, dtype=np.float32)

    d_grid = np.zeros((bh + 2, bw + 2), dtype=np.float32)
    m_pad = np.zeros((bh + 2, bw + 2), dtype=bool)
    m_pad[1:-1, 1:-1] = m_box
    s_pad = np.zeros((bh + 2, bw + 2), dtype=bool)
    s_pad[1:-1, 1:-1] = seeds_mask
    d_grid[fg_ys + 1, fg_xs + 1] = d_fg.astype(np.float32)

    py = fg_ys + 1
    px = fg_xs + 1
    non_seed = ~s_pad[py, px]
    py_ns, px_ns = py[non_seed], px[non_seed]
    for _ in range(3):
        c_val = d_grid[py_ns, px_ns]
        up = np.where(m_pad[py_ns - 1, px_ns], d_grid[py_ns - 1, px_ns], c_val + 1.0)
        dn = np.where(m_pad[py_ns + 1, px_ns], d_grid[py_ns + 1, px_ns], c_val + 1.0)
        lf = np.where(m_pad[py_ns, px_ns - 1], d_grid[py_ns, px_ns - 1], c_val + 1.0)
        rt = np.where(m_pad[py_ns, px_ns + 1], d_grid[py_ns, px_ns + 1], c_val + 1.0)
        d_grid[py_ns, px_ns] = 0.5 * c_val + 0.125 * (up + dn + lf + rt)

    c_all = d_grid[py, px]
    d_up = np.where(m_pad[py - 1, px], d_grid[py - 1, px], c_all + 1.25)
    d_dn = np.where(m_pad[py + 1, px], d_grid[py + 1, px], c_all + 1.25)
    d_lf = np.where(m_pad[py, px - 1], d_grid[py, px - 1], c_all + 1.25)
    d_rt = np.where(m_pad[py, px + 1], d_grid[py, px + 1], c_all + 1.25)

    gx = 0.5 * (d_lf - d_rt)
    gy = 0.5 * (d_up - d_dn)
    norm = np.hypot(gx, gy)
    active = norm > 1e-4
    vx[fg_ys[active], fg_xs[active]] = gx[active] / np.maximum(norm[active], 0.25)
    vy[fg_ys[active], fg_xs[active]] = gy[active] / np.maximum(norm[active], 0.25)
    return np.clip(vx, -1.0, 1.0), np.clip(vy, -1.0, 1.0), edt


def build_topoflow_targets(
    instances: list[AnnotatedInstance],
    h: int = H,
    w: int = W,
    spine_sigma: float = 2.0,
    sep_radius: int = 14,
) -> dict[str, np.ndarray]:
    """Fast O(sum_bbox_area) topological target builder (~25-40 ms per 2048x2048 set)."""
    body = np.zeros((h, w), dtype=np.float32)
    spine_map = np.zeros((h, w), dtype=np.float32)
    flow = np.zeros((2, h, w), dtype=np.float32)
    thin_boost_map = np.zeros((h, w), dtype=np.float32)

    d1 = np.full((h, w), float(sep_radius + 4), dtype=np.float32)
    d2 = np.full((h, w), float(sep_radius + 4), dtype=np.float32)

    pad = sep_radius + 4
    if instances:
        # Batch decode all RLEs in a single C call!
        rle_list = [{"size": [h, w], "counts": inst.rle_counts} for inst in instances]
        masks_3d = mask_util.decode(rle_list)  # (H, W, K) uint8 in C!

        for k_idx, inst in enumerate(instances):
            bx, by, bw_f, bh_f = inst.bbox
            sp = inst.spine_xy
            if sp.shape[0] > 0:
                sx0, sy0 = int(np.floor(sp[:, 0].min())), int(np.floor(sp[:, 1].min()))
                sx1, sy1 = int(np.ceil(sp[:, 0].max())) + 1, int(np.ceil(sp[:, 1].max())) + 1
            else:
                sx0, sy0, sx1, sy1 = int(bx), int(by), int(bx + bw_f) + 1, int(by + bh_f) + 1

            x0 = max(0, min(int(math.floor(bx)), sx0) - pad)
            y0 = max(0, min(int(math.floor(by)), sy0) - pad)
            x1 = min(w, max(int(math.ceil(bx + bw_f)) + 1, sx1) + pad)
            y1 = min(h, max(int(math.ceil(by + bh_f)) + 1, sy1) + pad)
            if y1 <= y0 or x1 <= x0:
                continue

            m_box = masks_3d[y0:y1, x0:x1, k_idx].astype(bool)
            if not np.any(m_box):
                continue

            body[y0:y1, x0:x1] = np.maximum(body[y0:y1, x0:x1], m_box.astype(np.float32))

            sp_local = sp.copy()
            sp_local[:, 0] -= x0
            sp_local[:, 1] -= y0
            s_bin = _rasterize_spine_polyline(sp_local, y1 - y0, x1 - x0, thickness=1)

            if np.any(s_bin):
                dist_sp = ndimage.distance_transform_edt(~s_bin).astype(np.float32)
                sp_heat = np.exp(-0.5 * (dist_sp / spine_sigma) ** 2).astype(np.float32)
                sp_heat[dist_sp > 3.5 * spine_sigma] = 0.0
                spine_map[y0:y1, x0:x1] = np.maximum(spine_map[y0:y1, x0:x1], sp_heat)

            vx_box, vy_box, edt_box = _compute_instance_geodesic_flow(m_box, s_bin)
            flow[0, y0:y1, x0:x1] = np.where(m_box, vx_box, flow[0, y0:y1, x0:x1])
            flow[1, y0:y1, x0:x1] = np.where(m_box, vy_box, flow[1, y0:y1, x0:x1])

            tb_box = np.where(m_box, 2.0 * np.exp(-0.5 * (np.maximum(edt_box - 1.0, 0.0) / 2.2) ** 2), 0.0)
            thin_boost_map[y0:y1, x0:x1] = np.maximum(thin_boost_map[y0:y1, x0:x1], tb_box)

            dist_inst = ndimage.distance_transform_edt(~m_box).astype(np.float32)
            cur_d1 = d1[y0:y1, x0:x1]
            cur_d2 = d2[y0:y1, x0:x1]
            better1 = dist_inst < cur_d1
            d2[y0:y1, x0:x1] = np.where(better1, cur_d1, np.minimum(cur_d2, dist_inst))
            d1[y0:y1, x0:x1] = np.where(better1, dist_inst, cur_d1)

    near_pair = (d1 + d2 < float(sep_radius)) & (np.abs(d1 - d2) <= 3.5)
    sep = np.zeros((h, w), dtype=np.float32)
    if np.any(near_pair):
        diff = d1[near_pair] - d2[near_pair]
        gap = np.maximum(d1[near_pair] + d2[near_pair] - 2.0, 0.0)
        sep[near_pair] = np.exp(-0.5 * (diff / 1.6) ** 2) * np.exp(-0.5 * (gap / 5.5) ** 2)
    touching = (d1 <= 1.0) & (d2 <= 2.5)
    sep[touching] = 1.0
    spine_map = spine_map * (1.0 - 0.85 * sep)
    fg_weight = (1.0 + thin_boost_map + 2.5 * sep).astype(np.float32)

    return {
        "body": body,
        "spine": spine_map.astype(np.float32),
        "sep": sep.astype(np.float32),
        "flow": flow,
        "fg_weight": fg_weight,
    }


def rotate_flip_tensor_and_flow(
    arr_chw: np.ndarray,
    flow_2hw: np.ndarray | None,
    angle_deg: float,
    hflip: bool = False,
    vflip: bool = False,
    center: tuple[float, float] | None = None,
    border_values: list[float] | None = None,
    interp: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray | None]:
    c, h, w = arr_chw.shape
    cx, cy = ((w - 1) * 0.5, (h - 1) * 0.5) if center is None else center
    if border_values is None:
        border_values = [0.0] * c

    out = np.empty_like(arr_chw)
    if abs(angle_deg) > 1e-5:
        mat = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), 1.0)
        for idx in range(c):
            out[idx] = cv2.warpAffine(
                arr_chw[idx],
                mat,
                (w, h),
                flags=interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(border_values[idx]),
            )
    else:
        out[:] = arr_chw

    if hflip:
        out = out[:, :, ::-1]
    if vflip:
        out = out[:, ::-1, :]
    out = np.ascontiguousarray(out)

    if flow_2hw is None:
        return out, None

    f_warped = np.empty_like(flow_2hw)
    if abs(angle_deg) > 1e-5:
        mat = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), 1.0)
        for idx in range(2):
            f_warped[idx] = cv2.warpAffine(
                flow_2hw[idx],
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        rad = math.radians(float(angle_deg))
        ca, sa = math.cos(rad), math.sin(rad)
        vx_new = ca * f_warped[0] + sa * f_warped[1]
        vy_new = -sa * f_warped[0] + ca * f_warped[1]
        f_warped[0], f_warped[1] = vx_new, vy_new
    else:
        f_warped[:] = flow_2hw

    if hflip:
        f_warped = f_warped[:, :, ::-1]
        f_warped[0] = -f_warped[0]
    if vflip:
        f_warped = f_warped[:, ::-1, :]
        f_warped[1] = -f_warped[1]

    return out, np.ascontiguousarray(f_warped)


def invert_rotate_flip_predictions(
    probs_chw: np.ndarray,
    flow_2hw: np.ndarray,
    angle_deg: float,
    hflip: bool = False,
    vflip: bool = False,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    p = probs_chw
    f = flow_2hw.copy()
    if vflip:
        p = p[:, ::-1, :]
        f = f[:, ::-1, :]
        f[1] = -f[1]
    if hflip:
        p = p[:, :, ::-1]
        f = f[:, :, ::-1]
        f[0] = -f[0]
    p = np.ascontiguousarray(p)
    f = np.ascontiguousarray(f)
    if abs(angle_deg) > 1e-5:
        p, f = rotate_flip_tensor_and_flow(p, f, angle_deg=-float(angle_deg), hflip=False, vflip=False, center=center)
    return p, f


@dataclass
class CandidateInstance:
    mask: np.ndarray | None   # (H, W) bool, or None in lightweight RLE-only IPC mode
    rle_counts: str           # ASCII RLE counts
    area: int
    body_mean: float
    body_q85: float
    spine_max: float
    sep_border_mean: float
    score: float
    source_tag: str = "topoflow"


def _encode_bbox_mask(sub_mask: np.ndarray, y0: int, x0: int, h: int, w: int) -> tuple[np.ndarray, str]:
    full_m = np.zeros((h, w), dtype=bool)
    full_m[y0 : y0 + sub_mask.shape[0], x0 : x0 + sub_mask.shape[1]] = sub_mask
    return full_m, encode_mask(full_m)


def decode_topoflow(
    body_prob: np.ndarray,
    spine_prob: np.ndarray,
    sep_prob: np.ndarray,
    flow_xy: np.ndarray,
    valid_disk: np.ndarray | None = None,
    t_body: float = 0.40,
    t_seed: float = 0.50,
    alpha_sep: float = 0.78,
    spine_bridge_thr: float = 0.58,
    n_steps: int = 22,
    step_size: float = 1.25,
    min_area: int = 180,
    score_gate: float = 0.0,
    max_candidates: int = 90,
    source_tag: str = "topoflow",
    keep_masks: bool = True,
) -> list[CandidateInstance]:
    """Fast, exact-boundary Topological Spine-Flow Decoder (~10 ms/image):
      1. Exact foreground F = (body_prob >= t_body) (zero artificial dilation of filament widths!).
      2. Spine-bridging connects separated islands into a shared seed ID ONLY across genuine
         spine corridors (spine_prob >= spine_bridge_thr & sep_prob < 0.28) without fattening F.
      3. Isolated single-seed blobs keep 100% of their crisp boundary pixels in O(1).
      4. Multi-seed merged blobs (N:1 touching filaments!) are partitioned cleanly inside their
         local bounding box via 22-step damped Euler (vx, vy) flow tracking + nearest-seed capture,
         preserving 100% of foreground pixels without boundary erosion!
    """
    h, w = body_prob.shape
    fg = (body_prob >= t_body) & valid_disk if valid_disk is not None else (body_prob >= t_body)
    if not np.any(fg):
        return []

    struct8 = np.ones((3, 3), dtype=int)

    # 1. Construct repulsive-cut seed attractors strictly inside fg
    seed_energy = np.maximum(
        spine_prob * (1.0 - 0.85 * sep_prob),
        body_prob * (1.0 - alpha_sep * sep_prob),
    )
    seed_core = fg & (seed_energy >= t_seed) & (sep_prob < 0.55)
    if not np.any(seed_core):
        return []

    # Allow strong human-like spine corridor to link disconnected seed_core islands of the SAME filament
    # across optically faint pinches (where sep_prob < 0.25), WITHOUT fattening fg outside the bridge!
    if spine_bridge_thr < 1.0:
        spine_corridor = (spine_prob >= spine_bridge_thr) & (body_prob >= max(0.18, t_body * 0.45)) & (sep_prob < 0.25)
        if valid_disk is not None:
            spine_corridor &= valid_disk
    else:
        spine_corridor = np.zeros_like(fg)

    seed_union = seed_core | spine_corridor
    seed_lbl, n_raw_seeds = ndimage.label(seed_union, structure=struct8)
    if n_raw_seeds == 0:
        return []

    # Retain only seed IDs that contain real seed_core pixels (filtering orphan corridor specks)
    valid_seed_ids = np.unique(seed_lbl[seed_core])
    valid_seed_ids = valid_seed_ids[valid_seed_ids > 0]
    if valid_seed_ids.size == 0:
        return []
    remap = np.zeros(n_raw_seeds + 1, dtype=np.int32)
    remap[valid_seed_ids] = np.arange(1, valid_seed_ids.size + 1, dtype=np.int32)
    # Seed labels strictly on seed_core pixels (+ thin bridge pixels that actually join >=1 valid seed)
    seed_on_fg = np.where(seed_core, remap[seed_lbl], 0)

    #Also include thin 1-px spine bridge pixels into fg ONLY if they belong to a valid seed_lbl
    if spine_bridge_thr < 1.0 and np.any(spine_corridor):
        bridged_corridor = spine_corridor & (remap[seed_lbl] > 0)
        fg_work = fg | bridged_corridor
        seed_work = np.where(seed_core | bridged_corridor, remap[seed_lbl], 0)
    else:
        fg_work = fg
        seed_work = seed_on_fg

    # 2. Label connected components of fg_work; resolve 1-seed blobs in O(1) and split >=2-seed blobs via Euler flow!
    cc_lbl, n_cc = ndimage.label(fg_work, structure=struct8)
    cc_slices = ndimage.find_objects(cc_lbl)

    inst_masks_by_seed: dict[int, list[tuple[slice, slice, np.ndarray]]] = {}

    for cid, sl in enumerate(cc_slices, start=1):
        if sl is None:
            continue
        sub_cc = cc_lbl[sl] == cid
        if int(sub_cc.sum()) < 8:
            continue
        sub_seeds = seed_work[sl][sub_cc]
        uniq_s = np.unique(sub_seeds)
        uniq_s = uniq_s[uniq_s > 0]

        if uniq_s.size == 0:
            # Failed hysteresis t_seed gate -> drop FP component!
            continue
        elif uniq_s.size == 1:
            # Fast exact path: single seed attractor owns 100% of this component's crisp mask
            sid = int(uniq_s[0])
            # Exclude artificial corridor pixels that weren't in original fg unless bridging two islands
            clean_sub = sub_cc & fg[sl]
            if np.any(clean_sub):
                inst_masks_by_seed.setdefault(sid, []).append((sl[0], sl[1], clean_sub))
        else:
            # N:1 MERGED BLOB DETECTED (>= 2 distinct seed attractors inside one connected blob)!
            # Run 22-step damped Euler Spine-Flow inside this local padded bounding box!
            y0 = max(0, sl[0].start - 4)
            y1 = min(h, sl[0].stop + 4)
            x0 = max(0, sl[1].start - 4)
            x1 = min(w, sl[1].stop + 4)
            bh, bw = y1 - y0, x1 - x0

            loc_cc = (cc_lbl[y0:y1, x0:x1] == cid) & fg[y0:y1, x0:x1]
            ys_l, xs_l = np.nonzero(loc_cc)
            if ys_l.size == 0:
                continue

            loc_seed = np.where(cc_lbl[y0:y1, x0:x1] == cid, seed_work[y0:y1, x0:x1], 0)
            loc_sp = spine_prob[y0:y1, x0:x1]
            loc_sep = sep_prob[y0:y1, x0:x1]
            loc_vx = flow_xy[0, y0:y1, x0:x1]
            loc_vy = flow_xy[1, y0:y1, x0:x1]

            px = xs_l.astype(np.float32)
            py = ys_l.astype(np.float32)
            for _ in range(n_steps):
                ix = np.clip(np.round(px).astype(np.int32), 0, bw - 1)
                iy = np.clip(np.round(py).astype(np.int32), 0, bh - 1)
                # Anchor once inside a seed core
                in_seed = loc_seed[iy, ix] > 0
                damp = np.where(in_seed, 0.15, 1.0) * step_size
                nx = np.clip(px + loc_vx[iy, ix] * damp, 0.0, float(bw - 1))
                ny = np.clip(py + loc_vy[iy, ix] * damp, 0.0, float(bh - 1))
                nix = np.round(nx).astype(np.int32)
                niy = np.round(ny).astype(np.int32)
                # Block stepping off foreground or up a repulsive sep_prob wall
                bad = (~loc_cc[niy, nix]) | ((loc_sep[niy, nix] > 0.45) & (loc_sep[niy, nix] > loc_sep[iy, ix] + 0.05))
                px = np.where(bad, px, nx)
                py = np.where(bad, py, ny)

            end_x = np.clip(np.round(px).astype(np.int32), 0, bw - 1)
            end_y = np.clip(np.round(py).astype(np.int32), 0, bh - 1)

            # Nearest seed attractor in local box (guaranteed every pixel in loc_cc gets one of uniq_s!)
            _, near_idx = ndimage.distance_transform_edt(loc_seed == 0, return_indices=True)
            assigned_sids = loc_seed[near_idx[0, end_y, end_x], near_idx[1, end_y, end_x]]

            part_map = np.zeros((bh, bw), dtype=np.int32)
            part_map[ys_l, xs_l] = assigned_sids
            part_slices = ndimage.find_objects(part_map)
            for sid in uniq_s:
                s_int = int(sid)
                if s_int <= len(part_slices) and part_slices[s_int - 1] is not None:
                    ps = part_slices[s_int - 1]
                    m_part = part_map[ps] == s_int
                    if np.any(m_part):
                        inst_masks_by_seed.setdefault(s_int, []).append(
                            (slice(y0 + ps[0].start, y0 + ps[0].stop), slice(x0 + ps[1].start, x0 + ps[1].stop), m_part)
                        )

    # 3. Assemble per-seed instances (supporting multi-island bridged filaments!) and score them before RLE encoding
    raw_scored: list[tuple[float, int, int, int, np.ndarray, float, float, float, float]] = []
    for sid, pieces in inst_masks_by_seed.items():
        y0 = min(p[0].start for p in pieces)
        y1 = max(p[0].stop for p in pieces)
        x0 = min(p[1].start for p in pieces)
        x1 = max(p[1].stop for p in pieces)
        sub_union = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for sy, sx, sm in pieces:
            sub_union[sy.start - y0 : sy.stop - y0, sx.start - x0 : sx.stop - x0] |= sm

        area = int(sub_union.sum())
        if area < min_area:
            continue

        b_vals = body_prob[y0:y1, x0:x1][sub_union]
        s_vals = spine_prob[y0:y1, x0:x1][sub_union]
        sep_vals = sep_prob[y0:y1, x0:x1][sub_union]
        b_mean = float(b_vals.mean())
        b_q85 = float(np.percentile(b_vals, 85.0))
        s_max = float(s_vals.max())
        sep_mean = float(sep_vals.mean())
        score = float(0.55 * b_q85 + 0.30 * b_mean + 0.15 * s_max)
        if score < score_gate:
            continue
        raw_scored.append((score, y0, x0, area, sub_union, b_mean, b_q85, s_max, sep_mean))

    raw_scored.sort(key=lambda item: item[0], reverse=True)
    if len(raw_scored) > max_candidates:
        raw_scored = raw_scored[:max_candidates]

    occupied = np.zeros((h, w), dtype=bool)
    candidates: list[CandidateInstance] = []
    for score, y0, x0, area, sub_union, b_mean, b_q85, s_max, sep_mean in raw_scored:
        sub_clean = sub_union & (~occupied[y0 : y0 + sub_union.shape[0], x0 : x0 + sub_union.shape[1]])
        clean_area = int(sub_clean.sum())
        if clean_area < min_area:
            continue
        occupied[y0 : y0 + sub_union.shape[0], x0 : x0 + sub_union.shape[1]] |= sub_clean
        full_m, rle_c = _encode_bbox_mask(sub_clean, y0, x0, h, w)
        candidates.append(
            CandidateInstance(
                mask=full_m if keep_masks else None,
                rle_counts=rle_c,
                area=clean_area,
                body_mean=b_mean,
                body_q85=b_q85,
                spine_max=s_max,
                sep_border_mean=sep_mean,
                score=score,
                source_tag=source_tag,
            )
        )

    return candidates


def _rle_obj(counts: str) -> dict:
    return {"size": [H, W], "counts": counts.encode("ascii")}


def _solve_mwis_component(weights: np.ndarray, adj_masks: list[int]) -> list[int]:
    """Exact Branch-and-Bound Maximum-Weight Independent Set on a conflict connected component."""
    n = len(weights)
    if n == 0:
        return []
    if n == 1:
        return [0] if weights[0] > 0 else []

    order = np.argsort(-weights)
    w_ord = weights[order]
    adj_ord = [0] * n
    for idx, orig in enumerate(order):
        m = 0
        nbr_bits = adj_masks[int(orig)]
        for j_idx, j_orig in enumerate(order):
            if (nbr_bits >> int(j_orig)) & 1:
                m |= 1 << j_idx
        adj_ord[idx] = m

    if n > 26:
        chosen_mask = 0
        for i in range(n):
            if w_ord[i] <= 0:
                break
            if (chosen_mask & adj_ord[i]) == 0:
                chosen_mask |= 1 << i
        return [int(order[i]) for i in range(n) if (chosen_mask >> i) & 1]

    suffix_pos = np.zeros(n + 1, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        suffix_pos[i] = suffix_pos[i + 1] + max(0.0, float(w_ord[i]))

    best_val = 0.0
    best_mask = 0

    def dfs(idx: int, forbidden: int, cur_val: float, cur_mask: int) -> None:
        nonlocal best_val, best_mask
        if cur_val > best_val:
            best_val = cur_val
            best_mask = cur_mask
        if idx >= n or cur_val + suffix_pos[idx] <= best_val + 1e-9:
            return
        if w_ord[idx] <= 0.0:
            return
        bit = 1 << idx
        if (forbidden & bit) == 0:
            dfs(idx + 1, forbidden | adj_ord[idx], cur_val + float(w_ord[idx]), cur_mask | bit)
        dfs(idx + 1, forbidden, cur_val, cur_mask)

    dfs(0, 0, 0.0, 0)
    return [int(order[i]) for i in range(n) if (best_mask >> i) & 1]


def pq_mbr_solve(
    candidate_pool: list[CandidateInstance],
    reference_parses: list[tuple[list[str], float]],
    tau_pq: float = 0.47,
    fp_cost_scale: float = 1.0,
    dedup_iou: float = 0.93,
) -> list[CandidateInstance]:
    """Select a 100% pixel-disjoint subset of candidate instances maximizing expected Pooled PQ."""
    if not candidate_pool or not reference_parses:
        return []

    sorted_pool = sorted(candidate_pool, key=lambda c: c.score, reverse=True)
    rles_all = [_rle_obj(c.rle_counts) for c in sorted_pool]
    iou_self = np.asarray(mask_util.iou(rles_all, rles_all, [0] * len(rles_all)), dtype=np.float32).reshape(len(rles_all), len(rles_all))
    keep_indices: list[int] = []
    for i in range(len(sorted_pool)):
        if any(iou_self[i, j] >= dedup_iou for j in keep_indices):
            continue
        keep_indices.append(i)

    cands = [sorted_pool[i] for i in keep_indices]
    cand_rles = [rles_all[i] for i in keep_indices]
    n_c = len(cands)

    total_w = sum(w for _, w in reference_parses)
    if total_w <= 0:
        return []

    utilities = np.zeros(n_c, dtype=np.float64)
    fp_penalty = 0.5 * tau_pq * fp_cost_scale

    for ref_counts, ref_w in reference_parses:
        nw = ref_w / total_w
        if not ref_counts:
            utilities -= nw * fp_penalty
            continue
        ref_rles = [_rle_obj(rc) for rc in ref_counts]
        iou_mat = np.asarray(mask_util.iou(cand_rles, ref_rles, [0] * len(ref_rles)), dtype=np.float64).reshape(n_c, len(ref_rles))
        max_iou = iou_mat.max(axis=1)
        hit = max_iou > 0.5
        gain = np.where(hit, max_iou - 0.5 * tau_pq, -fp_penalty)
        utilities += nw * gain

    pos_idx = [i for i in range(n_c) if utilities[i] > 0.0]
    if not pos_idx:
        return []

    cands = [cands[i] for i in pos_idx]
    w_vec = utilities[pos_idx]
    m_num = len(cands)

    cand_rles_pos = [_rle_obj(c.rle_counts) for c in cands]
    iou_pos = np.asarray(mask_util.iou(cand_rles_pos, cand_rles_pos, [0] * m_num), dtype=np.float64).reshape(m_num, m_num)
    adj_masks = [0] * m_num
    adj_lists: list[list[int]] = [[] for _ in range(m_num)]
    for i in range(m_num):
        for j in range(i + 1, m_num):
            if iou_pos[i, j] > 0.0:
                adj_masks[i] |= 1 << j
                adj_masks[j] |= 1 << i
                adj_lists[i].append(j)
                adj_lists[j].append(i)

    visited = [False] * m_num
    selected: list[CandidateInstance] = []
    for i in range(m_num):
        if visited[i]:
            continue
        comp: list[int] = []
        stack = [i]
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj_lists[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        if len(comp) == 1:
            selected.append(cands[comp[0]])
        else:
            local_map = {g_idx: l_idx for l_idx, g_idx in enumerate(comp)}
            sub_w = w_vec[comp]
            sub_adj = [0] * len(comp)
            for l_i, g_i in enumerate(comp):
                bits = 0
                for g_j in adj_lists[g_i]:
                    bits |= 1 << local_map[g_j]
                sub_adj[l_i] = bits
            chosen_local = _solve_mwis_component(sub_w, sub_adj)
            for l_idx in chosen_local:
                selected.append(cands[comp[l_idx]])

    if any(c.mask is None for c in selected):
        # _solve_mwis_component already enforces exact IoU == 0.0 (zero pixel overlap) between all selected RLEs!
        return sorted(selected, key=lambda x: x.score, reverse=True)

    occupied = np.zeros((H, W), dtype=bool)
    clean_selected: list[CandidateInstance] = []
    for c in sorted(selected, key=lambda x: x.score, reverse=True):
        m = c.mask & (~occupied)
        area = int(m.sum())
        if area < 100:
            continue
        occupied |= m
        if area != c.area:
            c = CandidateInstance(
                mask=m,
                rle_counts=encode_mask(m),
                area=area,
                body_mean=c.body_mean,
                body_q85=c.body_q85,
                spine_max=c.spine_max,
                sep_border_mean=c.sep_border_mean,
                score=c.score,
                source_tag=c.source_tag,
            )
        clean_selected.append(c)

    return clean_selected
