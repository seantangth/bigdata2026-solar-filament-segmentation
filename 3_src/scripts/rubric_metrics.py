"""E-062: the rubric's quantitative distributions (official self-evaluation notebook) on the full CV-A OOF.

For every variant the OOF prediction masks are rebuilt through the deployed mask paths
(A-base: SHARED `pp_candidates` + gate/min_area, as `pp_submit.py`; D1: same candidate labels, kept cids with
cross-fitted yhat >= tau, as `apply_scorer.py`), encoded to RLE and scored with `sfsc.metrics.evaluate`
(official-equivalent, E-009). Reported per variant:
  * PQ/SQ/RQ (validation gate: must equal the r1 report),
  * IoU > 0 pair distribution (n, mean IoU, mean Dice, share > 0.5); Dice = 2·IoU/(1+IoU) per pair,
  * GT degree (#predictions with IoU > 0 per GT, per annotator row): 1:0 / 1:1 / 1:n,
  * prediction degree (#GT with IoU > 0 per prediction, per annotator row): 0:1 / 1:1 / n:1.
Optional human reference (--hh): annotator b's masks as predictions for annotator a on every multi-annotator image.
Figures replicate the notebook's `plot_distribution` / `plot_m2n_counts` layout.

Usage:
  python rubric_metrics.py --spec spec.json --gt gt_train_v1.csv --geometry disk_geometry_v1.csv --out OUT [--workers 8] [--hh]
spec.json = {"variants": {"A21": {"probs": DIR, "t_lo": .5, "t_hi": .7, "merge": 16, "method": "gate", "gate": .85, "min_area": 200},
                          "D21": {"probs": DIR, "t_lo": .5, "t_hi": .7, "merge": 16, "method": "d1", "candidates": CSV, "tau": .22}},
             "expected_pq": {"A21": 0.4565, ...}, "figures": ["A21", "D21"]}
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT / '3_src'))
from tune_postprocess import pp_candidates, pp_valid_mask  # noqa: E402
from sfsc.metrics import evaluate  # noqa: E402
from sfsc.rle import encode_mask  # noqa: E402
from fraglink import linked_candidates  # noqa: E402
from postops import apply_ops  # noqa: E402

FLOOR = 50   # instance_scorer candidate floor (D1 candidates); gate variants use their own min_area


def _work(job):
    stem, geo, variants, keep_d1, links = job
    labs, out = {}, {}
    for name, V in variants.items():
        key = (V['probs'], V['t_lo'], V['t_hi'], int(V['merge']), V.get('links'), V.get('link_thr'))
        if key not in labs:
            prob = np.load(Path(V['probs']) / f'{stem}.npz')['prob']; valid = pp_valid_mask(*geo)
            if V.get('links'):   # E-063 learned fragment links instead of the merge_px dilation
                labs[key] = linked_candidates(prob, valid, V['t_lo'], V['t_hi'], links[(V['links'], V['link_thr'])].get(stem, [])) + (prob, valid)
            else:
                labs[key] = pp_candidates(prob, valid, V['t_lo'], V['t_hi'], int(V['merge'])) + (prob, valid)
        lab, n, area, meanp, prob, valid = labs[key]
        if V['method'] == 'gate':
            ids = (np.flatnonzero((area >= V['min_area']) & (meanp >= V['gate'])) + 1).tolist()
        else:
            ids = [c for c in keep_d1.get(name, []) if area[c - 1] >= FLOOR]
        if V.get('post'):        # E-064 / E-065 post-operations on the final instance map
            final = np.zeros(lab.shape, np.int32)
            for k, c in enumerate(ids, 1): final[lab == c] = k
            img = None
            if any(op.get('c_min') is not None for op in V['post']):
                from PIL import Image
                img = np.asarray(Image.open(Path(V['images']) / f'{stem}.jpeg'))
            aux = np.load(Path(V['aux_probs']) / f'{stem}.npz')['prob'] if V.get('aux_probs') else None   # E-071 growth source
            final, _ = apply_ops(final, prob, valid, V['post'], img, aux)
            out[name] = [(f'{stem}_{k}', encode_mask(final == k)) for k in range(1, len(ids) + 1)]
        else:
            out[name] = [(f'{stem}_{k}', encode_mask(lab == c)) for k, c in enumerate(ids, 1)]
    return stem, out


def stats(res) -> dict:
    iou = res.ious_pos; dice = 2 * iou / (1 + iou)
    g, p = res.gt_degree, res.pred_degree
    frac = lambda d, lo, hi=None: float(((d >= lo) if hi is None else (d == lo)).mean()) if len(d) else 0.0
    return dict(PQ=round(res.pq, 5), SQ=round(res.sq, 4), RQ=round(res.rq, 4), TP=res.tp, FP=res.fp, FN=res.fn,
                pairs_iou_pos=int(len(iou)), mean_iou=round(float(iou.mean()), 4), mean_dice=round(float(dice.mean()), 4),
                share_iou_gt_0p5=round(float((iou > 0.5).mean()), 4), median_iou=round(float(np.median(iou)), 4),
                n_gt_rows=int(len(g)), gt_1to0=round(frac(g, 0, 0), 4), gt_1to1=round(frac(g, 1, 1), 4), gt_1toN=round(frac(g, 2), 4),
                n_pred_rows=int(len(p)), pred_0to1=round(frac(p, 0, 0), 4), pred_1to1=round(frac(p, 1, 1), 4), pred_Nto1=round(frac(p, 2), 4),
                gt_degree_counts={int(k): int(v) for k, v in sorted(Counter(g.tolist()).items())},
                pred_degree_counts={int(k): int(v) for k, v in sorted(Counter(p.tolist()).items())})


def figures(name, res, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    iou = res.ious_pos
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, (vals, title) in enumerate([(iou, 'IoU'), (2 * iou / (1 + iou), 'Dice')]):
        counts, edges = np.histogram(vals, bins=np.linspace(0, 1, 51))
        ax[k].bar((edges[:-1] + edges[1:]) / 2, counts / max(counts.sum(), 1), width=0.02, edgecolor='black', alpha=0.7, color='navy')
        ax[k].axvline(vals.mean(), color='crimson', ls='--', lw=1.5, label=f'Mean = {vals.mean():.3f}'); ax[k].legend(fontsize=8)
        ax[k].set_title(f'{name}: {title} Distribution'); ax[k].set_xlabel(title); ax[k].grid(axis='y', ls='--')
    gc, pc = Counter(res.gt_degree.tolist()), Counter(res.pred_degree.tolist()); off = 0.5
    for d, c in gc.items(): ax[2].bar(d + off, c, width=0.8, color='crimson' if d == 0 else 'forestgreen', edgecolor='black', alpha=0.7)
    for d, c in pc.items(): ax[2].bar(-(d + off), c, width=0.8, color='darkorange' if d == 0 else 'steelblue', edgecolor='black', alpha=0.7)
    pos = sorted([d + off for d in gc] + [-(d + off) for d in pc])
    ax[2].set_xticks(pos, [f'{int(round(-v - off))}:1' if v < 0 else f'1:{int(round(v - off))}' for v in pos], rotation=45, fontsize=8)
    ax[2].axvline(0, color='black', lw=0.8); ax[2].grid(axis='y', ls='--'); ax[2].set_title(f'{name}: Many to One and One to Many')
    ax[2].legend(handles=[Patch(facecolor='forestgreen', label=f'GT Hits (1:n, n>0): {sum(v for k, v in gc.items() if k)}'),
                          Patch(facecolor='crimson', label=f'GT Misses (1:0): {gc.get(0, 0)}'),
                          Patch(facecolor='darkorange', label=f'Prediction Misses (0:1): {pc.get(0, 0)}'),
                          Patch(facecolor='steelblue', label=f'Prediction Hits (n:1, n>0): {sum(v for k, v in pc.items() if k)}')], fontsize=8)
    fig.tight_layout(); fig.savefig(out / f'{name}.png', dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    for k in ('spec', 'gt', 'geometry', 'out'):
        ap.add_argument('--' + k, type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--hh', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); (a.out / 'figures').mkdir(exist_ok=True); t0 = time.time()
    spec = json.loads(a.spec.read_text()); variants = spec['variants']
    gt = pd.read_csv(a.gt)[['filament_id', 'segmentation_rle']]
    geo = pd.read_csv(a.geometry).set_index('stem')
    stems = sorted(p.stem for p in Path(next(iter(variants.values()))['probs']).glob('*.npz'))
    keep_d1 = {}
    for name, V in variants.items():
        if V['method'] == 'd1':
            C = pd.read_csv(V['candidates'], usecols=['stem', 'cid', 'yhat'])
            keep_d1[name] = {s: g.sort_values('cid').cid.astype(int).tolist() for s, g in C[C.yhat >= V['tau']].groupby('stem')}
    links = {}
    for V in variants.values():
        if V.get('links') and (V['links'], V['link_thr']) not in links:
            Lk = pd.read_csv(V['links'], usecols=['stem', 'a', 'b', 'p']); Lk = Lk[Lk.p >= V['link_thr']]
            links[(V['links'], V['link_thr'])] = {s: list(zip(g.a.astype(int), g.b.astype(int))) for s, g in Lk.groupby('stem')}
    jobs = [(s, tuple(float(geo.loc[s, k]) for k in ('cx', 'cy', 'r')), variants,
             {v: keep_d1[v].get(s, []) for v in keep_d1}, {k: {s: L.get(s, [])} for k, L in links.items()}) for s in stems]
    preds = {v: [] for v in variants}
    with Pool(a.workers) as pool:
        for i, (stem, out) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            for v, rows in out.items(): preds[v].extend(rows)
            if i % 100 == 0 or i == len(jobs): print(f'[{time.time() - t0:5.0f}s] masks {i}/{len(jobs)}', flush=True)
    rep = dict(spec=spec, n_images=len(stems), variants={}); per_img = {}
    for v in variants:
        pdf = pd.DataFrame(preds[v], columns=['filament_id', 'segmentation_rle'])
        res = evaluate(gt, pdf); rep['variants'][v] = stats(res)
        per_img[v] = res.rows.groupby('image')[['tp', 'fp', 'fn', 'sum_iou']].sum().reindex(stems).fillna(0).to_numpy(float)
        exp = spec.get('expected_pq', {}).get(v)
        # 1e-4: sfsc.metrics decodes every GT independently (official); the instance_scorer packs behind the r1 numbers paint GT into one
        # label map, so the 5 overlapping same-annotator GT pairs of the train set lose one GT there (E-062: +1 TP / -1 FN, +6e-5 PQ)
        rep['variants'][v]['pq_gate'] = None if exp is None else bool(abs(res.pq - exp) < 1e-4)
        print(v, {k: rep['variants'][v][k] for k in ('PQ', 'mean_iou', 'mean_dice', 'gt_1to0', 'gt_1toN', 'pred_0to1', 'pred_Nto1', 'pq_gate')}, flush=True)
        if v in spec.get('figures', []): figures(v, res, a.out / 'figures')
        if a.hh:   # same population as the human reference: multi-annotator images only
            gimg = gt.filament_id.str.split('_', n=1).str[0].str.split('-', n=1).str[1]
            multi = set(gimg[gt.groupby(gimg).filament_id.transform(lambda x: x.str.split('-', n=1).str[0].nunique()) > 1])
            pm = pdf[pdf.filament_id.str.split('_', n=1).str[0].isin(multi)]
            rep['variants'][v + '@multi'] = stats(evaluate(gt[gimg.isin(multi)], pm)); rep['variants'][v + '@multi']['n_images'] = len(multi)
    if a.hh:   # human reference: annotator b as prediction for annotator a (ordered pairs, multi-annotator images)
        key = gt.filament_id.str.split('_', n=1).str[0]; ann = key.str.split('-', n=1).str[0]; img = key.str.split('-', n=1).str[1]
        G, P = [], []
        for s, grp in gt.groupby(img):
            codes = sorted(ann[grp.index].unique())
            if len(codes) < 2: continue
            for b in codes:
                for c in codes:
                    if b == c: continue
                    u = f'{s}x{b}x{c}'
                    ga = grp[ann[grp.index] == c]; pb = grp[ann[grp.index] == b]
                    G += [(f'{c}-{u}_{k}', r) for k, r in enumerate(ga.segmentation_rle, 1)]
                    P += [(f'{u}_{k}', r) for k, r in enumerate(pb.segmentation_rle, 1)]
        res = evaluate(pd.DataFrame(G, columns=['filament_id', 'segmentation_rle']), pd.DataFrame(P, columns=['filament_id', 'segmentation_rle']))
        rep['variants']['HH'] = stats(res); rep['variants']['HH']['n_ordered_pairs'] = int(res.rows.shape[0])
        print('HH', {k: rep['variants']['HH'][k] for k in ('PQ', 'mean_iou', 'mean_dice', 'gt_1to0', 'gt_1toN', 'pred_0to1', 'pred_Nto1', 'n_ordered_pairs')}, flush=True)
        figures('HH', res, a.out / 'figures')
    if spec.get('reference') in per_img:   # paired image-level bootstrap of PQ against the reference variant
        rng = np.random.default_rng(2026); idx = [rng.integers(0, len(stems), len(stems)) for _ in range(2000)]
        f = lambda t, ix: (lambda q: q[3] / (q[0] + 0.5 * q[1] + 0.5 * q[2]))(t[ix].sum(0))
        ref = spec['reference']; rep['paired_vs_reference'] = {}
        for v in variants:
            if v == ref: continue
            d = np.array([f(per_img[v], ix) - f(per_img[ref], ix) for ix in idx])
            rep['paired_vs_reference'][v] = dict(delta=round(rep['variants'][v]['PQ'] - rep['variants'][ref]['PQ'], 5),
                                                 ci95=[round(float(np.quantile(d, .025)), 5), round(float(np.quantile(d, .975)), 5)])
        print(json.dumps(rep['paired_vs_reference'], indent=1), flush=True)
    rep['minutes'] = round((time.time() - t0) / 60, 1)
    (a.out / 'report.json').write_text(json.dumps(rep, indent=2))
    cols = ['PQ', 'SQ', 'RQ', 'pairs_iou_pos', 'mean_iou', 'mean_dice', 'share_iou_gt_0p5', 'gt_1to0', 'gt_1to1', 'gt_1toN', 'pred_0to1', 'pred_1to1', 'pred_Nto1']
    tab = pd.DataFrame({v: {c: r[c] for c in cols} for v, r in rep['variants'].items()}).T
    tab.to_csv(a.out / 'summary.csv'); print(tab.to_string(), flush=True)


if __name__ == '__main__':
    main()
