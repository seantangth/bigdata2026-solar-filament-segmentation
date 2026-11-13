"""Fetch the 10 Pure Dual-Scale TopoFlowNet checkpoints listed in 7_reproduce/weights_manifest.json
into OUT/<name>/best.pt and verify each checkpoint by exact byte size and MD5.

Sources
  default       the GitHub release named in the manifest (plain HTTPS; no account needed once the repository is public)
  --base-url U  any location serving the same asset names as U/<asset>
  --from-dir D  assets already downloaded, e.g. with the GitHub CLI:
                gh release download weights-v1 -R seantangth/bigdata2026-solar-filament-segmentation -D D

Usage: python 7_reproduce/download_weights.py --out 4_models [--base-url URL | --from-dir DIR] [--only name1,name2 | --config CFG.json]
  --config  only the checkpoints needed by a pipeline config (e.g. 7_reproduce/final_e121_dual.json fetches the 10 TopoFlowNet models)
"""
from __future__ import annotations

import argparse, hashlib, json, shutil, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 1 << 23


def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def ok(p: Path, c: dict) -> bool:
    return p.exists() and p.stat().st_size == c['size'] and md5sum(p) == c['md5']


def fetch(url: str, dst: Path, size: int, tries: int = 3):
    for t in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dst, 'wb') as f:
                done, t0 = 0, time.time()
                for b in iter(lambda: r.read(CHUNK), b''):
                    f.write(b); done += len(b)
                    print(f'\r  {done / 1e6:7.1f} / {size / 1e6:.1f} MB  {done / 1e6 / max(time.time() - t0, 1e-6):5.1f} MB/s', end='', flush=True)
            print()
            return
        except OSError as e:          # URLError / HTTPError / timeouts: retry, then give up
            print(f'\n  attempt {t}/{tries} failed: {e}', flush=True)
            time.sleep(5 * t)
    raise RuntimeError(f'could not download {url}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True, help='weights directory (run_e121_dual_final.py --weights)')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--base-url', help='URL prefix serving <asset> files (default: the GitHub release in the manifest)')
    src.add_argument('--from-dir', type=Path, help='directory holding the already downloaded <asset> files')
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument('--only', help='comma-separated subset of checkpoint names')
    sel.add_argument('--config', type=Path, help='a pipeline config (e.g. final_e121_dual.json): fetch only its models')
    ap.add_argument('--manifest', type=Path, default=HERE / 'weights_manifest.json')
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    want = set(a.only.split(',')) if a.only else None
    if a.config:
        cfg = json.loads(a.config.read_text())
        want = {m['name'] for m in cfg['models']} | ({cfg['aux_model']['name']} if cfg.get('aux_model') else set())
    todo = [c for c in man['checkpoints'] if want is None or c['name'] in want]
    if want and len(todo) != len(want):
        sys.exit(f'unknown checkpoint name(s): {sorted(want - {c["name"] for c in todo})}')
    base = (a.base_url or man['release']['base_url']).rstrip('/')
    bad = []
    for i, c in enumerate(todo, 1):
        dst = a.out / c['file']
        print(f'[{i}/{len(todo)}] {c["name"]} ({c["size"] / 1e6:.1f} MB)', flush=True)
        if ok(dst, c):
            print('  present, md5 ok'); continue
        dst.parent.mkdir(parents=True, exist_ok=True); part = dst.with_suffix('.pt.partial')
        if a.from_dir:
            shutil.copyfile(a.from_dir / c['asset'], part)
        else:
            fetch(f'{base}/{c["asset"]}', part, c['size'])
        if ok(part, c):
            part.replace(dst); print('  md5 ok')
        else:
            bad.append(c['name']); print(f'  FAILED: size/md5 differ from the manifest (kept as {part})')
    print(f'{len(todo) - len(bad)}/{len(todo)} checkpoints verified in {a.out}' + (f'; failed: {bad}' if bad else ''))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
