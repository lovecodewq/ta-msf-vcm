import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import math
import matplotlib.pyplot as plt


def find_summary_files(inputs: List[str], pattern: str) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        path = Path(p)
        if path.is_dir():
            files.extend(path.rglob(pattern))
        elif path.is_file():
            files.append(path)
    # de-duplicate
    return sorted(list(dict.fromkeys(files)))


def load_point(fp: Path) -> Dict[str, Any]:
    with open(fp, 'r') as f:
        data = json.load(f)
    run_dir = fp.parent.parent  # .../summary/bpp_vs_map50.json → run root
    data['__source_file'] = str(fp)
    data['__run_dir'] = str(run_dir)
    return data


def find_vtm_image_anchor_dirs(inputs: List[str]) -> List[Path]:
    """Find directories that look like VTM image anchor outputs.

    A valid directory contains both 'metadata.json' and 'preds.json'.
    We search recursively under any input directories; if an input is itself a
    directory that matches, include it.
    """
    dirs: List[Path] = []
    seen: Dict[str, bool] = {}
    for p in inputs:
        path = Path(p)
        candidates: List[Path] = []
        if path.is_dir():
            # Try itself
            if (path / 'metadata.json').exists() and (path / 'preds.json').exists():
                candidates.append(path)
            # Recurse: look for any 'metadata.json' and check sibling 'preds.json'
            for meta in path.rglob('metadata.json'):
                if (meta.parent / 'preds.json').exists():
                    candidates.append(meta.parent)
        elif path.is_file():
            # If a file is provided, check its parent
            par = path.parent
            if (par / 'metadata.json').exists() and (par / 'preds.json').exists():
                candidates.append(par)
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen[key] = True
                dirs.append(c)
    return sorted(dirs)


def load_vtm_point(run_dir: Path) -> Dict[str, Any]:
    """Load avg_bpp from metadata.json and map50 from preds.json."""
    meta_fp = run_dir / 'metadata.json'
    preds_fp = run_dir / 'preds.json'
    with open(meta_fp, 'r') as f:
        meta = json.load(f)
    with open(preds_fp, 'r') as f:
        preds = json.load(f)
    # metadata.json structure: { 'summary': { 'avg_bpp': ... }, 'per_image': [...] }
    avg_bpp = float(meta.get('summary', {}).get('avg_bpp'))
    # preds.json structure: { 'metrics': { 'map50': ... }, ... }
    map50 = float(preds.get('metrics', {}).get('map50'))
    return {
        '__source_file': str(meta_fp),
        '__run_dir': str(run_dir),
        'image_compression': {
            'avg_bpp': avg_bpp,
            'map50': map50,
            'model_type': 'vtm_image_anchor',
            'checkpoint': str(run_dir),
        }
    }


def linear_fit(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if len(points) < 2:
        return (float('nan'), float('nan'))
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def load_raw_point(files: List[Path]) -> float:
    for fp in files:
        d = load_point(fp)
        if 'raw' in d and isinstance(d['raw'], dict) and 'map50' in d['raw']:
            return float(d['raw']['map50'])
    return float('nan')

def load_image_compression_point(files: List[Path]) -> Dict[str, Any]:
    image_by_model: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        d = load_point(fp)
        img = d.get('image_compression', {})
        # note: avg_bpp and map50 might be NaN or null
        if 'avg_bpp' in img and 'map50' in img:
            try:
                xbpp = float(img['avg_bpp'])
                ymap = float(img['map50'])
                if not (math.isfinite(xbpp) and math.isfinite(ymap)):
                    pass
            except Exception:
                pass
            mt = img.get('model_type', '')
            if not mt:
                continue
            # only save one point per checkpoint per model type
            image_by_model.setdefault(mt, {'points': [], 'by_ckpt': {}})
            ckpt = img.get('checkpoint') or f"{d.get('__run_dir')}/image"
            if ckpt not in image_by_model[mt]['by_ckpt']:
                pt = {
                    'avg_bpp': xbpp,
                    'map50': ymap,
                    'lambda': img.get('lambda'),
                    'checkpoint': ckpt,
                    'run_dir': d.get('__run_dir'),
                    'source': d.get('__source_file'),
                }
                image_by_model[mt]['by_ckpt'][ckpt] = pt
                image_by_model[mt]['points'].append(pt)
    return image_by_model

def load_feature_compression_point(files: List[Path]) -> Dict[str, Dict[str, Any]]:
    feature_by_model: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        d = load_point(fp)
        feat = d.get('feature_compression', {})
        if not feat:
            continue
        if 'avg_bpp' in feat and 'map50' in feat:
            try:
                xbpp = float(feat['avg_bpp'])
                ymap = float(feat['map50'])
                if not (math.isfinite(xbpp) and math.isfinite(ymap)):
                    continue
            except Exception:
                continue
            mt = feat.get('model_type', '')
            if not mt:
                continue
            feature_by_model.setdefault(mt, {'points': []})
            if feat.get('detection_loss_weight') is not None:
                feature_by_model[mt]['points'].append({
                    'avg_bpp': xbpp,
                    'map50': ymap,
                    'lambda': feat.get('lambda'),
                    'detection_loss_weight': feat.get('detection_loss_weight'),
                    'checkpoint': feat.get('checkpoint'),
                    'run_dir': d.get('__run_dir'),
                    'source': d.get('__source_file'),
                })
            else:
                feature_by_model[mt]['points'].append({
                    'avg_bpp': xbpp,
                    'map50': ymap,
                    'lambda': feat.get('lambda'),
                    'checkpoint': feat.get('checkpoint'),
                    'run_dir': d.get('__run_dir'),
                    'source': d.get('__source_file'),
                })
    return feature_by_model

def load_lmsfc_feature_anchor_point(files: List[Path]) -> Dict[str, Dict[str, Any]]:
    feature_by_model: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        d = load_point(fp)
        l = d.get('lmsfc_feature_compression', {})
        if not l:
            continue
        try:
            xbpp = float(l.get('avg_bpp'))
            ymap = float(l.get('map50'))
        except Exception:
            continue
        mt = l.get('model_type', 'lmsfc_anchor')
        feature_by_model.setdefault(mt, {'points': []})
        feature_by_model[mt]['points'].append({
            'avg_bpp': xbpp,
            'map50': ymap,
            'checkpoint': l.get('checkpoint'),
            'run_dir': d.get('__run_dir'),
            'source': d.get('__source_file'),
        })
    return feature_by_model

def load_vtm_image_compression_point(vtm_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    image_by_model: Dict[str, Dict[str, Any]] = {}
    for ddir in vtm_dirs:
        try:
            d = load_vtm_point(ddir)
        except Exception:
            continue
        img = d.get('image_compression', {})
        if 'avg_bpp' in img and 'map50' in img:
            try:
                xbpp = float(img['avg_bpp'])
                ymap = float(img['map50'])
                if not (math.isfinite(xbpp) and math.isfinite(ymap)):
                    continue
            except Exception:
                continue
            mt = img.get('model_type', 'vtm_image_anchor')
            image_by_model.setdefault(mt, {'points': [], 'by_ckpt': {}})
            ckpt = img.get('checkpoint') or f"{d.get('__run_dir')}/vtm"
            if ckpt not in image_by_model[mt]['by_ckpt']:
                pt = {
                    'avg_bpp': xbpp,
                    'map50': ymap,
                    'run_dir': d.get('__run_dir'),
                    'source': d.get('__source_file'),
                }
                image_by_model[mt]['by_ckpt'][ckpt] = pt
                image_by_model[mt]['points'].append(pt)
    return image_by_model

def merge_points(files: List[Path], vtm_dirs: List[Path]) -> Dict[str, Any]:
    # Single raw baseline (take the first found; they should be identical across runs)
    raw_map_value: float = load_raw_point(files)
    image_by_model: Dict[str, Dict[str, Any]] = load_image_compression_point(files)
    feature_by_model: Dict[str, Dict[str, Any]] = load_feature_compression_point(files)

    # VTM image anchor points
    image_by_model.update(load_vtm_image_compression_point(vtm_dirs))

    # Fit lines per model type
    for mt, bundle in image_by_model.items():
        pts = [(p['avg_bpp'], p['map50']) for p in bundle['points']]
        try:
            a, b = linear_fit(pts)
            image_by_model[mt]['fit'] = {'a': a, 'b': b}
        except Exception:
            logging.warning(f'Failed to fit linear model for {mt}')
            logging.warning(f'Points: {pts}')
            pass

    for mt, bundle in feature_by_model.items():
        pts_source = bundle['points']
        # Only build best_points if detection_loss_weight is present for this model type
        has_weight = any(p.get('detection_loss_weight') is not None for p in pts_source)
        if has_weight:
            # For each lambda, keep the point with the highest mAP (tie-breaker: lowest bpp)
            by_lambda: Dict[Any, List[Dict[str, Any]]] = {}
            for p in pts_source:
                lam = p.get('lambda')
                by_lambda.setdefault(lam, []).append(p)
            # Expose grouping structure
            feature_by_model[mt]['by_lambda'] = by_lambda
            # Best per lambda
            best_points: List[Dict[str, Any]] = []
            for lam, plist in by_lambda.items():
                best = sorted(plist, key=lambda q: (-q['map50'], q['avg_bpp']))[0]
                best_points.append(best)
            feature_by_model[mt]['best_points'] = best_points
            pts = [(p['avg_bpp'], p['map50']) for p in best_points]
        else:
            # No detection weight hyperparameter: use all points directly
            pts = [(p['avg_bpp'], p['map50']) for p in pts_source]
        a, b = linear_fit(pts)
        feature_by_model[mt]['fit'] = {'a': a, 'b': b}

    merged = {
        'raw': {
            'map50': raw_map_value,
        },
        'image': {
            'by_model': image_by_model,
        },
        'feature': {
            'by_model': feature_by_model,
        }
    }
    return merged


def plot_merged(merged: Dict[str, Any], out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Raw baseline
    map_raw = merged.get('raw', {}).get('map50')
    if map_raw is not None:
        ax.axhline(y=map_raw, color='gray', linestyle='--', linewidth=1.5, label=f'Raw baseline (mAP@0.5={map_raw:.3f})')
        new_lossness_map = 0.9 * map_raw
        ax.axhline(y=new_lossness_map, color='green', linestyle='--', linewidth=1.5, label=f'Near-lossless baseline (mAP@0.5={new_lossness_map:.3f})')
    # Consistent colors per (category, model_type)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    color_map: Dict[Tuple[str, str], str] = {}
    def get_color(category: str, model_type: str) -> str:
        key = (category, model_type)
        if key not in color_map:
            color_map[key] = color_cycle[len(color_map) % len(color_cycle)]
        return color_map[key]

    # # Plot image models
    # for mt, bundle in merged.get('image', {}).get('by_model', {}).items():
    #     pts = bundle.get('points', [])
    #     if not pts:
    #         continue
    #     # sort by bpp for polyline
    #     pts_sorted = sorted(pts, key=lambda p: p['avg_bpp'])
    #     x = [p['avg_bpp'] for p in pts_sorted]
    #     y = [p['map50'] for p in pts_sorted]
    #     color = get_color('image', mt)
    #     if mt == 'vtm_image_anchor':
    #         ax.scatter(x, y, label=f'Image: {mt}', marker='*', color=color)
    #     else:
    #         ax.scatter(x, y, label=f'Image: {mt}', marker='o', color=color)
    #     if len(pts_sorted) >= 2:
    #         ax.plot(x, y, linestyle='--', alpha=0.8, color=color)

    # Plot feature models
    for mt, bundle in merged.get('feature', {}).get('by_model', {}).items():
        # Only apply best-points filtering for models trained with detection loss
        if mt == 'fused_feature_with_detection_loss' and bundle.get('best_points'):
            pts = bundle['best_points']
        else:
            pts = bundle.get('points', [])
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p['avg_bpp'])
        x = [p['avg_bpp'] for p in pts_sorted]
        y = [p['map50'] for p in pts_sorted]
        color = get_color('feature', mt)
        # TA-MSF and L-MSFC are the only two feature models that we are interested in
        if mt == 'fused_feature_with_detection_loss':
            label = f'Feature: TA-MSF'
            ax.scatter(x, y, label=label, marker='s', color=color)
        # elif mt == 'lmsfc_anchor':
        #     label = f'Feature: L-MSFC'
        #     ax.scatter(x, y, label=label, marker='s', color=color)
        # elif mt == 'vtm_feature_anchor':
        #     label = f'Feature: VTM-Anchor'
        #     ax.scatter(x, y, label=label, marker='*', color=color)
        elif mt == 'fused_feature':
            label = f'Feature: TA-MSF w/o det loss'
            ax.scatter(x, y, label=label, marker='s', color=color)
        else:
            continue
        if len(pts_sorted) >= 2:
            ax.plot(x, y, linestyle='--', alpha=0.8, color=color)

    ax.set_xlabel('Bits Per Pixel (avg)')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('BPP vs mAP@0.5')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Merge pipeline RD summaries and optional VTM anchor results to plot BPP vs mAP@0.5')
    parser.add_argument('--inputs', nargs='*', default=None, help='Pipeline summary roots to search (e.g., evaluation_results/pipeline)')
    # Directories containing VTM image anchor outputs (each must have metadata.json and preds.json)
    parser.add_argument('--vtm_image_anchor_dir', nargs='*', default=None, help='Directories containing VTM image anchor outputs (each must have metadata.json and preds.json)')
    # Allow passing precomputed feature anchor summary files directly
    parser.add_argument('--vtm_feature_anchor_files', nargs='*', default=None, help='One or more feature-anchor summary files (summary/bpp_vs_map50.json)')
    parser.add_argument('--pattern', type=str, default='bpp_vs_map50.json', help='Filename pattern to search within pipeline run directories')
    parser.add_argument('--lmsfc_feature_anchor_dir', nargs='*', default=None, help='Directories containing L-MSFC anchor runs (we will search for summary/bpp_vs_map50.json)')
    parser.add_argument('--out_json', type=str, required=True, help='Output JSON file for merged anchors')
    parser.add_argument('--out_plot', type=str, required=True, help='Output PNG file for merged plot')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Collect pipeline summary files
    files: List[Path] = find_summary_files(args.inputs, args.pattern) if args.inputs else []

    # Include any explicit feature-anchor summary files
    if args.vtm_feature_anchor_files:
        for f in args.vtm_feature_anchor_files:
            p = Path(f)
            if p.exists() and p.is_file():
                files.append(p)
            else:
                logging.warning(f"Feature-anchor summary not found or not a file: {f}")
    # Include any explicit L-MSFC feature-anchor summary files
    lmsfc_files: List[Path] = []
    # From directories: find all matching summaries
    if args.lmsfc_feature_anchor_dir:
        lmsfc_dir_files = find_summary_files(args.lmsfc_feature_anchor_dir, args.pattern)
        lmsfc_files.extend(lmsfc_dir_files)
    # De-duplicate
    lmsfc_files = sorted(list(dict.fromkeys(lmsfc_files)))
    # De-duplicate files
    files = sorted(list(dict.fromkeys(files)))

    # Resolve image anchor dirs
    vtm_image_dirs: List[Path] = find_vtm_image_anchor_dirs(args.vtm_image_anchor_dir) if args.vtm_image_anchor_dir else []

    if not files and not vtm_image_dirs:
        raise SystemExit('No inputs found. Provide --inputs (pipeline summaries) and/or --vtm_image_anchor_dir (image anchors) and/or --vtm_feature_anchor_files (feature anchors).')
    logging.info(f'Found {len(files)} pipeline/feature summary files and {len(vtm_image_dirs)} VTM image anchor runs')

    merged = merge_points(files, vtm_image_dirs)
    # Merge in L-MSFC points under feature category
    if lmsfc_files:
        lmsfc_pts = load_lmsfc_feature_anchor_point(lmsfc_files)
        merged.setdefault('feature', {}).setdefault('by_model', {}).update(lmsfc_pts)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(merged, f, indent=2)
    logging.info(f'Wrote merged anchors to {out_json}')

    plot_merged(merged, Path(args.out_plot))
    logging.info(f'Wrote merged plot to {args.out_plot}')


if __name__ == '__main__':
    main()

