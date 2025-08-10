import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
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


def linear_fit(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if len(points) < 2:
        return (float('nan'), float('nan'))
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def merge_points(files: List[Path]) -> Dict[str, Any]:
    # Single raw baseline (take the first found; they should be identical across runs)
    raw_map_value: float = None  # type: ignore
    image_by_model: Dict[str, Dict[str, Any]] = {}
    feature_by_model: Dict[str, Dict[str, Any]] = {}

    for fp in files:
        d = load_point(fp)
        # raw
        raw = d.get('raw', {})
        if raw_map_value is None and isinstance(raw, dict) and 'map50' in raw:
            try:
                raw_map_value = float(raw['map50'])
            except Exception:
                pass

        # image
        img = d.get('image_compression', {})
        if 'avg_bpp' in img and 'map50' in img:
            mt = img.get('model_type', 'image_model')
            image_by_model.setdefault(mt, {'points': [], 'by_ckpt': {}})
            ckpt = img.get('checkpoint') or f"{d.get('__run_dir')}/image"
            if ckpt not in image_by_model[mt]['by_ckpt']:
                pt = {
                    'avg_bpp': float(img['avg_bpp']),
                    'map50': float(img['map50']),
                    'lambda': img.get('lambda'),
                    'checkpoint': ckpt,
                    'run_dir': d.get('__run_dir'),
                    'source': d.get('__source_file'),
                }
                image_by_model[mt]['by_ckpt'][ckpt] = pt
                image_by_model[mt]['points'].append(pt)

        # feature
        feat = d.get('feature_compression', {})
        if 'avg_bpp' in feat and 'map50' in feat:
            mt = feat.get('model_type', 'feature_model')
            feature_by_model.setdefault(mt, {'points': []})
            if feat.get('detection_loss_weight') is not None:
                feature_by_model[mt]['points'].append({
                    'avg_bpp': float(feat['avg_bpp']),
                    'map50': float(feat['map50']),
                    'lambda': feat.get('lambda'),
                    'detection_loss_weight': feat.get('detection_loss_weight'),
                    'checkpoint': feat.get('checkpoint'),
                    'run_dir': d.get('__run_dir'),
                    'source': d.get('__source_file'),
                })
            else:
                feature_by_model[mt]['points'].append({
                    'avg_bpp': float(feat['avg_bpp']),
                    'map50': float(feat['map50']),
                    'lambda': feat.get('lambda'),
                    'checkpoint': feat.get('checkpoint'),
                    'run_dir': d.get('__run_dir'),
                    'source': d.get('__source_file'),
                })

    # Fit lines per model type
    for mt, bundle in image_by_model.items():
        pts = [(p['avg_bpp'], p['map50']) for p in bundle['points']]
        a, b = linear_fit(pts)
        image_by_model[mt]['fit'] = {'a': a, 'b': b}

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
    map_raw = merged.get('raw', {}).get('map50_mean')
    if map_raw is not None:
        ax.axhline(y=map_raw, color='gray', linestyle='--', linewidth=1.5, label=f'Raw baseline (mAP@0.5={map_raw:.3f})')

    # Consistent colors per (category, model_type)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    color_map: Dict[Tuple[str, str], str] = {}
    def get_color(category: str, model_type: str) -> str:
        key = (category, model_type)
        if key not in color_map:
            color_map[key] = color_cycle[len(color_map) % len(color_cycle)]
        return color_map[key]

    # Plot image models
    for mt, bundle in merged.get('image', {}).get('by_model', {}).items():
        pts = bundle.get('points', [])
        if not pts:
            continue
        # sort by bpp for polyline
        pts_sorted = sorted(pts, key=lambda p: p['avg_bpp'])
        x = [p['avg_bpp'] for p in pts_sorted]
        y = [p['map50'] for p in pts_sorted]
        color = get_color('image', mt)
        ax.scatter(x, y, label=f'Image: {mt}', marker='o', color=color)
        if len(pts_sorted) >= 2:
            ax.plot(x, y, linestyle='--', alpha=0.8, color=color)

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
        ax.scatter(x, y, label=f'Feature: {mt}', marker='s', color=color)
        if len(pts_sorted) >= 2:
            ax.plot(x, y, linestyle='--', alpha=0.8, color=color)

    ax.set_xlabel('Bits Per Pixel (avg)')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Merged BPP vs mAP@0.5')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Merge multiple pipeline RD summaries and plot combined BPP vs mAP@0.5')
    parser.add_argument('--inputs', nargs='+', required=True, help='Paths to summary.json files or run directories to search')
    parser.add_argument('--pattern', type=str, default='bpp_vs_map50.json', help='Filename pattern to search within directories')
    parser.add_argument('--out_json', type=str, required=True, help='Output JSON file for merged anchors')
    parser.add_argument('--out_plot', type=str, required=True, help='Output PNG file for merged plot')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    files = find_summary_files(args.inputs, args.pattern)
    if not files:
        raise SystemExit(f'No summary files found matching pattern {args.pattern} in {args.inputs}')
    logging.info(f'Found {len(files)} summary files')

    merged = merge_points(files)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(merged, f, indent=2)
    logging.info(f'Wrote merged anchors to {out_json}')

    plot_merged(merged, Path(args.out_plot))
    logging.info(f'Wrote merged plot to {args.out_plot}')


if __name__ == '__main__':
    main()

