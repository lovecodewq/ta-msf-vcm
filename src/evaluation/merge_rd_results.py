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
    raw_map_values: List[float] = []
    image_by_model: Dict[str, Dict[str, Any]] = {}
    feature_by_model: Dict[str, Dict[str, Any]] = {}

    for fp in files:
        d = load_point(fp)
        # raw
        raw = d.get('raw', {})
        if isinstance(raw, dict) and 'map50' in raw:
            try:
                raw_map_values.append(float(raw['map50']))
            except Exception:
                pass

        # image
        img = d.get('image_compression', {})
        if 'avg_bpp' in img and 'map50' in img:
            mt = img.get('model_type', 'image_model')
            image_by_model.setdefault(mt, {'points': []})
            image_by_model[mt]['points'].append({
                'avg_bpp': float(img['avg_bpp']),
                'map50': float(img['map50']),
                'lambda': img.get('lambda'),
                'checkpoint': img.get('checkpoint'),
                'run_dir': d.get('__run_dir'),
                'source': d.get('__source_file'),
            })

        # feature
        feat = d.get('feature_compression', {})
        if 'avg_bpp' in feat and 'map50' in feat:
            mt = feat.get('model_type', 'feature_model')
            feature_by_model.setdefault(mt, {'points': []})
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
        pts = [(p['avg_bpp'], p['map50']) for p in bundle['points']]
        a, b = linear_fit(pts)
        feature_by_model[mt]['fit'] = {'a': a, 'b': b}

    merged = {
        'raw': {
            'map50_values': raw_map_values,
            'map50_mean': float(np.mean(raw_map_values)) if raw_map_values else None,
            'n': len(raw_map_values),
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

    # Plot image models
    for mt, bundle in merged.get('image', {}).get('by_model', {}).items():
        pts = bundle.get('points', [])
        if not pts:
            continue
        x = [p['avg_bpp'] for p in pts]
        y = [p['map50'] for p in pts]
        ax.scatter(x, y, label=f'Image: {mt}', marker='o')
        a = bundle.get('fit', {}).get('a')
        b = bundle.get('fit', {}).get('b')
        if a == a and b == b:  # not NaN
            xs = np.linspace(min(x), max(x), 50)
            ys = a * xs + b
            ax.plot(xs, ys, linestyle='-', alpha=0.6)

    # Plot feature models
    for mt, bundle in merged.get('feature', {}).get('by_model', {}).items():
        pts = bundle.get('points', [])
        if not pts:
            continue
        x = [p['avg_bpp'] for p in pts]
        y = [p['map50'] for p in pts]
        ax.scatter(x, y, label=f'Feature: {mt}', marker='s')
        a = bundle.get('fit', {}).get('a')
        b = bundle.get('fit', {}).get('b')
        if a == a and b == b:
            xs = np.linspace(min(x), max(x), 50)
            ys = a * xs + b
            ax.plot(xs, ys, linestyle='-', alpha=0.6)

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

