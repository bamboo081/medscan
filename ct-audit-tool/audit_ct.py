#!/usr/bin/env python3
"""Read-only leakage + shortcut-cue audit for image-level CT classification.

Uses existing split manifest (path,label,split), requires no patient IDs or training.
Outputs are candidate signals, not proof of matching patient identity.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter

LABEL_COLS = ('label', 'class', 'target')
PATH_COLS = ('path', 'image_path', 'filepath', 'file_path')
SPLITS = ('train', 'val', 'test')
FEATURES = [
    'width', 'height', 'aspect_ratio', 'file_size_kb', 'mean_intensity',
    'std_intensity', 'black_fraction', 'white_fraction', 'border_mean',
    'border_black_fraction', 'border_edge_fraction', 'corner_edge_fraction',
    'center_edge_fraction', 'corner_brightness', 'corner_minus_center_edges',
]


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/processed/ct/splits.csv')
    p.add_argument('--output', default='runs/ct_resnet18/audit')
    p.add_argument('--data-root', help='Optional base for relative manifest paths (defaults to manifest directory)')
    p.add_argument('--phash-distance', type=int, default=10,
                   help='Maximum pHash Hamming distance for near-duplicate candidates (default: 10)')
    p.add_argument('--neighbors', type=int, default=3,
                   help='Maximum nearest cross-split candidates per image per other split (default: 3)')
    p.add_argument('--max-near-pairs', type=int, default=300,
                   help='Max candidate pairs to rank with structural similarity (default: 300)')
    p.add_argument('--preview-pairs', type=int, default=24)
    p.add_argument('--skip-shortcut-model', action='store_true',
                   help='Skip lightweight training-only artifact-feature classifier')
    return p.parse_args()


def col_for(cols, alternatives):
    return next((c for c in alternatives if c in cols), None)


def load_manifest(manifest, data_root):
    manifest = Path(manifest).expanduser().resolve()
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    path_col = col_for(df.columns, PATH_COLS)
    label_col = col_for(df.columns, LABEL_COLS)
    if path_col is None or label_col is None or 'split' not in df.columns:
        raise ValueError('Need path, label, split columns; found: ' + ', '.join(df.columns))
    df = df.rename(columns={path_col: 'path', label_col: 'label'})
    if df[['path', 'label', 'split']].eq('').any().any():
        raise ValueError('Manifest contains blank required cells')
    df['split'] = df['split'].str.strip().str.lower().replace({'valid': 'val', 'validation': 'val'})
    if not set(df['split']).issubset(SPLITS):
        raise ValueError('Unsupported splits: ' + str(sorted(set(df['split']) - set(SPLITS))))
    root = Path(data_root).expanduser().resolve() if data_root else manifest.parent
    def resolve(name):
        p = Path(name).expanduser()
        if p.is_absolute():
            return p
        for candidate in (root / p, Path.cwd() / p):
            if candidate.is_file():
                return candidate.resolve()
        return (root / p).resolve()
    df['resolved_path'] = df['path'].map(resolve)
    missing = df[~df.resolved_path.map(Path.is_file)]
    if len(missing):
        raise FileNotFoundError(f'{len(missing)} files unavailable. First: {missing.path.iloc[0]}. Try --data-root.')
    if df.resolved_path.duplicated().any():
        raise ValueError('One or more paths are repeated in the manifest')
    return df.reset_index(drop=True)


def phash(gray):
    small = gray.resize((32, 32), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.float32)
    freq = dctn(a, type=2, norm='ortho')[:8, :8].reshape(-1)
    # Exclude DC component from median, as in common pHash variants.
    bits = (freq > np.median(freq[1:])).astype(np.uint8)
    return int.from_bytes(np.packbits(bits).tobytes(), byteorder='big')


def edge_fraction(img):
    if min(img.shape) < 3:
        return 0.0
    a = np.abs(np.diff(img, axis=0))[:, :-1]
    b = np.abs(np.diff(img, axis=1))[:-1, :]
    return float(np.mean((a + b) > 36))


def analyze_one(path):
    path = Path(path)
    raw_digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            raw_digest.update(block)
    with Image.open(path) as img:
        gray = ImageOps.exif_transpose(img).convert('L')
        w, h = gray.size
        a = np.asarray(gray, dtype=np.uint8)
        pixel_hash = hashlib.sha256(f'{w}x{h}|'.encode() + a.tobytes()).hexdigest()
        fingerprint = phash(gray)
        small = np.asarray(gray.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    rim = 12
    border_mask = np.ones((128, 128), dtype=bool)
    border_mask[rim:-rim, rim:-rim] = False
    border = small[border_mask]
    corner_regions = [small[:28, :28], small[:28, -28:], small[-28:, :28], small[-28:, -28:]]
    corners = np.concatenate([a.ravel() for a in corner_regions])
    corner_edge = float(np.mean([edge_fraction(c) for c in corner_regions]))
    center_edge = edge_fraction(small[42:86, 42:86])
    # These engineered features are only *proxies* for potential visible shortcuts.
    f = dict(
        width=w, height=h, aspect_ratio=w / h,
        file_size_kb=path.stat().st_size / 1024,
        mean_intensity=float(small.mean()), std_intensity=float(small.std()),
        black_fraction=float(np.mean(small < 18)), white_fraction=float(np.mean(small > 237)),
        border_mean=float(border.mean()), border_black_fraction=float(np.mean(border < 18)),
        border_edge_fraction=edge_fraction(small[:12, :]),
        corner_edge_fraction=corner_edge, center_edge_fraction=center_edge,
        corner_brightness=float(corners.mean()),
        corner_minus_center_edges=float(corner_edge - center_edge),
    )
    return raw_digest.hexdigest(), pixel_hash, fingerprint, f


def exact_groups(df, key):
    groups = []
    for digest, grp in df.groupby(key, sort=False):
        if len(grp) < 2:
            continue
        group = {'digest': digest, 'size': len(grp),
                 'cross_split': grp.split.nunique() > 1,
                 'cross_label': grp.label.nunique() > 1,
                 'splits': sorted(grp.split.unique().tolist()),
                 'labels': sorted(grp.label.unique().tolist()),
                 'paths': grp.path.tolist()[:10]}
        groups.append(group)
    return groups


def exact_pairs(df):
    records = []
    for _, group in df.groupby('pixel_sha256', sort=False):
        if len(group) < 2:
            continue
        by_split = {s: g for s, g in group.groupby('split')}
        for i, sa in enumerate(SPLITS):
            if sa not in by_split:
                continue
            for sb in SPLITS[i+1:]:
                if sb in by_split:
                    a = by_split[sa].iloc[0]
                    b = by_split[sb].iloc[0]
                    records.append(make_pair(a, b, 0, 1.0, exact=True))
    return records


def make_pair(a, b, distance, ssim, exact=False):
    return {
        'split_a': a['split'], 'label_a': a['label'], 'path_a': a['path'],
        'split_b': b['split'], 'label_b': b['label'], 'path_b': b['path'],
        'phash_distance': int(distance), 'ssim_128': float(ssim),
        'exact_pixels': bool(exact),
        'label_conflict': bool(a['label'] != b['label']),
        'index_a': int(a.name), 'index_b': int(b.name),
    }


def find_candidates(df, radius, neighbors):
    """Batch vectorized 64-bit Hamming search; no N-by-N similarity matrix stored."""
    by_split = {s: df.index[df.split == s].to_numpy(dtype=np.int64) for s in SPLITS}
    hashes = df['phash64'].to_numpy(dtype=np.uint64)
    pairs = {}
    for i, sa in enumerate(SPLITS):
        for sb in SPLITS[i+1:]:
            aidx, bidx = by_split[sa], by_split[sb]
            if not len(aidx) or not len(bidx):
                continue
            bvals = hashes[bidx]
            for start in range(0, len(aidx), 128):
                block = aidx[start:start+128]
                xor = np.bitwise_xor(hashes[block, None], bvals[None, :])
                if hasattr(np, 'bitwise_count'):
                    dist = np.bitwise_count(xor)
                else:  # Compatibility fallback for older NumPy.
                    table = np.array([bin(n).count('1') for n in range(256)], dtype=np.uint8)
                    dist = table[xor.view(np.uint8).reshape(len(block), len(bidx), 8)].sum(axis=2)
                for j, row in enumerate(dist):
                    eligible = np.flatnonzero(row <= radius)
                    if len(eligible) == 0:
                        continue
                    if len(eligible) > neighbors:
                        chosen = eligible[np.argpartition(row[eligible], neighbors-1)[:neighbors]]
                    else:
                        chosen = eligible
                    for k in chosen:
                        u, v = int(block[j]), int(bidx[k])
                        if df.loc[u, 'pixel_sha256'] == df.loc[v, 'pixel_sha256']:
                            continue
                        pairs[(u, v)] = int(row[k])
    return pairs


def normalize_for_ssim(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('L').resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def ssim_128(a, b):
    # Conventional luminance/contrast/structure index using Gaussian local statistics.
    x, y = normalize_for_ssim(a), normalize_for_ssim(b)
    ux = gaussian_filter(x, 1.5); uy = gaussian_filter(y, 1.5)
    vx = np.maximum(0, gaussian_filter(x*x, 1.5) - ux*ux)
    vy = np.maximum(0, gaussian_filter(y*y, 1.5) - uy*uy)
    vxy = gaussian_filter(x*y, 1.5) - ux*uy
    c1 = .01**2; c2 = .03**2
    return float(np.mean((2*ux*uy+c1)*(2*vxy+c2)/((ux*ux+uy*uy+c1)*(vx+vy+c2))))


def save_grid(pairs, df, dest, count):
    pairs = pairs[:count]
    if not pairs:
        return
    width, height, pad, title_h = 256, 210, 12, 46
    grid = Image.new('RGB', (width * 2 + pad * 3, (height + title_h + pad) * len(pairs) + pad), 'white')
    draw = ImageDraw.Draw(grid)
    for n, p in enumerate(pairs):
        for col, suffix in enumerate(('a', 'b')):
            path = df.loc[p[f'index_{suffix}'], 'resolved_path']
            with Image.open(path) as im:
                thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (width, height))
            x = pad + col * (width + pad)
            y = pad + n * (height + title_h + pad)
            grid.paste(thumb, (x + (width-thumb.width)//2, y))
            caption = f"{p[f'split_{suffix}']} | {p[f'label_{suffix}']}"
            draw.text((x, y+height+3), caption[:42], fill='black')
            draw.text((x, y+height+18), Path(str(p[f'path_{suffix}'])).name[:38], fill='black')
        draw.text((pad, y+height+33), f"pHash Hamming={p['phash_distance']}, SSIM={p['ssim_128']:.3f}"[:78], fill='black')
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid.save(dest, optimize=True)


def artifact_stats(df, out, train_only=True):
    subset = df[df.split == 'train'] if train_only and 'train' in set(df.split) else df
    # Long-form feature summary by class. Use only the training split.
    records = []
    for label, group in subset.groupby('label'):
        for feature in FEATURES:
            values = group[feature]
            records.append({'label':label,'feature':feature,'n':len(group),
                            'p10':float(values.quantile(.1)), 'median':float(values.median()),
                            'p90':float(values.quantile(.9))})
    pd.DataFrame(records).to_csv(out / 'artifact_feature_by_class.csv', index=False)


def shortcut_probe(df, out):
    """Lightweight train-only feature probe, not a valid generalization assessment."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = df[df.split == 'train']
    counts = train.label.value_counts()
    if len(counts) < 2 or int(counts.min()) < 4:
        return {'skipped': 'Need >=4 train examples in at least two classes'}
    X = train[FEATURES].to_numpy(dtype=float)
    y = train.label.to_numpy()
    try:
        X0, X1, y0, y1 = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight='balanced'))
        estimator.fit(X0, y0)
        pred = estimator.predict(X1)
        score = balanced_accuracy_score(y1, pred)
        report = classification_report(y1, pred, output_dict=True, zero_division=0)
        coef = np.mean(np.abs(estimator.named_steps['logisticregression'].coef_), axis=0)
        pd.DataFrame({'feature': FEATURES, 'mean_abs_standardized_coefficient':coef}).sort_values(
            'mean_abs_standardized_coefficient', ascending=False).to_csv(out/'artifact_feature_importance.csv', index=False)
        return {'train_only_internal_validation_balanced_accuracy': float(score),
                'random_chance_balanced_accuracy': 1/len(counts),
                'classes': sorted(counts.index.tolist()),
                'warning': 'Image-level internal holdout; shared patients may inflate score. This probe uses only metadata/intensity/borders, not anatomy.',
                'classification_report': report}
    except ValueError as e:
        return {'skipped': str(e)}


def main():
    args = cli()
    if args.max_near_pairs < 0 or args.preview_pairs < 0 or args.neighbors < 1 or not 0 <= args.phash_distance <= 64:
        raise ValueError('Invalid CLI limits')
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_manifest(args.manifest, args.data_root)
    print(f'Loaded manifest: {len(df)} images, classes {sorted(df.label.unique())}; no training performed.', flush=True)
    data = []
    for i, row in df.iterrows():
        try:
            raw_hash, pixel_hash, ph, feats = analyze_one(row.resolved_path)
        except (OSError, ValueError, UnidentifiedImageError) as e:
            raise RuntimeError(f'Cannot decode image {row.path} (manifest row {i+2}): {e}') from e
        data.append({'file_sha256': raw_hash, 'pixel_sha256': pixel_hash, 'phash64': np.uint64(ph), **feats})
        if (i + 1) % 2000 == 0:
            print(f'  Read {i+1}/{len(df)} images', flush=True)
    data_df = pd.DataFrame(data, index=df.index)
    df = pd.concat([df, data_df], axis=1)
    # Persist a per-file metadata audit with no pixel arrays or medical predictions.
    safe = df.drop(columns=['resolved_path', 'phash64'])
    safe.to_csv(out/'image_features.csv', index=False)
    raw_groups = exact_groups(df, 'file_sha256')
    pixel_groups = exact_groups(df, 'pixel_sha256')
    with (out/'exact_duplicate_groups.json').open('w') as f:
        json.dump({'same_file_bytes':raw_groups,'same_decoded_pixels':pixel_groups}, f, indent=2)
    exact = exact_pairs(df)
    pair_cols = ['split_a', 'label_a', 'path_a', 'split_b', 'label_b', 'path_b',
                 'phash_distance', 'ssim_128', 'exact_pixels', 'label_conflict']
    pd.DataFrame(exact, columns=pair_cols + ['index_a', 'index_b'])[pair_cols].to_csv(
        out/'exact_cross_split_pairs.csv', index=False)
    print(f'Exact pixel duplicates: {len(pixel_groups)} groups; cross-split pairs represented: {len(exact)}', flush=True)
    print('Searching cross-split perceptual-hash neighbors...', flush=True)
    candidates = find_candidates(df, args.phash_distance, args.neighbors)
    # Round-robin across split/label pair categories before the SSIM cap so
    # high-volume normal images do not consume the full review budget.
    buckets = defaultdict(list)
    for (a, b), distance in sorted(candidates.items(), key=lambda item: item[1]):
        key = (df.loc[a, 'split'], df.loc[b, 'split'],
               df.loc[a, 'label'], df.loc[b, 'label'])
        buckets[key].append(((a, b), distance))
    ranked = []
    keys = sorted(buckets)
    while len(ranked) < args.max_near_pairs and keys:
        remaining = []
        for key in keys:
            if len(ranked) >= args.max_near_pairs:
                break
            ranked.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    near = []
    for (a,b), distance in ranked:
        similarity = ssim_128(df.loc[a, 'resolved_path'], df.loc[b, 'resolved_path'])
        near.append(make_pair(df.loc[a], df.loc[b], distance, similarity))
    near.sort(key=lambda pair: (-pair['ssim_128'], pair['phash_distance']))
    pd.DataFrame(near, columns=pair_cols + ['index_a', 'index_b'])[pair_cols].to_csv(
        out/'near_duplicate_candidates.csv', index=False)
    save_grid(exact, df, out/'exact_cross_split_examples.jpg', args.preview_pairs)
    save_grid(near, df, out/'near_duplicate_examples.jpg', args.preview_pairs)
    artifact_stats(df, out)
    probe = {'skipped':'--skip-shortcut-model'} if args.skip_shortcut_model else shortcut_probe(df, out)
    summary = {
        'images':len(df), 'split_counts': {str(k):int(v) for k,v in df.split.value_counts().items()},
        'label_counts':{str(k):int(v) for k,v in df.label.value_counts().items()},
        'split_by_label':{s:{c:int(n) for c,n in group.label.value_counts().items()} for s,group in df.groupby('split')},
        'exact_byte_duplicate_groups':len(raw_groups),
        'exact_pixel_duplicate_groups':len(pixel_groups),
        'exact_pixel_groups_crossing_splits':sum(g['cross_split'] for g in pixel_groups),
        'exact_pixel_groups_with_label_conflict':sum(g['cross_label'] for g in pixel_groups),
        'near_candidates_before_cap':len(candidates),
        'near_candidates_ranked':len(near),
        'near_candidates_with_ssim_at_least_0_90':sum(p['ssim_128'] >= .90 for p in near),
        'near_search_phash_max_distance':args.phash_distance,
        'near_search_limitations':'Top K cross-split perceptual-hash neighbors, capped before SSIM; not exhaustive, and similarity cannot prove patient identity.',
        'artifact_feature_probe':probe,
        'data_limitations':'Unknown patient IDs, JPEG no DICOM/HU, image-level test has already been used; audit cannot establish patient-independent generalization.',
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# CT dataset audit (exploratory)', '',
             f"- Reviewed **{len(df):,} images** already referenced by the manifest; did not modify images, splits or checkpoints.",
             f"- **{len(pixel_groups)} exact decoded-pixel duplicate groups**; **{summary['exact_pixel_groups_crossing_splits']}** cross train/val/test.",
             f"- **{len(near)} near-duplicate candidates** scored; {summary['near_candidates_with_ssim_at_least_0_90']} have SSIM >= 0.90 (not proof of same patient).",
             '- See `near_duplicate_examples.jpg` and `exact_cross_split_examples.jpg` for image comparisons.',
             '- See `artifact_feature_by_class.csv` and `artifact_feature_importance.csv` for possible label-specific nonanatomical cues.',
             f"- Artifact-only training subset probe: {probe.get('train_only_internal_validation_balanced_accuracy', 'not run')}; random balanced-accuracy baseline {probe.get('random_chance_balanced_accuracy', 'N/A')}.",
             '', '## Interpretation',
             '- Any exact cross-split copies are direct leakage evidence; manually review and replace split strategy before trusting the previous score.',
             '- High pHash/SSIM similarity flags related-looking scans but cannot establish whether they share a patient.',
             '- The feature-only probe is a weak shortcut-cue diagnostic; it may itself be inflated by image-level overlap.',
             '- Artifact metrics cannot reliably recognize burned-in text. Manually inspect the montage for markings, borders, layout and class-specific patterns.',
             '- Even if no problems are found, **100% CT test macro-F1 remains exploratory** until tested on an independent patient-indexed cohort.',
             '- Do not repair the old test set and call it a fresh unbiased test. After changes, obtain a new unseen patient-independent test cohort.',
             '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(f'Wrote audit to {out}', flush=True)
    print(f"Cross-split exact pixel duplicate groups: {summary['exact_pixel_groups_crossing_splits']}")
    print(f"Near pairs scored: {len(near)}; SSIM >= 0.90: {summary['near_candidates_with_ssim_at_least_0_90']}")
    print(f"Artifact-only probe balanced accuracy: {probe.get('train_only_internal_validation_balanced_accuracy', 'not run')}")
    print('Review REPORT.md + comparison JPGs. These results cannot establish patient independence.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f'AUDIT ERROR: {e}', file=sys.stderr)
        sys.exit(2)
