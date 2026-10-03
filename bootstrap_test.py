#!/usr/bin/env python3
"""
Paired bootstrap of the proposed Whisper+MI model vs. the handcrafted-only Random Forest
on the official test set (Section 5.1). Reports the difference in Macro-F1, accuracy and
Western F1 with 95% percentile intervals, and a two-sided p-value for Western F1.

Usage:
  python paper/bootstrap_comment.py --labels <test_labels.txt> --prop-pred <predictions.txt>

Files: one line per clip, "<file_id> <Dialect>".
"""
import os
import sys
import json
import argparse
import numpy as np

NAMES = ['Central', 'Northern', 'Southern', 'Western']
WEST = 3
SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def rp(p):
    """Absolute paths and existing relative paths are kept; otherwise resolve against ROOT."""
    if os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(ROOT, p)


def read_labelled(path):
    d = {}
    for line in open(path, encoding='utf-8'):
        toks = line.strip().replace(',', ' ').split()
        if len(toks) < 2:
            continue
        fid = os.path.splitext(os.path.basename(toks[0]))[0]
        lab, idx = toks[-1], None
        for i, n in enumerate(NAMES):
            if n.lower() in lab.lower():
                idx = i
        if idx is None and lab.isdigit():
            idx = int(lab)
        if idx is not None:
            d[fid] = idx
    return d


def handcrafted_predict(train_cache, test_path, out_path):
    """Handcrafted-only baseline: Random Forest trained on all training data."""
    from prototype_model import FeatureExtractor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    data = np.load(train_cache, allow_pickle=True).item()
    X, y = data['feats'], data['labels']
    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED,
                                 class_weight='balanced', n_jobs=-1)
    clf.fit(scaler.transform(X), y)

    ext = FeatureExtractor()
    files = sorted(f for f in os.listdir(test_path) if f.endswith('.wav'))
    feats = []
    for f in files:
        v = ext.extract(os.path.join(test_path, f))  # no transcript at test time
        feats.append(v if v is not None else np.zeros(X.shape[1]))
    preds = clf.predict(scaler.transform(np.array(feats)))
    with open(out_path, 'w') as fh:
        for f, p in zip(files, preds):
            fh.write(f"{os.path.splitext(f)[0]} {NAMES[p]}\n")
    print(f"Handcrafted test predictions written to {out_path}")


def f1_stats(y, p, k=4):
    """Per-class F1, macro F1 and accuracy from a confusion matrix."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1, f1.mean(), tp.sum() / cm.sum()


def ci(a):
    return np.percentile(a, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help="official test labels")
    ap.add_argument('--prop-pred', required=True, help="proposed model's test predictions")
    ap.add_argument('--hand-pred', default=None, help="handcrafted predictions (skips RF step)")
    ap.add_argument('--train-cache', default='handcrafted_features_baseline.npy')
    ap.add_argument('--test-path', default='data/Test')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--out', default='cv_results/bootstrap_results.json')
    a = ap.parse_args()

    a.labels, a.prop_pred = rp(a.labels), rp(a.prop_pred)
    a.train_cache, a.test_path = rp(a.train_cache), rp(a.test_path)
    a.out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    hand_path = rp(a.hand_pred) if a.hand_pred else None
    if hand_path is None:
        hand_path = os.path.join(ROOT, 'paper/review/handcrafted_test_predictions.txt')
        os.makedirs(os.path.dirname(hand_path), exist_ok=True)
        handcrafted_predict(a.train_cache, a.test_path, hand_path)

    lab, prop, hand = read_labelled(a.labels), read_labelled(a.prop_pred), read_labelled(hand_path)
    ids = sorted(set(lab) & set(prop) & set(hand))
    print(f"Labels: {len(lab)}, proposed: {len(prop)}, handcrafted: {len(hand)}, common: {len(ids)}")
    if not ids:
        raise SystemExit("No common file ids - check id formats and that --labels is the TEST labels file.")
    y = np.array([lab[i] for i in ids])
    pp = np.array([prop[i] for i in ids])
    ph = np.array([hand[i] for i in ids])

    f1p, mp, accp = f1_stats(y, pp)
    f1h, mh, acch = f1_stats(y, ph)
    print(f"\nPoint estimates (n={len(y)})")
    print(f"  Proposed    : Acc={accp:.4f}  MacroF1={mp:.4f}  West F1={f1p[WEST]:.3f}")
    print(f"  Handcrafted : Acc={acch:.4f}  MacroF1={mh:.4f}  West F1={f1h[WEST]:.3f}")

    # Paired bootstrap: the same resampled test clips are scored for both systems.
    rng = np.random.default_rng(SEED)
    n, B = len(y), a.n_boot
    d_macro, d_acc, d_west = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        fp_, mp_, ap_ = f1_stats(y[idx], pp[idx])
        fh_, mh_, ah_ = f1_stats(y[idx], ph[idx])
        d_macro[b], d_acc[b], d_west[b] = mp_ - mh_, ap_ - ah_, fp_[WEST] - fh_[WEST]

    def p_two_sided(d):
        return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))

    res = {
        'n_test': int(n), 'n_boot': B, 'seed': SEED,
        'macro_f1': {'proposed': mp, 'handcrafted': mh, 'delta': mp - mh,
                     'delta_ci95': ci(d_macro).tolist()},
        'accuracy': {'proposed': accp, 'handcrafted': acch, 'delta': accp - acch,
                     'delta_ci95': ci(d_acc).tolist()},
        'western_f1': {'proposed': float(f1p[WEST]), 'handcrafted': float(f1h[WEST]),
                       'delta': float(f1p[WEST] - f1h[WEST]),
                       'delta_ci95': ci(d_west).tolist(),
                       'p_two_sided': p_two_sided(d_west)},
    }

    print("\nPaired bootstrap, proposed - handcrafted (95% percentile CI)")
    for key, label in [('macro_f1', 'Macro-F1'), ('accuracy', 'Accuracy'), ('western_f1', 'Western F1')]:
        r = res[key]
        print(f"  {label:10s} {r['delta']:+.4f}  [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}]")
    print(f"  Western F1 two-sided p = {res['western_f1']['p_two_sided']:.4f}")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2)
    print(f"\nSaved {a.out}")


if __name__ == '__main__':
    main()