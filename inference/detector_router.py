"""Second synthetic/natural router: features from detector outputs instead of audio statistics.

Shared by train/train_detector_router.py (training) and inference/route_combined.py (Track 2 Phase 2 routing).
Per detector, 14 shape statistics of the presence / onset / offset frame tracks; the organisers' synthetic clips give
saturated, flat, sharply switching tracks, natural recordings soft ones.
"""
import numpy as np

MEMBERS = ["cb_big", "gv_lg3_A", "rb_beats_gold"]


def clip_features(ps, ons, offs):
    f = []
    for p, on, off in zip(ps, ons, offs):
        d = np.abs(np.diff(p)) if len(p) > 1 else np.array([0.0])
        f += [p.mean(), (p > 0.9).mean(), (p < 0.05).mean(), (p > 0.5).mean(),
              d.mean(), d.max() if len(d) else 0.0, np.percentile(p, 90), np.percentile(p, 10),
              on.max(), on.mean(), (on > 0.5).mean(), off.max(), off.mean(), (off > 0.5).mean()]
    return f


def build(probs_dir, ids):
    """Feature matrix for the clips that have all three probability caches (infer.py --save-probs)."""
    Z = {m: np.load(f"{probs_dir}/{m}.npz") for m in MEMBERS}
    X, keep = [], []
    for c in ids:
        try:
            ps = [Z[m][c].astype(np.float32) for m in MEMBERS]
            ons = [Z[m][f"{c}||on"].astype(np.float32) for m in MEMBERS]
            offs = [Z[m][f"{c}||off"].astype(np.float32) for m in MEMBERS]
        except KeyError:
            continue
        X.append(clip_features(ps, ons, offs))
        keep.append(c)
    return np.array(X), keep
