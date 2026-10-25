#!/usr/bin/env python3
"""Train the detector-output router (assets/detector_router.pkl), used only for the Track 2 Phase 2 routing.

    python train/train_detector_router.py <validation probs dir> work/val_holdA.jsonl work/val_holdB.jsonl assets/detector_router.pkl

<validation probs dir> holds infer.py --save-probs caches of cb_big, gv_lg3_A and rb_beats_gold on the 2,501 official
validation clips; labels are the clips' syntheticData flags (field "synthetic" of the manifests). Random forest,
300 trees, random_state 0 (5-fold CV accuracy 99.36 %).
"""
import json
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from detector_router import build


def main():
    probs_dir, hold_a, hold_b, out = sys.argv[1:5]
    val = [json.loads(l) for h in (hold_a, hold_b) for l in open(h) if l.strip()]
    label = {r["clip_id"]: 1 if r.get("synthetic") else 0 for r in val}
    X, keep = build(probs_dir, [r["clip_id"] for r in val])
    y = np.array([label[c] for c in keep])
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    print(f"detector router: {len(keep)} validation clips | 5-fold CV accuracy "
          f"{100 * cross_val_score(clf, X, y, cv=5).mean():.2f}%", flush=True)
    clf.fit(X, y)
    joblib.dump(clf, out)


if __name__ == "__main__":
    main()
