#!/usr/bin/env python3
"""Combined synthetic/natural routing, used for the Track 2 Phase 2 test audio (CodeAMU Track 2 inputs/phase2/route.json).

    python inference/route_combined.py          # cwd = run dir, after stage 2 of run_pipeline.py

On the Phase 2 audio the audio router drifted (20.7 % of clips in its uncertain band 0.3-0.7, vs 0.2 % on Phase 1), so
its probability (work/eval_synclf_proba.npy, written by stage 2) is averaged with the detector router's
(work/detector_router.pkl on work/evalprobs/{cb_big,gv_lg3_A,rb_beats_gold}.npz). Synthetic iff the mean >= 0.61, the
cost-based break-even: on validation a natural clip decoded by the synthetic branch costs 0.095 Combined, the reverse
0.061. Writes work/route_combined.json ({clip_id: bool}).
"""
import json

import joblib
import numpy as np

from detector_router import build

TH = 0.61


def main():
    ids = [l.strip() for l in open("work/eval_clip_ids.txt") if l.strip()]
    p_aud = dict(zip(ids, np.load("work/eval_synclf_proba.npy")))
    X, keep = build("work/evalprobs", ids)
    p_det = dict(zip(keep, joblib.load("work/detector_router.pkl").predict_proba(X)[:, 1]))
    route = {c: bool(0.5 * (float(p_aud[c]) + float(p_det.get(c, p_aud[c]))) >= TH) for c in ids}
    json.dump(route, open("work/route_combined.json", "w"))
    print(f"{len(ids)} clips | synthetic {sum(route.values())} (audio router alone at 0.5: "
          f"{sum(bool(p >= 0.5) for p in p_aud.values())})", flush=True)


if __name__ == "__main__":
    main()
