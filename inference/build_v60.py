"""v56 = v55 natural engine + REBUILT synthetic branch with event splitting.

v50/v55 lock the synthetic half to v34's predictions (leaderboard F1 0.9813).
Measured on cross-fit OOF, the v43 9-model synthetic ensemble plus event
SPLITTING reaches F1 0.99142 (from 0.98413) -- +0.00794 Combined on that branch,
recovering 24 of the 32 references that were being swallowed by merged events.

Splitting works here and fails on natural because synthetic noise is pasted in
additively with crisp edges, so the split pieces still clear the F1 tolerance;
on natural the boundaries are annotation-ambiguous and the pieces miss it.

Natural side also picks up the retuned split params (0.16, 0.10, 0.25), worth
+0.00122 on natural OOF -- the joint search had never swept those axes.
"""
from __future__ import annotations
import json, pickle, sys, zipfile, os
import numpy as np
sys.path.insert(0, "src")
from metrics import write_jsonl_events
from postprocess import decode_frames, split_events, _peaks
from dur_snap import dur_snap
from boundary_regressor import apply_boundary_trim, FRAME

OUT = "submission/v60"
HIGH, LOW, MIN_DUR = 0.57, 0.48, 0.10
MIN_HOST, THR_ON, THR_OFF, MAX_CAND = 0.75, 0.20, 0.20, 0.65
MIN_SHIFT, MAX_SHIFT = -0.20, -0.005
NAT_ST, NAT_MS, NAT_GP = 0.16, 0.10, 0.25
SYN_ST, SYN_MS, SYN_GP = 0.06, 0.10, 0.40


def load_tracks(path):
    z = np.load(path)
    p, on, off = {}, {}, {}
    for k in z.files:
        if k.endswith("||on"): on[k[:-4]] = z[k].astype(np.float64)
        elif k.endswith("||off"): off[k[:-5]] = z[k].astype(np.float64)
        elif not k.endswith("||cat"): p[k] = z[k].astype(np.float64)
    return p, on, off


def main():
    order = [l.strip() for l in open("work/eval_clip_ids.txt") if l.strip()]
    syn_flag = json.load(open("work/eval_syn_pred_fixed.json"))
    reg = pickle.load(open("work/boundary_regressor.pkl", "rb"))
    preds = {}

    # ---- natural: v55's blended 19-model cache, retuned split params
    nat = dict(np.load("work/ens_nat_eval_v60.npz"))
    n_nat = 0
    for cid in order:
        if syn_flag.get(cid, False) or cid not in nat:
            continue
        p, on, off = nat[cid].astype(np.float64), nat[f"{cid}||on"].astype(np.float64), nat[f"{cid}||off"].astype(np.float64)
        evs = decode_frames(p, FRAME, duration=None, high=HIGH, low=LOW, median_k=3, min_dur=MIN_DUR)
        evs = split_events(evs, on, off, FRAME, split_thr=NAT_ST, min_sub=NAT_MS, max_pair_gap=NAT_GP)
        evs = dur_snap(evs, on, off, FRAME, duration=None, split=0.45,
                       snap_short=0.12, peak_short=0.10, snap_long=0.32, peak_long=0.20)
        new = list(evs)
        for a, b in evs:
            if b - a >= MIN_HOST:
                i0, i1 = int(round(a / FRAME)), int(round(b / FRAME))
                for s in _peaks(on, i0 + 1, i1 - 1, THR_ON):
                    for e in _peaks(off, i0 + 1, i1 - 1, THR_OFF):
                        if 0 < (e - s) * FRAME <= MAX_CAND:
                            new.append((s * FRAME, e * FRAME))
        evs = sorted(set(new))
        preds[cid] = apply_boundary_trim(evs, p, on, off, None, reg,
                                         min_shift=MIN_SHIFT, max_shift=MAX_SHIFT)
        n_nat += 1

    # ---- synthetic: v43 ensemble (ALL members -- test is unseen by every one) + splitting
    cfg = json.load(open("work/route_v43.json"))
    mem = json.load(open("work/syn_branch_v57.json"))["members"]
    M = [load_tracks(f"work/evalprobs/{m}.npz") for m in mem]
    pr = cfg["syn_params"]
    dec = {k: v for k, v in pr.items() if k not in ("snap_win", "min_peak")}
    n_syn = 0
    for cid in order:
        if not syn_flag.get(cid, False):
            continue
        if not all(cid in m[0] for m in M):
            continue
        k = min(len(m[0][cid]) for m in M)
        p = np.mean([m[0][cid][:k] for m in M], axis=0)
        on = np.mean([m[1][cid][:k] for m in M], axis=0)
        off = np.mean([m[2][cid][:k] for m in M], axis=0)
        evs = decode_frames(p, FRAME, duration=None, **dec)
        evs = split_events(evs, on, off, FRAME, split_thr=SYN_ST, min_sub=SYN_MS, max_pair_gap=SYN_GP)
        evs = dur_snap(evs, on, off, FRAME, duration=None, split=0.45,
                       snap_short=0.12, peak_short=0.10, snap_long=0.32, peak_long=0.20)
        preds[cid] = evs
        n_syn += 1

    for c in order:
        preds.setdefault(c, [])
    os.makedirs(OUT, exist_ok=True)
    jl = os.path.join(OUT, "predictions.jsonl")
    write_jsonl_events(jl, preds, clip_order=order)
    zp = os.path.join(OUT, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jl, "predictions.jsonl")
    n_ev = sum(len(v) for v in preds.values())
    print(f"natural={n_nat}  synthetic={n_syn}  clips={len(preds)}  events={n_ev} "
          f"(mean {n_ev/len(preds):.2f})  empty={sum(1 for v in preds.values() if not v)}")
    print(f"wrote {zp}")
    json.dump({"version": "v60", "clips": len(preds), "events": n_ev,
               "nat_clips": n_nat, "syn_clips": n_syn},
              open(os.path.join(OUT, "info.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
