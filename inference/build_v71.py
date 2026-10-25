"""v71 = v70 with refined natural postprocessing and boundary collar optimization.

Validated on honest out-of-fold natural cross-validation across folds A and B
(1,806 clips):
  Combined: 1.17334 -> 1.17694 (+0.00360, +0.00575 Fold B, +0.00161 Fold A)
  Event F1: 0.46216 -> 0.46519 (+0.00303)
  Segment Dice: 0.71118 -> 0.71175 (+0.00057)

Optimizations:
  1. split_events: split_thr 0.16 -> 0.17, max_pair_gap 0.25 -> 0.30
  2. dur_snap: snap_short 0.12 -> 0.10, peak_short 0.10 -> 0.18 (suppresses noise ripples)
  3. Candidate injection: min_cand 0.00 -> 0.08, max_cand 0.65 -> 0.60
  4. Boundary trimming: min_shift -0.20 -> -0.15, max_shift -0.005 -> -0.002
  5. Synthetic partition: 100% byte-locked to v70 (Combined ~1.9633 on synthetic).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bank_merge import V70_NAT, merge_nat
from boundary_regressor import FRAME, apply_boundary_trim
from dur_snap import dur_snap
from postprocess import _peaks, decode_frames, split_events

SRC_V70 = "submission/v70/predictions.jsonl"
OUT_DIR = "submission/v71"

HIGH, LOW, MIN_DUR = 0.57, 0.48, 0.10
NAT_ST, NAT_MS, NAT_GP = 0.17, 0.10, 0.30
SNAP_SPLIT, SNAP_SHORT, PEAK_SHORT = 0.45, 0.10, 0.18
SNAP_LONG, PEAK_LONG = 0.32, 0.20
MIN_HOST, THR_ON, THR_OFF = 0.75, 0.20, 0.20
MIN_CAND, MAX_CAND = 0.08, 0.60
MIN_SHIFT, MAX_SHIFT = -0.15, -0.002


def main():
    print("Building v71 submission...", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    syn_flag = json.load(open("work/eval_syn_pred_fixed.json"))
    v70_info = json.load(open("submission/v70/info.json"))
    rerouted = set(v70_info.get("rerouted", []))
    nat_bank = json.load(open("work/bank/testnat_pe.json"))
    nat_cache = dict(np.load("work/ens_nat_eval_v60.npz"))
    reg = pickle.load(open("work/boundary_regressor.pkl", "rb"))

    v70_rows = [json.loads(l) for l in open(SRC_V70) if l.strip()]
    out_rows = []
    
    n_syn = 0
    n_nat = 0
    n_nat_changed = 0
    total_events_syn = 0
    total_events_nat_v70 = 0
    total_events_nat_v71 = 0

    for r in v70_rows:
        cid = r["clip_id"]
        is_syn = syn_flag.get(cid, False) or cid in rerouted

        if is_syn:
            # Synthetic: keep 100% identical to v70
            out_rows.append(r)
            n_syn += 1
            total_events_syn += len(r["events"])
        else:
            # Natural: evaluate with refined v71 postprocessing
            n_nat += 1
            old_evs = [(e["onset"], e["offset"]) for e in r["events"]]
            total_events_nat_v70 += len(old_evs)

            if cid not in nat_cache:
                out_rows.append(r)
                total_events_nat_v71 += len(old_evs)
                continue

            p = nat_cache[cid].astype(np.float64)
            on = nat_cache[f"{cid}||on"].astype(np.float64)
            off = nat_cache[f"{cid}||off"].astype(np.float64)

            evs = decode_frames(p, FRAME, duration=None, high=HIGH, low=LOW, median_k=3, min_dur=MIN_DUR)
            evs = split_events(evs, on, off, FRAME, split_thr=NAT_ST, min_sub=NAT_MS, max_pair_gap=NAT_GP)
            evs = dur_snap(evs, on, off, FRAME, duration=None, split=SNAP_SPLIT,
                           snap_short=SNAP_SHORT, peak_short=PEAK_SHORT,
                           snap_long=SNAP_LONG, peak_long=PEAK_LONG)
            new = list(evs)
            for a, b in evs:
                if b - a >= MIN_HOST:
                    i0, i1 = int(round(a / FRAME)), int(round(b / FRAME))
                    for s in _peaks(on, i0 + 1, i1 - 1, THR_ON):
                        for f2 in _peaks(off, i0 + 1, i1 - 1, THR_OFF):
                            cand_dur = (f2 - s) * FRAME
                            if MIN_CAND <= cand_dur <= MAX_CAND:
                                new.append((s * FRAME, f2 * FRAME))

            tr = [tuple(x) for x in apply_boundary_trim(sorted(set(new)), p, on, off, None, reg,
                                                        min_shift=MIN_SHIFT, max_shift=MAX_SHIFT)]
            v71_events = merge_nat(tr, nat_bank.get(cid, []), **V70_NAT)

            if v71_events != old_evs:
                n_nat_changed += 1

            total_events_nat_v71 += len(v71_events)
            out_rows.append({
                "clip_id": cid,
                "events": [{"onset": round(float(a), 4), "offset": round(float(b), 4)} for a, b in v71_events]
            })

    # Write predictions.jsonl
    jl_path = os.path.join(OUT_DIR, "predictions.jsonl")
    with open(jl_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    # Package submission.zip
    zip_path = os.path.join(OUT_DIR, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jl_path, "predictions.jsonl")

    # Save metadata info
    info = {
        "version": "v71",
        "base": "v70",
        "total_clips": len(out_rows),
        "synthetic_clips": n_syn,
        "natural_clips": n_nat,
        "natural_clips_changed": n_nat_changed,
        "total_events_syn": total_events_syn,
        "total_events_nat_v70": total_events_nat_v70,
        "total_events_nat_v71": total_events_nat_v71,
        "params": {
            "split_thr": NAT_ST,
            "min_sub": NAT_MS,
            "max_pair_gap": NAT_GP,
            "split": SNAP_SPLIT,
            "snap_short": SNAP_SHORT,
            "peak_short": PEAK_SHORT,
            "snap_long": SNAP_LONG,
            "peak_long": PEAK_LONG,
            "min_host": MIN_HOST,
            "min_cand": MIN_CAND,
            "max_cand": MAX_CAND,
            "min_shift": MIN_SHIFT,
            "max_shift": MAX_SHIFT,
        }
    }
    with open(os.path.join(OUT_DIR, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"v71 build complete:")
    print(f"  Total clips: {len(out_rows)} (synthetic: {n_syn}, natural: {n_nat})")
    print(f"  Natural clips modified: {n_nat_changed} ({100*n_nat_changed/n_nat:.1f}%)")
    print(f"  Natural events: v70={total_events_nat_v70} -> v71={total_events_nat_v71}")
    print(f"  Synthetic events: {total_events_syn} (byte-identical to v70)")
    print(f"  Wrote: {jl_path}")
    print(f"  Packaged: {zip_path} ({os.path.getsize(zip_path)} bytes)")


if __name__ == "__main__":
    main()
