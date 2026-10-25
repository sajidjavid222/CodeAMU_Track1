"""v80 = v76 Champion + Strict Intra-Host Duplicate NMS Pruning (IoU >= 0.30, dur_ratio >= 0.30).

Strategic Context & Leaderboard Objective:
  - On CodaBench Track 1, competitor Hydra submitted entry 966955 on 2026-10-07 scoring 1.6127:
      * Event F1: 0.7892
      * Segment Dice: 0.8235
      * Combined: 1.6127
      * Duration: 0.61s
  - Our v76 submission (ID 950980) scored 1.6115:
      * Event F1: 0.7853
      * Segment Dice: 0.8261 (we lead Hydra by +0.0026 Dice!)
      * Combined: 1.6115
  - Root Cause Analysis:
      * Hydra sacrificed 0.0026 Segment Dice (0.8261 -> 0.8235) by aggressively suppressing
        false alarm candidate events, boosting Event F1 to 0.7892 (+0.0039).
      * In v76, duplicate NMS pruning only pruned duplicates with IoU >= 0.50 and dur_ratio >= 0.50.
      * Full 2D grid sweep on honest out-of-fold validation (Folds A & B, 1,806 clips) proved:
          - iou_thr = 0.30, dur_ratio = 0.30 prunes 60 shadow duplicate false alarms on the test set.
          - Combined: 1.17683 -> 1.17826 (+0.00143, +0.00220 Fold A, +0.00063 Fold B, strictly positive on both folds).
          - Event F1: 0.46510 -> 0.46681 (+0.00171).
          - Segment Dice: 0.71173 -> 0.71145 (-0.00028).
  - Projected CodaBench Score for v80:
      * Event F1: 0.7870 - 0.7885
      * Segment Dice: 0.8258 - 0.8261 (retaining our huge Dice advantage over Hydra)
      * Combined Score: 1.6130 - 1.6145 -> RECLAIM RANK 1 ON CODABENCH!
"""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np

SRC_V76 = "submission/v76/predictions.jsonl"
OUT_DIR = "submission/v80"
FRAME = 0.02

IOU_THR = 0.30
DUR_RATIO = 0.30


def prune_duplicates(events: list[dict], p_track: np.ndarray,
                     iou_thr: float = IOU_THR, dur_ratio: float = DUR_RATIO) -> tuple[list[dict], list[dict]]:
    if len(events) <= 1:
        return events, []
    scored = []
    L = len(p_track)
    for e in events:
        a, b = e["onset"], e["offset"]
        i0 = max(0, min(L - 1, int(round(a / FRAME))))
        i1 = max(0, min(L - 1, int(round(b / FRAME))))
        seg = p_track[i0:max(i0 + 1, i1)]
        mean_p = float(np.mean(seg)) if len(seg) > 0 else 0.0
        scored.append((e, mean_p))

    scored.sort(key=lambda x: x[1], reverse=True)
    kept = []
    pruned = []
    for ev, score in scored:
        a1, b1 = ev["onset"], ev["offset"]
        suppress = False
        host = None
        for ev2 in kept:
            a2, b2 = ev2["onset"], ev2["offset"]
            inter = max(0.0, min(b1, b2) - max(a1, a2))
            union = max(b1, b2) - min(a1, a2)
            iou = inter / union if union > 0 else 0
            if iou >= iou_thr:
                dur1 = b1 - a1
                dur2 = b2 - a2
                if min(dur1, dur2) / max(dur1, dur2) >= dur_ratio:
                    suppress = True
                    host = ev2
                    break
        if suppress:
            pruned.append({"pruned": ev, "host": host, "score": score})
        else:
            kept.append(ev)
    return sorted(kept, key=lambda x: (x["onset"], x["offset"])), pruned


def main():
    print("Building v80 submission...", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    syn_flag = json.load(open("work/eval_syn_pred_fixed.json"))
    v70_info = json.load(open("submission/v70/info.json"))
    rerouted = set(v70_info.get("rerouted", []))
    nat_cache = dict(np.load("work/ens_nat_eval_v60.npz"))

    v76_rows = [json.loads(l) for l in open(SRC_V76) if l.strip()]
    out_rows = []

    n_syn = 0
    n_nat = 0
    n_nat_pruned = 0
    total_events_syn_v80 = 0
    total_events_nat_v76 = 0
    total_events_nat_v80 = 0

    pruned_details = []

    for r in v76_rows:
        cid = r["clip_id"]
        is_syn = syn_flag.get(cid, False) or cid in rerouted

        if is_syn:
            # 100% byte-locked to the champion v76 synthetic branch
            n_syn += 1
            total_events_syn_v80 += len(r["events"])
            out_rows.append(r)
        else:
            n_nat += 1
            evs = r["events"]
            total_events_nat_v76 += len(evs)

            p = nat_cache.get(cid)
            if p is None:
                out_rows.append(r)
                total_events_nat_v80 += len(evs)
                continue

            cleaned, pruned = prune_duplicates(evs, p, iou_thr=IOU_THR, dur_ratio=DUR_RATIO)
            if pruned:
                n_nat_pruned += len(pruned)
                pruned_details.append({
                    "clip_id": cid,
                    "pruned_events": pruned
                })

            total_events_nat_v80 += len(cleaned)
            out_rows.append({"clip_id": cid, "events": cleaned})

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
        "version": "v80",
        "base": "v76",
        "objective": "Retake Rank 1 from Hydra (1.6127)",
        "total_clips": len(out_rows),
        "synthetic_clips": n_syn,
        "natural_clips": n_nat,
        "natural_clips_with_pruned_duplicates": len(pruned_details),
        "total_natural_events_pruned": n_nat_pruned,
        "total_events_syn_v80": total_events_syn_v80,
        "total_events_nat_v76": total_events_nat_v76,
        "total_events_nat_v80": total_events_nat_v80,
        "total_events_v80": total_events_syn_v80 + total_events_nat_v80,
        "params": {
            "iou_thr": IOU_THR,
            "dur_ratio_min": DUR_RATIO
        },
        "oof_natural_validation": {
            "v76_baseline": {"comb": 1.17683, "f1": 0.46510, "dice": 0.71173},
            "v80_score":    {"comb": 1.17826, "f1": 0.46681, "dice": 0.71145},
            "delta_comb":   +0.00143,
            "delta_fold_A": +0.00220,
            "delta_fold_B": +0.00063
        }
    }
    with open(os.path.join(OUT_DIR, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("=" * 70, flush=True)
    print("v80 BUILD COMPLETE", flush=True)
    print(f"Total clips: {len(out_rows)} (Synthetic: {n_syn}, Natural: {n_nat})", flush=True)
    print(f"Synthetic events: {total_events_syn_v80} (100% byte-locked to v76)", flush=True)
    print(f"Natural events: v76={total_events_nat_v76} -> v80={total_events_nat_v80} (pruned {n_nat_pruned} shadow false alarms)", flush=True)
    print(f"Natural clips pruned: {len(pruned_details)} / {n_nat} ({len(pruned_details)/n_nat*100:.2f}%)", flush=True)
    print(f"Total submission events: {total_events_syn_v80 + total_events_nat_v80}", flush=True)
    print(f"Submission package saved to: {zip_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
