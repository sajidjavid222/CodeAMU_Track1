"""Train and serialize the final boundary regressor for IndoML 2026 Track 1.

Trains on all matched candidate events from natural validation clips (Fold A + Fold B),
using Huber loss gradient boosting to robustly fit offset shifts.
"""
from __future__ import annotations
import json, os, sys, pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, "src")
from metrics import score_submission, match_events
from postprocess import decode_frames, split_events, _peaks
from dur_snap import dur_snap
from boundary_regressor import extract_event_features, apply_boundary_trim, FRAME

OVDIR = "work/ovprobs"

def load_npz(directory: str, name: str):
    path = os.path.join(directory, f"{name}.npz")
    if not os.path.exists(path): return None
    z = np.load(path)
    pres, on, off = {}, {}, {}
    for k in z.files:
        if k.endswith("||on"): on[k[:-4]] = z[k].astype(np.float64)
        elif k.endswith("||off"): off[k[:-5]] = z[k].astype(np.float64)
        elif not k.endswith("||cat"): pres[k] = z[k].astype(np.float64)
    if name.startswith("hr_"):
        for k in list(pres.keys()):
            p = pres[k]
            L = 2 * (len(p) // 2)
            pres[k] = (p[:L:2] + p[1:L:2]) / 2.0
        for k in list(on.keys()):
            o = on[k]
            Lo = 2 * (len(o) // 2)
            on[k] = np.maximum(o[:Lo:2], o[1:Lo:2])
        for k in list(off.keys()):
            f = off[k]
            Lf = 2 * (len(f) // 2)
            off[k] = np.maximum(f[:Lf:2], f[1:Lf:2])
    return pres, on, off

def main():
    print("1. Loading ground truth and route configuration...", flush=True)
    recs_a = [json.loads(l) for l in open("work/val_holdA.jsonl") if l.strip() and not json.loads(l).get("synthetic")]
    recs_b = [json.loads(l) for l in open("work/val_holdB.jsonl") if l.strip() and not json.loads(l).get("synthetic")]
    refs, durs = {}, {}
    for r in recs_a + recs_b:
        refs[r["clip_id"]] = [(e["onset"], e["offset"]) for e in r["events"]]
        durs[r["clip_id"]] = r["duration"]

    route = json.load(open("work/route_v31.json"))
    nat_members = route["nat_members"]
    wp = dict(route["nat_weights_presence"])
    wb = dict(route["nat_weights_boundary"])

    M = {m: load_npz(OVDIR, m) for m in nat_members}

    print("2. Generating baseline candidates with tuned embedded extraction...", flush=True)
    preds = {}
    ens_data = {}
    fold_map = {}

    # Embedded extraction parameters matching v31 champion
    min_host = 0.8
    t_on = 0.20
    t_off = 0.20
    max_cand = 0.6
    max_f = int(max_cand / FRAME)
    min_f = int(0.08 / FRAME)

    for fold_name, recs in [("A", recs_a), ("B", recs_b)]:
        active = [m for m in nat_members if not (m.startswith("gv_") and not m.endswith(f"_{fold_name}"))]
        nw_p = [wp[m]/sum(wp[x] for x in active) for m in active]
        nw_b = [wb[m]/sum(wb[x] for x in active) for m in active]
        for r in recs:
            cid = r["clip_id"]
            fold_map[cid] = fold_name
            dur = durs[cid]
            p_list = [M[m][0][cid] for m in active]
            on_list = [M[m][1][cid] for m in active]
            off_list = [M[m][2][cid] for m in active]
            k = min(len(p) for p in p_list)
            ko = min(len(o) for o in on_list)
            kf = min(len(f) for f in off_list)
            p_ens = np.sum([p[:k] * x for p, x in zip(p_list, nw_p)], axis=0)
            on_ens = np.sum([o[:ko] * x for o, x in zip(on_list, nw_b)], axis=0)
            off_ens = np.sum([f[:kf] * x for f, x in zip(off_list, nw_b)], axis=0)
            ens_data[cid] = (p_ens, on_ens, off_ens, dur)

            evs = decode_frames(p_ens, FRAME, duration=dur, **{k:v for k,v in route["nat_params"].items() if k not in ("snap_win", "min_peak")})
            if route.get("nat_split"): evs = split_events(evs, on_ens, off_ens, FRAME, **route["nat_split"])
            if route.get("nat_dursnap"): evs = dur_snap(evs, on_ens, off_ens, FRAME, duration=dur, **route["nat_dursnap"])

            new_emb = []
            for ev_on, ev_off in list(evs):
                if ev_off - ev_on >= min_host:
                    i0, i1 = int(ev_on / FRAME), int(ev_off / FRAME)
                    ons = _peaks(on_ens, i0 + 2, i1 - 2, t_on)
                    offs = _peaks(off_ens, i0 + 2, i1 - 2, t_off)
                    for o in ons:
                        cands = [f for f in offs if o + min_f <= f <= o + max_f]
                        if cands:
                            chosen = cands[0]
                            score = on_ens[o] * off_ens[chosen]
                            new_emb.append((o * FRAME, chosen * FRAME, score))
            new_emb.sort(key=lambda x: x[2], reverse=True)
            kept_emb = []
            for e_on, e_off, sc in new_emb:
                if not any(not (e_off <= k_on or e_on >= k_off) for k_on, k_off in kept_emb):
                    kept_emb.append((round(float(e_on), 3), round(float(e_off), 3)))
            evs = list(evs) + kept_emb
            evs.sort(key=lambda x: (x[0], x[1]))
            preds[cid] = evs

    sc_cand = score_submission(preds, refs, durs)
    print(f"Candidates before boundary trim: Comb={sc_cand['combined']:.5f} (F1={sc_cand['event_f1']:.5f}, Dice={sc_cand['segment_dice']:.5f}) TP={sc_cand['tp']}")

    print("3. Building dataset of candidate matches...", flush=True)
    all_features = []
    all_shifts = []

    for cid, evs in preds.items():
        p_ens, on_ens, off_ens, dur = ens_data[cid]
        r_list = refs.get(cid, [])
        for pi, ev in enumerate(evs):
            p_on, p_off = ev
            best_r = None
            best_iou = 0.0
            for ri, (r_on, r_off) in enumerate(r_list):
                inter = max(0.0, min(p_off, r_off) - max(p_on, r_on))
                union = max(p_off, r_off) - min(p_on, r_on)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_r = (r_on, r_off)

            if best_r is not None and best_iou >= 0.35:
                r_on, r_off = best_r
                shift_off = float(np.clip(r_off - p_off, -0.30, 0.15))
                feat = extract_event_features(ev, pi, evs, p_ens, on_ens, off_ens, dur)
                all_features.append(feat)
                all_shifts.append(shift_off)

    print(f"Total well-matched candidate pairs for training: {len(all_features)}")

    print("4. Training final Huber Gradient Boosting Regressor...", flush=True)
    X = np.array(all_features)
    y = np.array(all_shifts)

    reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.04, loss="huber", random_state=42)
    reg.fit(X, y)

    # Serialize trained model
    out_model = "work/boundary_regressor.pkl"
    with open(out_model, "wb") as f:
        pickle.dump(reg, f)
    print(f"Saved serialized boundary regressor to {out_model} ({os.path.getsize(out_model)} bytes)")

    print("5. Evaluating full OOF predictions with boundary trimmer...", flush=True)
    adj_preds = {}
    for cid, evs in preds.items():
        p_ens, on_ens, off_ens, dur = ens_data[cid]
        adj_preds[cid] = apply_boundary_trim(evs, p_ens, on_ens, off_ens, dur, reg,
                                             min_shift=-0.10, max_shift=-0.02)

    sc_final = score_submission(adj_preds, refs, durs)
    print("==========================================")
    print(f"Final OOF Combined Score: {sc_final['combined']:.5f} (F1={sc_final['event_f1']:.5f}, Dice={sc_final['segment_dice']:.5f})")
    print(f"TP={sc_final['tp']} (gain: {sc_final['tp'] - sc_cand['tp']:+d}), FP={sc_final['fp']}, FN={sc_final['fn']}")
    print("==========================================")

if __name__ == "__main__":
    main()
