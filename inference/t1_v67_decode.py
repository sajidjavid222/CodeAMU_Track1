"""Track 1 v67 decode (stages 2-6 of notebooks/v67_pipeline.ipynb) for any audio set.

Stage 1 (per-model frame probabilities, src/infer.py) must already have written
<probs>/<model>.npz for all 24 v67 checkpoints. This script routes, blends, decodes and writes
<out>/predictions.jsonl + submission.zip. Same constants as the notebook, which reproduces the
submitted v67 (1.6001) bit-for-bit on the Phase 1 test set.

    python src/t1_v67_decode.py <audio_dir> <clip_ids.txt> <probs_dir> <out_dir>
"""
import json
import os
import pickle
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
AUDIO, CLIPS, PROBS, OUT = sys.argv[1:5]
MATCH_SUBMITTED_ROUNDING = True

NAT16_PRESENCE = {"v8_whis_lg_gold": 1.3, "ov_med_syn": 0.8, "v2_crnn_gold": 1.0, "v9_lg3_s21": 1.3,
                  "v4_wavlm_gold": 1.0, "bg_sml_syn": 1.0, "gv_lg3_A": 1.3, "gv_lg3_B": 1.3,
                  "gv_med_A": 0.8, "gv_med_B": 0.8, "gv_sml_A": 1.0, "gv_sml_B": 1.0,
                  "rb_beats_gold": 1.4, "v9_lg2_gold": 0.4, "hr_med_syn": 0.0, "hr_med_syn_gold": 0.0}
NAT16_BOUNDARY = {"v8_whis_lg_gold": 1.3, "ov_med_syn": 0.8, "v2_crnn_gold": 1.0, "v9_lg3_s21": 1.3,
                  "v4_wavlm_gold": 1.0, "bg_sml_syn": 1.0, "gv_lg3_A": 1.3, "gv_lg3_B": 1.3,
                  "gv_med_A": 0.8, "gv_med_B": 0.8, "gv_sml_A": 1.0, "gv_sml_B": 1.0,
                  "rb_beats_gold": 0.0, "v9_lg2_gold": 0.8, "hr_med_syn": 0.5, "hr_med_syn_gold": 0.2}
FULL_GOLD = ["lg3_full_gold", "lg2_full_gold", "beats_full_gold"]
CB_BIG, CB_BIG_W = "cb_big", 0.20
CB_S7, CB_S7_W = "cb_s7", 0.15
SYN_MEMBERS = ["bg_lg3_syn", "bg_lg3_syn_gold", "cb_s7", "cb_s7_gold", "gv_lg3_A", "gv_lg3_B",
               "gv_med_A", "gv_med_B", "gv_sml_A", "gv_sml_B"]
ALL_MODELS = sorted(set(NAT16_PRESENCE) | set(FULL_GOLD) | {CB_BIG, CB_S7} | set(SYN_MEMBERS))


def load_tracks(model, clips):
    z = np.load(f"{PROBS}/{model}.npz")
    out = {}
    for c in clips:
        p = z[c].astype(np.float64); on = z[f"{c}||on"].astype(np.float64); off = z[f"{c}||off"].astype(np.float64)
        if model.startswith("hr_"):                       # 10 ms -> 20 ms
            L = 2 * (len(p) // 2); p = (p[:L:2] + p[1:L:2]) / 2
            L = 2 * (len(on) // 2); on = np.maximum(on[:L:2], on[1:L:2])
            L = 2 * (len(off) // 2); off = np.maximum(off[:L:2], off[1:L:2])
        out[c] = (p, on, off)
    return out


def weighted(tracks, weights, i):
    ms = [m for m, w in weights.items() if w > 0]
    tot = sum(weights[m] for m in ms)
    k = min(len(tracks[m][i]) for m in ms)
    return sum(tracks[m][i][:k] * (weights[m] / tot) for m in ms)


def stored(x, dtype):
    return x.astype(dtype).astype(np.float64) if MATCH_SUBMITTED_ROUNDING else x


def main():
    import joblib
    from boundary_regressor import FRAME, apply_boundary_trim
    from dur_snap import dur_snap
    from metrics import write_jsonl_events
    from postprocess import _peaks, decode_frames, split_events
    from synth_clf import build_matrix

    order = [l.strip() for l in open(CLIPS) if l.strip()]
    missing = [m for m in ALL_MODELS if not os.path.exists(f"{PROBS}/{m}.npz")]
    assert not missing, f"missing probability caches: {missing}"

    # routing: T1_ROUTE_JSON overrides the audio router with a precomputed {clip_id: bool} map
    route_json = os.environ.get("T1_ROUTE_JSON")
    if route_json:
        loaded = json.load(open(route_json))
        is_syn = {c: bool(loaded[c]) for c in order}
        print(f"routing loaded from {route_json}", flush=True)
    else:
        X = build_matrix([f"{AUDIO}/{c}.wav" for c in order], workers=8)
        p_syn = joblib.load("work/synth_clf.pkl").predict_proba(X)[:, 1]
        np.save("work/eval_synclf_proba.npy", p_syn)
        is_syn = {c: bool(p >= 0.5) for c, p in zip(order, p_syn)}
    nat_clips = [c for c in order if not is_syn[c]]
    syn_clips = [c for c in order if is_syn[c]]
    print(f"{len(order)} clips | natural {len(nat_clips)} | synthetic {len(syn_clips)}", flush=True)

    # natural ensemble
    T = {m: load_tracks(m, nat_clips) for m in NAT16_PRESENCE}
    GOLD = {m: load_tracks(m, nat_clips) for m in FULL_GOLD}
    CB, CB7 = load_tracks(CB_BIG, nat_clips), load_tracks(CB_S7, nat_clips)
    NAT = {}
    for c in nat_clips:
        per = {m: T[m][c] for m in T}
        tr = [weighted(per, NAT16_PRESENCE, 0), weighted(per, NAT16_BOUNDARY, 1), weighted(per, NAT16_BOUNDARY, 2)]
        tr = [stored(x, np.float32) for x in tr]
        for i in range(3):
            q = [tr[i]] + [GOLD[g][c][i] for g in FULL_GOLD]
            k = min(len(x) for x in q)
            v = stored((16 * q[0][:k] + sum(x[:k] for x in q[1:])) / 19, np.float16)
            cb = CB[c][i]; k = min(len(v), len(cb))
            v = (1 - CB_BIG_W) * v[:k] + CB_BIG_W * cb[:k]
            c7 = CB7[c][i]; k = min(len(v), len(c7))
            tr[i] = stored((1 - CB_S7_W) * v[:k] + CB_S7_W * c7[:k], np.float16)
        NAT[c] = tuple(tr)

    # natural decode
    reg = pickle.load(open("work/boundary_regressor.pkl", "rb"))
    preds = {}
    for c in nat_clips:
        p, on, off = NAT[c]
        evs = decode_frames(p, FRAME, duration=None, high=0.57, low=0.48, median_k=3, min_dur=0.10)
        evs = split_events(evs, on, off, FRAME, split_thr=0.16, min_sub=0.10, max_pair_gap=0.25)
        evs = dur_snap(evs, on, off, FRAME, duration=None, split=0.45,
                       snap_short=0.12, peak_short=0.10, snap_long=0.32, peak_long=0.20)
        new = list(evs)
        for a, b in evs:                                   # interior carving
            if b - a >= 0.75:
                i0, i1 = int(round(a / FRAME)), int(round(b / FRAME))
                for s in _peaks(on, i0 + 1, i1 - 1, 0.20):
                    for e in _peaks(off, i0 + 1, i1 - 1, 0.20):
                        if 0 < (e - s) * FRAME <= 0.65:
                            new.append((s * FRAME, e * FRAME))
        preds[c] = apply_boundary_trim(sorted(set(new)), p, on, off, None, reg, min_shift=-0.20, max_shift=-0.005)

    # synthetic branch
    syn_params = json.load(open("work/route_v43.json"))["syn_params"]
    dec = {k: v for k, v in syn_params.items() if k not in ("snap_win", "min_peak")}
    S = [load_tracks(m, syn_clips) for m in SYN_MEMBERS]
    for c in syn_clips:
        k = min(len(s[c][0]) for s in S)
        p, on, off = (np.mean([s[c][i][:k] for s in S], axis=0) for i in range(3))
        evs = decode_frames(p, FRAME, duration=None, **dec)
        evs = split_events(evs, on, off, FRAME, split_thr=0.06, min_sub=0.10, max_pair_gap=0.40)
        preds[c] = dur_snap(evs, on, off, FRAME, duration=None, split=0.45,
                            snap_short=0.12, peak_short=0.10, snap_long=0.22, peak_long=0.20)

    for c in order:
        preds.setdefault(c, [])
    # artifacts read by the later post-processing steps (inference/build_v*.py): the routing and the natural
    # ensemble tracks (20 ms p / onset / offset, float16), under the file names those steps expect
    os.makedirs("work", exist_ok=True)
    json.dump({c: bool(is_syn[c]) for c in order}, open("work/eval_syn_pred_fixed.json", "w"))
    np.savez_compressed("work/ens_nat_eval_v60.npz", **{k: v.astype(np.float16) for c, (p, on, off) in NAT.items()
                                                         for k, v in ((c, p), (f"{c}||on", on), (f"{c}||off", off))})
    os.makedirs(OUT, exist_ok=True)
    jl = os.path.join(OUT, "predictions.jsonl")
    write_jsonl_events(jl, preds, clip_order=order)
    with zipfile.ZipFile(os.path.join(OUT, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jl, "predictions.jsonl")
    print(f"clips {len(preds)} | events {sum(len(v) for v in preds.values())} | wrote {OUT}/submission.zip")


if __name__ == "__main__":
    main()
