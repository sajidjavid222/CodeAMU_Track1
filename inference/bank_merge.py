"""Merge noise-bank matches (src/bank_match.py) into synthetic-branch predictions.

A match is trusted when its normalised correlation >= th. For each predicted event:
  split  if >= 2 non-overlapping trusted matches (cor >= split_th) sit inside it and
         cover >= 60% of it, it was a merged event: emit the matches instead
  snap   otherwise take the boundaries of the best trusted match that overlaps it
         (gate="overlap": overlap > half the shorter one; gate="tol": both edges
         within max(tol_abs, tol_rel * duration) -- stops a long correct event being
         pulled onto a short crop of the same recording)
  keep   otherwise leave it as predicted (the bank may not hold that recording)
Then trusted matches with cor >= add_th that overlap nothing are added (misses).

`python src/bank_merge.py` scores a grid on validation, leave-one-clip-out, against
the v67 synthetic OOF predictions, per cross-fit fold.
"""
from __future__ import annotations

import json

# validated on LOO validation: synthetic OOF +0.0131 (fold A +0.0112, B +0.0149) over v67;
# th_ov 0.8 and 0.9 score identically -- 0.9 taken for match precision (98.9% exact)
V68 = dict(th=0.5, gate="hyb", tol_abs=0.15, tol_rel=0.3, th_ov=0.9, ratio=0.0)
# V69: for WHITENED matches (bank_match.py ... pe): pre-emphasis lifts match precision at
# cor>=0.5 from 65.8% to 98.4%, which makes adding unpredicted matches safe. LOO val
# synthetic 1.98842 vs V68 1.98686 (fold A +0.0003, B +0.0027); add_th plateau 0.6-0.8.
V69 = dict(V68, th=0.4, add_th=0.7)
# natural clips (merge_nat): honest natural OOF +0.009 (fold A +0.0056, B +0.0127); th 0.6-0.8 plateau
V70_NAT = dict(th=0.7, tol_abs=0.2, tol_rel=0.5, add=True)


def _ov(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])


def merge(pred, matches, th=0.6, gate="overlap", tol_abs=0.1, tol_rel=0.2, split_th=9.0, add_th=9.0,
          ratio=0.0, th_ov=None, drop_th=None, tol_ratio=0.0):
    th_ov = th if th_ov is None else th_ov
    if drop_th is not None:
        # a predicted event that overlaps no match with cor >= drop_th is treated as a false alarm
        pred = [p for p in pred if any(m["cor"] >= drop_th and _ov(p, (m["onset"], m["offset"])) > 0
                                       for m in matches)]
    M = [(m["onset"], m["offset"], m["cor"]) for m in matches if m["cor"] >= min(th, th_ov, split_th, add_th)]
    if not M:
        return [tuple(p) for p in pred]
    out = []
    for p in pred:
        p = tuple(p); d = p[1] - p[0]
        inside = sorted([m for m in M if m[2] >= split_th and _ov(p, m) >= 0.8 * (m[1] - m[0])], key=lambda m: -m[2])
        sel = []
        for m in inside:
            if all(_ov(m, q) <= 0 for q in sel):
                sel.append(m)
        if len(sel) >= 2 and sum(m[1] - m[0] for m in sel) >= 0.6 * d:
            out += [(m[0], m[1]) for m in sel]
            continue
        tol = max(tol_abs, tol_rel * d)
        # tol_ratio: a weaker match (< 0.9) that is much shorter than the prediction is usually
        # a crop of a longer recording -- don't let it shrink a correct event
        c_tol = [m for m in M if m[2] >= th and abs(m[0] - p[0]) <= tol and abs(m[1] - p[1]) <= tol
                 and (m[2] >= 0.9 or min(d, m[1] - m[0]) >= tol_ratio * max(d, m[1] - m[0]))]
        c_ov = [m for m in M if m[2] >= th_ov and _ov(p, m) > 0.5 * min(d, m[1] - m[0])
                and min(d, m[1] - m[0]) >= ratio * max(d, m[1] - m[0])]
        if gate == "tol":
            c_ov = []
        elif gate == "overlap":
            c_tol = []
        if c_tol:                                   # hyb: tolerance gate first
            m = max(c_tol, key=lambda m: m[2]); out.append((m[0], m[1]))
        elif c_ov:
            m = max(c_ov, key=lambda m: (_ov(p, m), m[2])); out.append((m[0], m[1]))
        else:
            out.append(p)
    for m in M:
        if m[2] >= add_th and all(_ov(m, q) <= 0 for q in out):
            out.append((m[0], m[1]))
    return sorted(set(out))


def merge_nat(pred, matches, th=0.8, tol_abs=0.1, tol_rel=0.3, add=True):
    """Natural clips: the bank was CUT from natural recordings at their labelled edges,
    so a trusted match is an exact natural event. Natural labels nest (a long host can
    contain a short event), so a match never shrinks a longer prediction that contains
    it: it replaces only a prediction whose both edges already agree within tolerance,
    otherwise it is added as its own event."""
    out = [tuple(p) for p in pred]
    for m in sorted((m for m in matches if m["cor"] >= th), key=lambda m: -m["cor"]):
        e = (m["onset"], m["offset"]); d = e[1] - e[0]; tol = max(tol_abs, tol_rel * d)
        close = [p for p in out if abs(p[0] - e[0]) <= tol and abs(p[1] - e[1]) <= tol]
        if close:
            out.remove(min(close, key=lambda p: abs(p[0] - e[0]) + abs(p[1] - e[1])))
            out.append(e)
        elif add:
            out.append(e)
    return sorted(set(out))


def natural_eval():
    """Honest natural OOF (v60 decode on ens_nat_val_oof_honest + cb_big/cb_s7), per fold,
    with and without bank matches from bank_match.py valnat pe."""
    import pickle
    import numpy as np
    from metrics import score_submission
    from postprocess import decode_frames, split_events, _peaks
    from dur_snap import dur_snap
    from boundary_regressor import apply_boundary_trim, FRAME
    fold, recs = {}, []
    for h in ("A", "B"):
        for l in open(f"work/val_hold{h}.jsonl"):
            if l.strip() and not json.loads(l).get("synthetic"):
                r = json.loads(l); fold[r["clip_id"]] = h; recs.append(r)
    refs = {r["clip_id"]: [(e["onset"], e["offset"]) for e in r["events"]] for r in recs}
    durs = {r["clip_id"]: r["duration"] for r in recs}
    base = dict(np.load("work/ens_nat_val_oof_honest.npz"))
    reg = pickle.load(open("work/boundary_regressor.pkl", "rb"))

    def load(n):
        z = np.load(f"work/ovprobs/{n}.npz"); p, o, f = {}, {}, {}
        for k in z.files:
            if k.endswith("||on"): o[k[:-4]] = z[k].astype(np.float64)
            elif k.endswith("||off"): f[k[:-5]] = z[k].astype(np.float64)
            elif not k.endswith("||cat"): p[k] = z[k].astype(np.float64)
        return p, o, f
    Mx = {n: load(n) for n in ("cb_big", "cb_s7")}
    pred = {}
    for c in refs:
        p = base[c].astype(np.float64); on = base[f"{c}||on"].astype(np.float64); off = base[f"{c}||off"].astype(np.float64)
        for nm, w in (("cb_big", 0.20), ("cb_s7", 0.15)):
            P_, O_, F_ = Mx[nm]
            if c in P_:
                k = min(len(p), len(P_[c]))
                p = (1 - w) * p[:k] + w * P_[c][:k]; on = (1 - w) * on[:k] + w * O_[c][:k]; off = (1 - w) * off[:k] + w * F_[c][:k]
        d = durs[c]
        e = decode_frames(p, FRAME, duration=d, high=0.57, low=0.48, median_k=3, min_dur=0.10)
        e = split_events(e, on, off, FRAME, split_thr=0.16, min_sub=0.10, max_pair_gap=0.25)
        e = dur_snap(e, on, off, FRAME, duration=d, split=0.45, snap_short=0.12, peak_short=0.10,
                     snap_long=0.32, peak_long=0.20)
        new = list(e)
        for a, b in e:
            if b - a >= 0.75:
                i0, i1 = int(round(a / FRAME)), int(round(b / FRAME))
                for s in _peaks(on, i0 + 1, i1 - 1, 0.20):
                    for f2 in _peaks(off, i0 + 1, i1 - 1, 0.20):
                        if 0 < (f2 - s) * FRAME <= 0.65:
                            new.append((s * FRAME, f2 * FRAME))
        pred[c] = [tuple(x) for x in apply_boundary_trim(sorted(set(new)), p, on, off, d, reg,
                                                          min_shift=-0.20, max_shift=-0.005)]
    bank = json.load(open("work/bank/valnat_pe.json"))

    def score(out):
        r = {}
        for h in ("A", "B"):
            cs = [c for c in refs if fold[c] == h]
            r[h] = score_submission({c: out[c] for c in cs}, {c: refs[c] for c in cs}, {c: durs[c] for c in cs})['combined']
        return r, score_submission(out, refs, durs)
    b, bs = score(pred)
    print(f"natural honest OOF (v60): {bs['combined']:.5f} F1 {bs['event_f1']:.5f} Dice {bs['segment_dice']:.5f}")
    for th in (0.6, 0.7, 0.8, 0.9):
        for ta, tr in ((0.1, 0.3), (0.2, 0.5)):
            for add in (False, True):
                out = {c: merge_nat(pred[c], bank.get(c, []), th, ta, tr, add) for c in refs}
                r, s = score(out)
                print(f"  th {th} tol {ta}/{tr} add {add!s:5}: A {r['A'] - b['A']:+.5f} B {r['B'] - b['B']:+.5f} | "
                      f"{s['combined']:.5f} F1 {s['event_f1']:.5f} Dice {s['segment_dice']:.5f}", flush=True)


if __name__ == "__main__":
    import sys
    if "nat" in sys.argv[1:]:
        sys.path.insert(0, "src")
        natural_eval()
        sys.exit()
    ONLY_V68 = "v68" in sys.argv[1:]
    ERRORS = "errors" in sys.argv[1:]
    PEB = "pe" in sys.argv[1:]            # use the pre-emphasis (whitened) matches
    sys.argv_full = list(sys.argv)
    sys.argv = sys.argv[:1]
    import syn_fold_scan as S                     # loads v67 synthetic OOF tracks + refs
    from postprocess import decode_frames, split_events
    from dur_snap import dur_snap
    from metrics import score_submission

    P = S.V67
    pred = {}
    for c in S.refs:
        p, on, off = S.TR[c]; d = S.durs[c]
        e = decode_frames(p, S.FRAME, duration=d, high=P['high'], low=P['low'], median_k=P['median_k'],
                          gap_sec=P['gap_sec'], min_dur=P['min_dur'], expand_sec=P['expand_sec'])
        e = split_events(e, on, off, S.FRAME, split_thr=P['st'], min_sub=P['ms'], max_pair_gap=P['gp'])
        pred[c] = [tuple(x) for x in dur_snap(e, on, off, S.FRAME, duration=d, split=P['split'], snap_short=P['ss'],
                                              peak_short=P['ps'], snap_long=P['sl'], peak_long=P['pl'])]
    bank = json.load(open("work/bank/val_pe.json" if PEB else "work/bank/val.json"))
    n_ref = sum(len(v) for v in S.refs.values())
    print(f"bank file: {'val_pe.json (whitened)' if PEB else 'val.json'}")
    for q in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        n_m = n_ok = hit_r = 0
        for c, R in S.refs.items():
            ms = [m for m in bank.get(c, []) if m["cor"] >= q]
            n_m += len(ms)
            n_ok += sum(1 for m in ms if any(abs(m["onset"] - a) <= 0.01 and abs(m["offset"] - b) <= 0.01 for a, b in R))
            hit_r += sum(1 for a, b in R if any(abs(m["onset"] - a) <= 0.01 and abs(m["offset"] - b) <= 0.01 for m in ms))
        print(f"  cor>={q}: {n_m:5d} matches, {100 * n_ok / max(n_m, 1):5.1f}% exact | events recovered {100 * hit_r / n_ref:5.1f}%")

    def score(out):
        r = {}
        for h in ("A", "B"):
            cs = [c for c in S.refs if S.fold[c] == h]
            r[h] = score_submission({c: out[c] for c in cs}, {c: S.refs[c] for c in cs},
                                    {c: S.durs[c] for c in cs})['combined']
        return r, score_submission(out, S.refs, S.durs)

    b, bs = score(pred)
    print(f"v67 synthetic OOF: {bs['combined']:.5f} (F1 {bs['event_f1']:.5f} Dice {bs['segment_dice']:.5f})")
    if ERRORS:
        def hit(e, R):
            return any(abs(e[0] - a) <= max(0.2 * (b - a), 0.05) and abs(e[1] - b) <= max(0.2 * (b - a), 0.05)
                       for a, b in R)
        out = {c: merge(pred[c], bank.get(c, []), **V68) for c in S.refs}
        fp_m = fp_u = fn_m = fn_u = tp = 0
        best_fn = []
        for c, R in S.refs.items():
            P0 = set(pred[c]); ms = bank.get(c, [])
            for e in out[c]:
                if hit(e, R): tp += 1
                elif e in P0: fp_u += 1          # unmoved prediction, not a real event
                else: fp_m += 1                  # moved onto a match, still wrong
            for ra, rb in R:
                if not hit((ra, rb), out[c]):
                    cm = max([m["cor"] for m in ms if _ov((ra, rb), (m["onset"], m["offset"])) > 0] or [0])
                    best_fn.append(cm)
                    if any(abs(m["onset"] - ra) <= 0.01 and abs(m["offset"] - rb) <= 0.01 for m in ms): fn_m += 1
                    else: fn_u += 1
        print(f"after V68 on validation: TP {tp} | FP moved {fp_m}, FP unmoved {fp_u} | "
              f"FN with an exact bank match {fn_m}, FN without {fn_u}")
        print("  best overlapping match cor for each FN:", sorted(round(x, 2) for x in best_fn))
        if "detail" in sys.argv_full:
            for c, R in S.refs.items():
                errs = [("FP", e) for e in out[c] if not hit(e, R)] + [("FN", r) for r in R if not hit(r, out[c])]
                if not errs:
                    continue
                print(f"\n  {c[-30:]}  dur {S.durs[c]:.2f}")
                print("    ref :", [(round(a, 3), round(b, 3)) for a, b in sorted(R)])
                print("    pred:", [(round(a, 3), round(b, 3)) for a, b in sorted(pred[c])])
                print("    v68 :", [(round(a, 3), round(b, 3)) for a, b in sorted(out[c])])
                for k, (ea, eb) in errs:
                    near = sorted([m for m in bank.get(c, []) if _ov((ea, eb), (m["onset"], m["offset"])) > -0.05],
                                  key=lambda m: -m["cor"])[:3]
                    print(f"    {k} ({ea:.3f},{eb:.3f}) matches:",
                          [(round(m['onset'], 3), round(m['offset'], 3), m['cor']) for m in near])
        unm = [(c, p) for c in S.refs for p in out[c] if p in set(pred[c])]
        print(f"  unmoved predictions: {len(unm)}, of which real events {sum(1 for c, p in unm if hit(p, S.refs[c]))}")
        for dt in (0.2, 0.3, 0.4, 0.5):
            o2 = {c: merge(pred[c], bank.get(c, []), **dict(V68, drop_th=dt)) for c in S.refs}
            r, s = score(o2)
            print(f"  + drop unmatched (cor<{dt}): A {r['A'] - b['A']:+.5f} B {r['B'] - b['B']:+.5f} | "
                  f"{s['combined']:.5f} F1 {s['event_f1']:.5f} Dice {s['segment_dice']:.5f}")
        sys.exit()
    if ONLY_V68:
        out = {c: merge(pred[c], bank.get(c, []), **V68) for c in S.refs}
        r, s = score(out)
        n = sum(len(v) for v in pred.values())
        mv = sum(1 for c in S.refs for e in out[c] if e not in set(pred[c]))
        print(f"V68 {V68}: A {r['A'] - b['A']:+.5f} B {r['B'] - b['B']:+.5f} | {s['combined']:.5f} "
              f"F1 {s['event_f1']:.5f} Dice {s['segment_dice']:.5f} | moved {mv}/{n} ({100 * mv / n:.1f}%)")
        sys.exit()
    grid = [dict(V68)]
    grid += [dict(V68, th=t, add_th=a) for t in (0.4, 0.5) for a in (0.6, 0.7, 0.8)]
    grid += [dict(V68, th=0.4, add_th=0.7, th_ov=o) for o in (0.7, 0.8)]
    for cf in grid:
        out = {c: merge(pred[c], bank.get(c, []), **cf) for c in S.refs}
        r, s = score(out)
        print(f"  {json.dumps(cf):92} A {r['A'] - b['A']:+.5f} B {r['B'] - b['B']:+.5f} | "
              f"{s['combined']:.5f} F1 {s['event_f1']:.5f} Dice {s['segment_dice']:.5f}", flush=True)
