"""v93_s2 = v93 with synthetic events corrected by ENERGY-VERIFIED refnoise matches.

v93_s1 (cor >= 0.6, no audio check) scored 1.6139 (+0.0005): the template-subtraction energy profile shows
8 of its 23 corrected spans are truncated copies / false matches (subtracting the template's tail ADDS energy).
Validation calibration (src/t1_refnoise_energy_calib.py, cross-clip copies, true labels):
  cor 0.45-0.60 : energy pass (any frame >= 1 dB reduction, head/tail mean >= 0 dB) 24/24 agree; fail 1/33
  cor >= 0.60   : pass at head/tail >= -1 dB 1925/1926 agree (no failures exist on validation)
Rule: a refnoise span may replace an out-of-tolerance overlapping prediction, or be added where nothing is
predicted, only if it passes the energy test. Natural clips are byte-identical to the base.
    python src/build_v93_s2.py [base=submission/v93]
"""
import json
import os
import sys
import zipfile

import numpy as np
import soundfile as sf

ARGS = list(sys.argv)
sys.path.insert(0, os.path.dirname(__file__))
import t1_refnoise_extent_probe as X   # noqa: E402
from metrics import match_events   # noqa: E402

BASE = ARGS[1] if len(ARGS) > 1 else "submission/v93"
OUT = f"{BASE}_s2"
COR_MIN, HT_HI, HT_LO = 0.45, -1.0, 0.0      # head/tail thresholds for cor >= 0.6 / 0.45-0.6


def main():
    X.B.load_synth_templates()
    import refnoise_match as BM
    meta = list(BM.build_templates.meta)
    M = json.load(open("work/refnoise/test_pe.json"))
    syn = json.load(open("work/eval_syn_pred_fixed.json"))
    rer = set(json.load(open("submission/v70/info.json")).get("rerouted", []))
    rows = [json.loads(l) for l in open(f"{BASE}/predictions.jsonl") if l.strip()]
    n_rep = n_add = n_clip = n_rej = 0; log = []
    for r in rows:
        c = r["clip_id"]
        if not (syn.get(c, False) or c in rer):
            continue
        pr = [(e["onset"], e["offset"]) for e in r["events"]]
        cand = sorted(((m["cor"], m["onset"], m["onset"] + meta[m["k"]][2] - meta[m["k"]][1], m) for m in M.get(c, []) if m["cor"] >= COR_MIN),
                      key=lambda z: -z[0])
        rel = []
        for cr, a, b, m in cand:
            if all(min(b, y) - max(a, x) <= 0.5 * min(b - a, y - x) for x, y, _, _ in rel):
                rel.append((a, b, cr, m))
        if not rel:
            continue
        sp = [(a, b) for a, b, _, _ in rel]
        _, pairs = match_events(pr, sp)
        mp = {pi for pi, _ in pairs}; ms = {si for _, si in pairs}
        x = None; verdict = {}

        def ok(si):
            nonlocal x
            if si not in verdict:
                if x is None:
                    x, _ = sf.read(f"{X.TEST}/{c}.wav", dtype="float64"); x = x if x.ndim == 1 else x.mean(1)
                e = X.extent(x, rel[si][3], meta)
                if e is None:
                    verdict[si] = (False, None)
                else:
                    red = e[4]; ht = min(float(np.mean(red[:10])), float(np.mean(red[-10:])))
                    thr = HT_HI if rel[si][2] >= 0.6 else HT_LO
                    verdict[si] = (bool(np.any(red >= 1.0)) and ht >= thr, round(ht, 1))
            return verdict[si][0]

        new, changed, used = [], False, set(ms)
        for pi, (pa, pb) in enumerate(pr):
            if pi in mp:
                new.append((pa, pb)); continue
            ov = [si for si, (a, b) in enumerate(sp) if si not in used and min(b, pb) > max(a, pa)]
            good = [si for si in ov if ok(si)]
            n_rej += len(ov) - len(good)
            if good:
                for si in good:
                    new.append(sp[si]); used.add(si)
                n_rep += 1; changed = True
                log.append({"clip_id": c, "pred": [pa, pb], "spans": [[*sp[si], rel[si][2], verdict[si][1]] for si in good]})
            else:
                new.append((pa, pb))
        for si, (a, b) in enumerate(sp):
            if si not in used and not any(min(b, y) > max(a, x0) for x0, y in new):
                if ok(si):
                    new.append((a, b)); n_add += 1; changed = True
                    log.append({"clip_id": c, "added": [a, b, rel[si][2], verdict[si][1]]})
                else:
                    n_rej += 1
        if changed:
            n_clip += 1
            r["events"] = [{"onset": round(a, 6), "offset": round(b, 6)} for a, b in sorted(new)]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with zipfile.ZipFile(f"{OUT}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{OUT}/predictions.jsonl", "predictions.jsonl")
    json.dump({"base": BASE, "clips_changed": n_clip, "preds_replaced": n_rep, "spans_added": n_add,
               "candidates_rejected_by_energy": n_rej, "log": log}, open(f"{OUT}/info.json", "w"), indent=1)
    print(f"{OUT}: clips changed {n_clip} | predictions replaced {n_rep} | spans added {n_add} | "
          f"rejected by energy test {n_rej} | events {sum(len(r['events']) for r in rows)}")


if __name__ == "__main__":
    main()
