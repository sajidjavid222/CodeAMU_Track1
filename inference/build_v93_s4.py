"""v93_s4 = v93_s3 + energy-verified synthetic-refnoise matches applied to NATURAL clips.

The organisers' synthetic reference noise was cut from natural recordings, so some natural test clips contain the
exact same noise waveform (work/refnoise/testnat_pe.json). Validation natural calibration (src/t1_natrefnoise_calib.py):
span == human annotation  cor>=0.9: 96% | 0.7-0.9: 94% (energy pass 13/13) | 0.6-0.7: 43% (energy pass ht>=-1:
5/6) | 0.45-0.6: 8% (energy pass ht>=0: 8/8, fail 2%). v93's natural branch merged cor>=0.7 matches, but its
later rules moved/pruned some. Accepted spans: cor>=0.7 always; 0.6-0.7 energy pass (ht>=-1); 0.45-0.6 energy
pass (ht>=0). Actions per accepted span: snap a within-tolerance prediction exactly onto it; else replace
overlapping predictions that match nothing; else add it. Synthetic clips are byte-identical to s3.
    python src/build_v93_s4.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import t1_refnoise_extent_probe as X   # noqa: E402
from metrics import match_events   # noqa: E402

BASE, OUT = "submission/v93_s3", "submission/v93_s4"


def main():
    X.B.load_synth_templates()
    import refnoise_match as BM
    meta = list(BM.build_templates.meta)
    MN = json.load(open("work/refnoise/testnat_pe.json"))
    syn = json.load(open("work/eval_syn_pred_fixed.json"))
    rer = set(json.load(open("submission/v70/info.json")).get("rerouted", []))
    rows = [json.loads(l) for l in open(f"{BASE}/predictions.jsonl") if l.strip()]
    stat = {"snap": 0, "replace": 0, "add": 0, "rejected": 0, "accepted": {}}
    log = []
    for r in rows:
        c = r["clip_id"]
        if syn.get(c, False) or c in rer:
            continue
        cand = sorted((m for m in MN.get(c, []) if m["cor"] >= 0.45), key=lambda m: -m["cor"])
        if not cand:
            continue
        x = None; acc = []
        for m in cand:
            a, b = m["onset"], m["offset"]
            if any(min(b, y) - max(a, z) > 0.5 * min(b - a, y - z) for z, y, _ in acc):
                continue
            if m["cor"] < 0.7:
                if x is None:
                    x, _ = sf.read(f"{X.TEST}/{c}.wav", dtype="float64"); x = x if x.ndim == 1 else x.mean(1)
                e = X.extent(x, m, meta)
                if e is None:
                    stat["rejected"] += 1; continue
                red = e[4]; ht = min(float(np.mean(red[:10])), float(np.mean(red[-10:])))
                if not (np.any(red >= 1.0) and ht >= (-1.0 if m["cor"] >= 0.6 else 0.0)):
                    stat["rejected"] += 1; continue
            band = ">=0.7" if m["cor"] >= 0.7 else "0.6-0.7" if m["cor"] >= 0.6 else "0.45-0.6"
            stat["accepted"][band] = stat["accepted"].get(band, 0) + 1
            acc.append((a, b, m["cor"]))
        if not acc:
            continue
        pr = [(e["onset"], e["offset"]) for e in r["events"]]
        sp = [(a, b) for a, b, _ in acc]
        _, pairs = match_events(pr, sp)
        new = list(pr); used = set(); changed = False
        for pi, si in pairs:                                  # exact snap (Dice), F1 unchanged
            if abs(pr[pi][0] - sp[si][0]) > 0.001 or abs(pr[pi][1] - sp[si][1]) > 0.001:
                new[pi] = sp[si]; stat["snap"] += 1; changed = True
            used.add(si)
        matched_p = {pi for pi, _ in pairs}
        drop = set()
        for si, (a, b) in enumerate(sp):
            if si in used:
                continue
            ov = [pi for pi, (pa, pb) in enumerate(pr) if pi not in matched_p and pi not in drop and min(b, pb) > max(a, pa)]
            if ov:
                drop.update(ov); new.append((a, b)); stat["replace"] += 1; changed = True
                log.append({"clip_id": c, "replace": [pr[pi] for pi in ov], "span": [a, b, acc[si][2]]})
            elif not any(min(b, pb) > max(a, pa) for pa, pb in pr):
                new.append((a, b)); stat["add"] += 1; changed = True
                log.append({"clip_id": c, "add": [a, b, acc[si][2]]})
        if changed:
            new = [e for i, e in enumerate(new) if i not in drop]
            r["events"] = [{"onset": round(a, 6), "offset": round(b, 6)} for a, b in sorted(set(new))]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with zipfile.ZipFile(f"{OUT}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{OUT}/predictions.jsonl", "predictions.jsonl")
    json.dump({"base": BASE, **stat, "log": log}, open(f"{OUT}/info.json", "w"), indent=1, default=str)
    print(f"{OUT}: {stat} | events {sum(len(r['events']) for r in rows)}")
    for e in log:
        print("  ", e["clip_id"][-34:], {k: v for k, v in e.items() if k != "clip_id"})


if __name__ == "__main__":
    main()
