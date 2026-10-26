"""v93_s5 = v93_s4 + LOW-correlation (0.30-0.45) refnoise matches that pass a strict energy test.

Validation (src/t1_refnoise_lowcor_calib.py): at cor 0.30-0.45 a refnoise match marks an annotation only 0.6-10% of the
time, but with energy pass (any frame >= 1 dB, head/tail >= 0 dB) AND >= 30% of template frames explained:
natural 3/3, synthetic 12/13 agree. Same actions as s4 (snap within tolerance, replace overlapping
unmatched predictions, add). Spans overlapping an already-accepted (cor >= 0.45) span are skipped.
    python src/build_v93_s5.py
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

BASE, OUT = "submission/v93_s4", "submission/v93_s5"
LO, HI, FRAC = 0.30, 0.45, 0.3


def main():
    X.B.load_synth_templates()
    import refnoise_match as BM
    meta = list(BM.build_templates.meta)
    syn = json.load(open("work/eval_syn_pred_fixed.json"))
    MS, MN = json.load(open("work/refnoise/test_pe.json")), json.load(open("work/refnoise/testnat_pe.json"))
    rows = [json.loads(l) for l in open(f"{BASE}/predictions.jsonl") if l.strip()]
    stat = {"nat": {"snap": 0, "replace": 0, "add": 0}, "syn": {"snap": 0, "replace": 0, "add": 0}, "tested": 0, "passed": 0}
    log = []
    for r in rows:
        c = r["clip_id"]; kind = "syn" if syn.get(c, False) else "nat"
        allm = (MS if kind == "syn" else MN).get(c, [])
        hi_sp = [(m["onset"], m["offset"]) for m in allm if m["cor"] >= HI]
        cand = sorted((m for m in allm if LO <= m["cor"] < HI), key=lambda m: -m["cor"])
        if not cand:
            continue
        pr = [(e["onset"], e["offset"]) for e in r["events"]]
        x = None; acc = []
        for m in cand:
            a, b = m["onset"], m["offset"]
            if any(min(b, y) > max(a, z) for z, y in hi_sp + [(p, q) for p, q, _ in acc]):
                continue
            if match_events(pr, [(a, b)])[0] and any(abs(p - a) < 0.001 and abs(q - b) < 0.001 for p, q in pr):
                continue                                   # already exact -> nothing to change
            if x is None:
                x, _ = sf.read(f"{X.TEST}/{c}.wav", dtype="float64"); x = x if x.ndim == 1 else x.mean(1)
            stat["tested"] += 1
            e = X.extent(x, m, meta)
            if e is None:
                continue
            red = e[4]; ht = min(float(np.mean(red[:10])), float(np.mean(red[-10:])))
            if np.any(red >= 1.0) and ht >= 0.0 and float(np.mean(red >= 1.0)) >= FRAC:
                acc.append((a, b, m["cor"])); stat["passed"] += 1
        if not acc:
            continue
        sp = [(a, b) for a, b, _ in acc]
        _, pairs = match_events(pr, sp)
        new = list(pr); used = set(); changed = False
        for pi, si in pairs:
            if abs(pr[pi][0] - sp[si][0]) > 0.001 or abs(pr[pi][1] - sp[si][1]) > 0.001:
                new[pi] = sp[si]; stat[kind]["snap"] += 1; changed = True
                log.append({"clip_id": c, "kind": kind, "snap": [pr[pi], sp[si], acc[si][2]]})
            used.add(si)
        mp = {pi for pi, _ in pairs}; drop = set()
        for si, (a, b) in enumerate(sp):
            if si in used:
                continue
            ov = [pi for pi, (pa, pb) in enumerate(pr) if pi not in mp and pi not in drop and min(b, pb) > max(a, pa)]
            if ov:
                drop.update(ov); new.append((a, b)); stat[kind]["replace"] += 1; changed = True
                log.append({"clip_id": c, "kind": kind, "replace": [pr[pi] for pi in ov], "span": [a, b, acc[si][2]]})
            elif not any(min(b, pb) > max(a, pa) for pa, pb in pr):
                new.append((a, b)); stat[kind]["add"] += 1; changed = True
                log.append({"clip_id": c, "kind": kind, "add": [a, b, acc[si][2]]})
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
        print("  ", e["kind"], e["clip_id"][-32:], {k: v for k, v in e.items() if k not in ("clip_id", "kind")})


if __name__ == "__main__":
    main()
