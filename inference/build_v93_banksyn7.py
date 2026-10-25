"""v93_banksyn7 = v93_banksyn5 + one fill event in natural clips that are still EMPTY.

Segment Dice is macro per clip: an empty natural clip scores 0 although it almost surely holds noise (OOF:
29/29 empty clips have references, covering a median 13% of the clip). The fill spans the natural ensemble's
activity envelope [first frame p >= TH, last frame p >= TH] (fallback: clip minus 5% edges). Honest OOF (folds
A/B, v97 pipeline): TH 0.20 -> A +0.0017 / B +0.0039 (whole clip +0.0006 / +0.0036; TH 0.5 negative).
    python src/build_v93_banksyn7.py [th=0.20]
"""
import json
import os
import sys
import zipfile

import numpy as np

TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
BASE, OUT = "submission/v93_banksyn5", "submission/v93_banksyn7"
FRAME = 0.02


def main():
    syn = json.load(open("work/eval_syn_pred_fixed.json"))
    rer = set(json.load(open("submission/v70/info.json")).get("rerouted", []))
    nat = dict(np.load("work/ens_nat_eval_v60.npz"))
    rows = [json.loads(l) for l in open(f"{BASE}/predictions.jsonl") if l.strip()]
    log = []
    for r in rows:
        c = r["clip_id"]
        if r["events"] or syn.get(c, False) or c in rer:
            continue
        p = nat[c].astype(np.float64); d = len(p) * FRAME
        i = np.flatnonzero(p >= TH)
        a, b = (i[0] * FRAME, min(d, (i[-1] + 1) * FRAME)) if len(i) else (0.05 * d, 0.95 * d)
        r["events"] = [{"onset": round(float(a), 4), "offset": round(float(b), 4)}]
        log.append({"clip_id": c, "fill": [round(float(a), 3), round(float(b), 3)], "dur": round(d, 2), "pmax": round(float(p.max()), 3)})
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with zipfile.ZipFile(f"{OUT}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{OUT}/predictions.jsonl", "predictions.jsonl")
    json.dump({"base": BASE, "th": TH, "filled": len(log), "log": log}, open(f"{OUT}/info.json", "w"), indent=1)
    span = [ (e["fill"][1] - e["fill"][0]) / e["dur"] for e in log]
    print(f"{OUT}: filled {len(log)} empty natural clips | mean span {np.mean(span):.2f} of clip | events {sum(len(r['events']) for r in rows)}")


if __name__ == "__main__":
    main()
