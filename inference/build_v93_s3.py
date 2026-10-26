"""v93_s3 = v93_s2 minus synthetic predictions that no refnoise copy supports AND where the Track 2
enhancer finds no noise (src/t1_syn_fp_probe.py -> work/t1_syn_unsupported_D.json).

Noise meter (removed-energy share D, dB): refnoise-span regions median -2 (4% below -12), noise-free speech median
-64 (97% below -12). Unsupported predictions are bimodal: ~28 below -12 dB (speech only -> false alarms),
~40 near 0 dB (real noises absent from the refnoise -> kept). Never removes a clip's last event.
    python src/build_v93_s3.py [thr_db=-12]
"""
import json
import os
import sys
import zipfile

THR = float(sys.argv[1]) if len(sys.argv) > 1 else -12.0
BASE, OUT = "submission/v93_s2", "submission/v93_s3" if len(sys.argv) <= 1 else f"submission/v93_s3_{abs(int(THR))}"

drop = {(d["clip_id"], round(d["onset"], 6), round(d["offset"], 6)) for d in json.load(open("work/t1_syn_unsupported_D.json")) if d["D"] < THR}
rows = [json.loads(l) for l in open(f"{BASE}/predictions.jsonl") if l.strip()]
n_drop = n_guard = 0; log = []
for r in rows:
    keep = [e for e in r["events"] if (r["clip_id"], round(e["onset"], 6), round(e["offset"], 6)) not in drop]
    if len(keep) < len(r["events"]):
        if not keep:
            n_guard += 1; continue
        n_drop += len(r["events"]) - len(keep)
        log.append({"clip_id": r["clip_id"], "dropped": [e for e in r["events"] if e not in keep]})
        r["events"] = keep
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/predictions.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with zipfile.ZipFile(f"{OUT}/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/predictions.jsonl", "predictions.jsonl")
json.dump({"base": BASE, "thr_db": THR, "dropped": n_drop, "guarded_last_event": n_guard, "log": log}, open(f"{OUT}/info.json", "w"), indent=1)
print(f"{OUT}: dropped {n_drop} unsupported synthetic predictions (D < {THR} dB) | kept {n_guard} as a clip's last event | "
      f"events {sum(len(r['events']) for r in rows)}")
