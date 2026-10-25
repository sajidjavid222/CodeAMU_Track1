"""v68 = v67 + noise-bank boundary snapping on the synthetic branch.

Synthetic test clips are matched against the validation noise bank
(src/bank_match.py -> work/bank/test.json); src/bank_merge.merge with the V68 rule
moves each predicted synthetic event onto its matched recording's exact extent.
Natural clips are byte-identical to v67. Validated leave-one-clip-out on
validation: synthetic OOF +0.0131 over v67, both folds.
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(__file__))
from bank_merge import V68, V69, merge

SRC = "submission/v67/predictions.jsonl"
OUT = "submission/v68"
# `build_v68.py v69` = rule V69 on the WHITENED test matches (work/bank/test_pe.json) -> submission/v69
VER = "v69" if "v69" in sys.argv[1:] else "v68"
RULE = V69 if VER == "v69" else V68
if VER == "v69":
    OUT = "submission/v69"


def main():
    syn = json.load(open("work/eval_syn_pred_fixed.json"))
    bank = json.load(open("work/bank/test_pe.json" if VER == "v69" else "work/bank/test.json"))
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    n_syn = n_ev = n_moved = n_nomatch = 0
    for r in rows:
        c = r["clip_id"]
        if not syn.get(c, False):
            continue
        n_syn += 1
        if c not in bank:
            n_nomatch += 1
            continue
        pred = [(e["onset"], e["offset"]) for e in r["events"]]
        new = merge(pred, bank[c], **RULE)
        n_ev += len(pred)
        n_moved += sum(1 for e in new if e not in set(pred))
        r["events"] = [dict(onset=round(a, 4), offset=round(b, 4)) for a, b in new]
    os.makedirs(OUT, exist_ok=True)
    jl = os.path.join(OUT, "predictions.jsonl")
    with open(jl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with zipfile.ZipFile(os.path.join(OUT, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jl, "predictions.jsonl")
    trusted = sum(1 for v in bank.values() for m in v if m["cor"] >= 0.9)
    print(f"clips {len(rows)} | synthetic {n_syn} (no bank result {n_nomatch}) | "
          f"synthetic events {n_ev}, moved onto a bank match {n_moved} ({100 * n_moved / max(n_ev, 1):.1f}%) | "
          f"matches cor>=0.9: {trusted}")
    print(f"wrote {OUT}/submission.zip")
    json.dump(dict(version=VER, base="v67", rule=RULE, synthetic_events=n_ev, moved=n_moved),
              open(os.path.join(OUT, "info.json"), "w"), indent=2)


# ---- v70: v69 + ROUTING FIX. A natural-routed clip that contains validation-bank noise
# recordings (whitened matches) is synthetic; decode it with the synthetic branch instead.
# Threshold calibrated on the validation NATURAL control (bank_match.py valnat pe): chosen
# so that no known-natural validation clip would be rerouted. Natural clips legitimately hold
# bank SOURCES (the bank was cut from natural recordings): 2 known-natural validation clips
# have 2 matches >= 0.9, none has 3 (not even at >= 0.7).
REROUTE_COR, REROUTE_MIN_N = 0.9, 3
# Clips that stay natural get merge_nat (V70_NAT): a trusted match is an exact natural event.


def build_v70():
    from build_v60 import load_tracks
    from bank_merge import V70_NAT, merge_nat
    from postprocess import decode_frames, split_events
    from dur_snap import dur_snap
    from boundary_regressor import FRAME
    nat_bank = json.load(open("work/bank/testnat_pe.json"))
    move = sorted(c for c, v in nat_bank.items() if sum(1 for m in v if m["cor"] >= REROUTE_COR) >= REROUTE_MIN_N)
    pr = json.load(open("work/route_v43.json"))["syn_params"]
    dec = {k: v for k, v in pr.items() if k not in ("snap_win", "min_peak")}
    mem = json.load(open("work/syn_branch_v57.json"))["members"]
    M = [load_tracks(f"work/evalprobs/{m}.npz") for m in mem]
    rows = [json.loads(l) for l in open("submission/v69/predictions.jsonl") if l.strip()]
    idx = {r["clip_id"]: r for r in rows}
    done = 0
    for c in move:
        if not all(c in m[0] for m in M):
            continue
        k = min(len(m[0][c]) for m in M)
        p, on, off = (np.mean([m[i][c][:k] for m in M], axis=0) for i in range(3))
        e = decode_frames(p, FRAME, duration=None, **dec)
        e = split_events(e, on, off, FRAME, split_thr=0.06, min_sub=0.10, max_pair_gap=0.40)
        e = dur_snap(e, on, off, FRAME, duration=None, split=0.45, snap_short=0.12, peak_short=0.10,
                     snap_long=0.22, peak_long=0.20)
        new = merge([tuple(x) for x in e], nat_bank[c], **V69)
        old = len(idx[c]["events"])
        idx[c]["events"] = [dict(onset=round(a, 4), offset=round(b, 4)) for a, b in new]
        done += 1
        print(f"  rerouted {c[-34:]}  natural {old} ev -> synthetic {len(new)} ev")
    syn = json.load(open("work/eval_syn_pred_fixed.json"))
    n_nat = n_add = n_new = 0
    for c, v in nat_bank.items():            # natural-routed clips that stay natural
        if c in move or syn.get(c, False):
            continue
        old = [(e["onset"], e["offset"]) for e in idx[c]["events"]]
        new = merge_nat(old, v, **V70_NAT)
        if new != sorted(set(old)):
            n_nat += 1
            n_add += len(new) - len(set(old))
            n_new += sum(1 for e in new if e not in old)
            idx[c]["events"] = [dict(onset=round(a, 4), offset=round(b, 4)) for a, b in new]
    print(f"natural clips changed by exact bank matches: {n_nat} | events added {n_add}, "
          f"new or snapped events {n_new}")
    out = "submission/v70"
    os.makedirs(out, exist_ok=True)
    jl = os.path.join(out, "predictions.jsonl")
    with open(jl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jl, "predictions.jsonl")
    json.dump(dict(version="v70", base="v69", rerouted=move, rule=V69, nat_rule=V70_NAT, nat_clips_changed=n_nat,
                   reroute=dict(cor=REROUTE_COR, min_n=REROUTE_MIN_N)),
              open(os.path.join(out, "info.json"), "w"), indent=2)
    print(f"natural-routed clips with >= {REROUTE_MIN_N} bank matches at cor >= {REROUTE_COR}: {len(move)}; "
          f"rerouted {done} -> {out}/submission.zip")


if __name__ == "__main__":
    if "v70" in sys.argv[1:]:
        import numpy as np
        build_v70()
    else:
        main()
