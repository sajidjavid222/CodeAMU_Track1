"""Are v93_banksyn2's 75 bank-unsupported synthetic predictions real noise? Track 2 enhancer as a noise meter.

Per clip: enhance with the p2mix ensemble (whole-clip window), D = 10log10(E_removed / E_mix) per 10 ms frame.
Score each region by mean D. Controls in the same clips: bank spans (cor >= 0.6: noise present) and random
0.3 s stretches with no prediction and no bank span (speech only).
    python src/t1_syn_fp_probe.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # inference/enhance.py (Track 2 TF-DPCRN)
torch.set_num_threads(int(os.environ.get("NT", "24")))
import enhance as E  # noqa: E402

TEST = os.environ.get("T1_TEST_AUDIO", "test_audio")          # folder of the test WAVs
SR, HOP = 16000, 160


def main():
    syn = json.load(open("work/eval_syn_pred_fixed.json")); rer = set(json.load(open("submission/v70/info.json")).get("rerouted", []))
    P = {json.loads(l)["clip_id"]: [(e["onset"], e["offset"]) for e in json.loads(l)["events"]] for l in open("submission/v93_banksyn2/predictions.jsonl")}
    M = json.load(open("work/bank/test_pe.json"))
    S = json.load(open("work/t2_banksub_gate0.5_gls_grow_th0.25_white2/subtracted_spans.json"))
    E.load_models([os.path.join(os.environ.get("T1_ENHANCER_DIR", "checkpoints/enhancer"), f"t2_model_{i}.pt") for i in range(3)])
    rng = np.random.default_rng(0)
    sus, pos, neg = [], [], []
    clips = []
    for c, pr in P.items():
        if not (syn.get(c, False) or c in rer):
            continue
        spans = [(m["onset"], m["offset"]) for m in M.get(c, []) if m["cor"] >= 0.45] + [tuple(x[:2]) for x in S.get(c, [])]
        odd = [(a, b) for a, b in pr if not any(min(b, y) > max(a, x) for x, y in spans)]
        if odd:
            clips.append((c, pr, spans, odd))
    print(f"clips with unsupported predictions: {len(clips)}", flush=True)
    for c, pr, spans, odd in clips:
        x, _ = sf.read(f"{TEST}/{c}.wav", dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
        y = E.ens_enhance(x, [(0.0, len(x) / SR)]); n = min(len(x), len(y)) // HOP
        ex = (x[:n * HOP].reshape(n, HOP).astype(np.float64) ** 2).mean(1) + 1e-10
        er = ((x[:n * HOP] - y[:n * HOP]).reshape(n, HOP).astype(np.float64) ** 2).mean(1) + 1e-10
        D = 10 * np.log10(er / ex)
        md = lambda a, b: float(np.mean(D[int(a * 100):max(int(a * 100) + 1, int(b * 100))]))
        for a, b in odd:
            sus.append((c, a, b, md(a, b)))
        for x0, y0 in spans[:3]:
            pos.append(md(x0, y0))
        busy = sorted(spans + pr)
        for _ in range(3):
            for _try in range(30):
                a = rng.uniform(0, n / 100 - 0.3); b = a + 0.3
                if not any(min(b, y0) > max(a, x0) - 0.1 for x0, y0 in busy):
                    neg.append(md(a, b)); break
    pos, neg = np.array(pos), np.array(neg)
    s = np.array([v[3] for v in sus])
    q = lambda v: " ".join(f"{np.percentile(v, p):6.1f}" for p in (10, 25, 50, 75, 90))
    print(f"removed-energy share D (dB), percentiles 10/25/50/75/90")
    print(f"  bank-span regions (noise)      n={len(pos):3d}: {q(pos)}")
    print(f"  free stretches (speech only)   n={len(neg):3d}: {q(neg)}")
    print(f"  UNSUPPORTED predictions        n={len(s):3d}: {q(s)}")
    for thr in (-15, -12, -10, -8, -6):
        print(f"  D < {thr:4d} dB: unsupported {int(np.sum(s < thr)):3d}/{len(s)} | noise regions {np.mean(pos < thr):.2f} | speech regions {np.mean(neg < thr):.2f}")
    json.dump([{"clip_id": c, "onset": a, "offset": b, "D": d} for c, a, b, d in sus], open("work/t1_syn_unsupported_D.json", "w"), indent=1)


if __name__ == "__main__":
    main()
