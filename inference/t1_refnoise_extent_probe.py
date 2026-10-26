"""Where does a matched refnoise template really explain the Track 1 test audio? (truncated-copy check)

For a match (template k at onset), subtract g*t at the best lag; per 10 ms frame, red = 10log10(E_mix/E_resid).
Frames where the copy is present show a clear energy reduction; template frames past a truncation point show
red <= 0 (subtracting noise that is not there adds energy). Extent = first..last frame with red >= RED_DB.
    python src/t1_refnoise_extent_probe.py submission/v93_s1/info.json
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

ARGS = list(sys.argv)
sys.path.insert(0, os.path.dirname(__file__))
import phase2_refnoise_subtract_opt as B  # noqa: E402  (overwrites sys.argv)
from metrics import match_events      # noqa: E402

TEST = os.environ.get("T1_TEST_AUDIO", "test_audio")          # folder of the test WAVs
SR, FR, RED_DB = 16000, 160, 1.0


def extent(x, m, meta):
    t = B.get_template(m["k"], meta); L = len(t); p = int(round(m["onset"] * SR))
    best = max(((abs(np.dot(B._w2(x[a:a + L]), B._w2(t))), a) for a in range(p - 32, p + 33) if a >= 0 and a + L <= len(x)), default=None)
    if best is None:
        return None
    a = best[1]; seg = x[a:a + L]; g = B._gain(seg, t); r = seg - g * t
    n = L // FR
    ex = (seg[:n * FR].reshape(n, FR) ** 2).mean(1) + 1e-12; er = (r[:n * FR].reshape(n, FR) ** 2).mean(1) + 1e-12
    red = np.convolve(10 * np.log10(ex / er), np.ones(3) / 3, mode="same")
    on = np.flatnonzero(red >= RED_DB)
    if len(on) == 0:
        return (a / SR, (a + L) / SR, None, None, red)
    return (a / SR, (a + L) / SR, (a + on[0] * FR) / SR, (a + (on[-1] + 1) * FR) / SR, red)


def main():
    info = json.load(open(ARGS[1]))
    B.load_synth_templates()
    import refnoise_match as BM
    meta = list(BM.build_templates.meta)
    M = json.load(open("work/refnoise/test_pe.json"))
    cache = {}
    def audio(c):
        if c not in cache:
            x, _ = sf.read(f"{TEST}/{c}.wav", dtype="float64"); cache[c] = x if x.ndim == 1 else x.mean(1)
        return cache[c]
    def find(c, a):
        return min(M[c], key=lambda m: abs(m["onset"] - a))

    # control: reliable matches v93 already agrees with (known correct on the leaderboard)
    P = {json.loads(l)["clip_id"]: [(e["onset"], e["offset"]) for e in json.loads(l)["events"]] for l in open("submission/v93/predictions.jsonl")}
    rng = np.random.default_rng(0); ctl = []
    for c in (rng.permutation([c for c in M if c in P])[:400] if os.environ.get("CONTROL", "1") == "1" else []):
        for m in M[c]:
            if m["cor"] < 0.6:
                continue
            L = meta[m["k"]][2] - meta[m["k"]][1]; span = (m["onset"], m["onset"] + L)
            if match_events(P[c], [span])[0] == 0:
                continue
            e = extent(audio(c), m, meta)
            if e and e[2] is not None:
                ctl.append((e[2] - span[0], e[3] - span[1], L))
        if len(ctl) >= 300:
            break
    ctl = np.array(ctl)
    if len(ctl): print(f"CONTROL ({len(ctl)} agreeing matches): energy extent - refnoise span | onset median {np.median(ctl[:, 0]) * 1000:.0f} ms, "
          f"|.|<=50ms {np.mean(np.abs(ctl[:, 0]) <= 0.05):.2f} | offset median {np.median(ctl[:, 1]) * 1000:.0f} ms, "
          f"|.|<=0.2*dur {np.mean(np.abs(ctl[:, 1]) <= np.maximum(0.2 * ctl[:, 2], 0.05)):.2f}")
    print("\nCHANGED EVENTS: clip | refnoise span | old pred | energy extent | frac frames red>=1dB | last-100ms mean red")
    for e in info["log"]:
        c = e["clip_id"]
        spans = e["spans"] if "spans" in e else [e["added"]]
        for a, b, cr in spans:
            m = find(c, a); r = extent(audio(c), m, meta)
            old = e.get("pred")
            ee = f"({r[2]:.3f}, {r[3]:.3f})" if r and r[2] is not None else "none"
            frac = float(np.mean(r[4] >= RED_DB)) if r else 0.0
            tail = float(np.mean(r[4][-10:])) if r else 0.0
            print(f"{c[-28:]} | refnoise ({a:.3f}, {b:.3f}) cor {cr:.2f} | old {str(tuple(round(v, 3) for v in old)) if old else 'none':>16} | "
                  f"energy {ee} | {frac:.2f} | {tail:+.1f} dB")


if __name__ == "__main__":
    main()
