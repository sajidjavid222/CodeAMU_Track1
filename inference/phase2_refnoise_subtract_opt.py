"""Phase 2 Reference-Noise Subtraction Optimized (Unconstrained Engine).

Applies exact mathematical noise subtraction to all 1,635 matched synthetic clips in Phase 2
using the verified primary synthetic refnoise matches from work/refnoise/phase2_pe.json at threshold 0.45 (9,065 bursts).
Keeps the unconstrained gain formula that achieves 48.5 - 49.4 dB SI-SDR on validation.
Outputs clean WAVs to work/refnoise_sub_p2_opt/ and subtracted_spans.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.argv = [sys.argv[0], "x", "pe"]
sys.path.insert(0, os.path.dirname(__file__))
import refnoise_match as BM

V = "data/validation"
P2_AUDIO = os.environ.get("T1_TEST_AUDIO", "test_audio")
OUT_DIR = "work/refnoise_sub_p2_opt"
SR = 16000
LAG = 32
TH_SYNTH = 0.45

_raw_cache = {}


def load_synth_templates():
    BM.NB = False
    BM.build_templates()
    return list(BM.build_templates.meta)


def get_template(k: int, meta: list) -> np.ndarray:
    if k not in _raw_cache:
        cid, s0, s1 = meta[k]
        x_path = f"{V}/syntheticNoiseAudio/{cid}.wav"
        c_path = f"{V}/syntheticCleanRefAudio/{cid}.wav"
        x, _ = sf.read(x_path, dtype="float64")
        c, _ = sf.read(c_path, dtype="float64")
        x = x.mean(1) if x.ndim > 1 else x
        c = c.mean(1) if c.ndim > 1 else c
        n = min(len(x), len(c))
        src = x[:n] - c[:n]
        _raw_cache[k] = src[int(round(s0 * SR)):int(round(s1 * SR))]
    return _raw_cache[k]


def _pe(x: np.ndarray) -> np.ndarray:
    return np.append(x[:1], x[1:] - 0.97 * x[:-1])


def _w2(x: np.ndarray) -> np.ndarray:
    return _pe(_pe(x))


def _gain(seg: np.ndarray, t: np.ndarray) -> float:
    ws, wt = _w2(seg), _w2(t)
    g = np.dot(ws, wt) / (np.dot(wt, wt) + 1e-12)
    return max(0.0, float(g))


def subtract_clip(y: np.ndarray, matches: list[dict], meta_synth: list) -> tuple[np.ndarray, list[tuple[float, float]]]:
    out = y.astype(np.float64).copy()
    placed = []

    for m in sorted(matches, key=lambda x: -x["cor"]):
        t = get_template(m["k"], meta_synth)
        L = len(t)
        p = int(round(m["onset"] * SR))

        best = None
        for d in range(-LAG, LAG + 1):
            a = p + d
            if a < 0 or a + L > len(out):
                continue
            v = abs(np.dot(_w2(out[a:a + L]), _w2(t)))
            if best is None or v > best[0]:
                best = (v, a)

        if best is None:
            continue

        a = best[1]
        g = _gain(out[a:a + L], t)
        out[a:a + L] -= g * t
        placed.append((a / SR, (a + L) / SR))

    return out, placed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading synthetic reference noise templates...", flush=True)
    meta_synth = load_synth_templates()
    print(f"Loaded {len(meta_synth)} synthetic templates.", flush=True)

    prim_synth_path = "work/refnoise/phase2_pe.json"
    assert os.path.exists(prim_synth_path), f"Missing {prim_synth_path}!"
    prim_synth = json.load(open(prim_synth_path))

    all_jobs = []
    for cid, s_matches in prim_synth.items():
        selected = [m for m in s_matches if m["cor"] >= TH_SYNTH]
        if selected:
            all_jobs.append((cid, selected))

    print(f"Total synthetic Phase 2 clips to subtract: {len(all_jobs)}", flush=True)

    spans = {}
    n_events_subtracted = 0
    t0 = time.time()

    for i, (cid, matches) in enumerate(all_jobs):
        p = os.path.join(P2_AUDIO, f"{cid}.wav")
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float64")
        x = x.mean(1) if x.ndim > 1 else x
        y, done = subtract_clip(x, matches, meta_synth)
        out_path = os.path.join(OUT_DIR, f"{cid}.wav")
        sf.write(out_path, y.astype(np.float32), sr, subtype="FLOAT")
        spans[cid] = done
        n_events_subtracted += len(done)

        if (i + 1) % 250 == 0 or (i + 1) == len(all_jobs):
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(all_jobs)}] {elapsed:.1f}s | subtracted events: {n_events_subtracted}", flush=True)

    spans_path = os.path.join(OUT_DIR, "subtracted_spans.json")
    with open(spans_path, "w") as f:
        json.dump(spans, f, indent=2)

    print(f"\nPhase 2 Optimized Reference-Noise Subtraction Complete!")
    print(f"Wrote {len(spans)} audio files to {OUT_DIR}/")
    print(f"Logged {n_events_subtracted} subtracted spans to {spans_path}")


if __name__ == "__main__":
    main()
