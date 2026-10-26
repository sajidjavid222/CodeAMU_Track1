"""Locate synthetic noise events by matching against the validation reference noise.

The synthetic clips are clean speech plus noise clips pasted WHOLE from a finite
refnoise: in validation every sampled event has a sample-identical copy in another
clip (verified correlation 1.000, different speech underneath, same duration and
start to <=5 ms). Validation gives the refnoise for free: noise = mixture - clean,
cut at the labelled timestamps.

For each clip we cross-correlate the mixture against every refnoise template
(batched FFT at 8 kHz), keep peaks by normalised correlation, then greedily pick
non-overlapping matches by explained energy. A match gives the event's onset and
offset at sample precision.

    python src/refnoise_match.py val    # leave-one-clip-out on validation -> work/refnoise/val.json
    python src/refnoise_match.py test   # test synthetic clips             -> work/refnoise/test.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch

V = "data/validation"
TEST = os.environ.get("T1_TEST_AUDIO", "test_audio")          # folder of the test WAVs
SR = 8000
TH_KEEP = 0.20          # candidate floor; the working threshold is chosen on validation
OUT = "work/refnoise"
torch.set_num_threads(int(os.environ.get("REFNOISE_THREADS", "32")))


NB = "nb" in sys.argv[2:]      # add validation NATURAL labelled events to the refnoise
PE = "pe" in sys.argv[2:]      # pre-emphasis whitening: exact copies stay exact (linear filter),
                                 # chance alignments of tonal / low-band sounds and speech drop


def white(x):
    return np.append(x[:1], x[1:] - 0.97 * x[:-1]) if PE else x


def read8k(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    return white(ss.resample_poly(x, SR, sr) if sr != SR else x)


def build_templates():
    md = [r for r in json.load(open(f"{V}/validationMetadata.json")) if r.get("syntheticData")]
    raw = []
    for r in md:
        f = r["segmentFileName"]
        x, sr = sf.read(f"{V}/syntheticNoiseAudio/{f}", dtype="float64")
        c, _ = sf.read(f"{V}/syntheticCleanRefAudio/{f}", dtype="float64")
        x = x.mean(1) if x.ndim > 1 else x
        c = c.mean(1) if c.ndim > 1 else c
        k = min(len(x), len(c))
        n = white(ss.resample_poly(x[:k] - c[:k], SR, sr))
        for e in r["NoiseSubCategoryTimeStamp"]:
            a, b = int(round(float(e["start"]) * SR)), int(round(float(e["end"]) * SR))
            t = n[a:b]
            if len(t) >= int(0.05 * SR) and np.linalg.norm(t) > 1e-6:
                raw.append((t / np.linalg.norm(t), f[:-4], float(e["start"]), float(e["end"])))
    if NB:
        # the refnoise was CUT from natural recordings at their labelled edges, so every labelled
        # event of a validation NATURAL clip is a candidate refnoise recording as-is
        for r in json.load(open(f"{V}/validationMetadata.json")):
            if r.get("syntheticData"):
                continue
            f = r["segmentFileName"]
            n = read8k(f"{V}/naturalNoisyAudio/{f}")
            for e in r.get("NoiseSubCategoryTimeStamp") or []:
                a, b = int(round(float(e["start"]) * SR)), int(round(float(e["end"]) * SR))
                t = n[a:b]
                if len(t) >= int(0.05 * SR) and np.linalg.norm(t) > 1e-6:
                    raw.append((t / np.linalg.norm(t), f[:-4], float(e["start"]), float(e["end"])))
    # dedupe identical recordings (same length +-8 samples, normalised dot >= 0.995)
    raw.sort(key=lambda z: len(z[0]))
    refnoise, src, meta = [], [], []
    for t, cid, s0, s1 in raw:
        hit = None
        for j in range(len(refnoise) - 1, -1, -1):
            if len(refnoise[j]) < len(t) - 8:
                break
            m = min(len(t), len(refnoise[j]), 4000)
            if abs(np.dot(t[:m], refnoise[j][:m])) / (np.linalg.norm(t[:m]) * np.linalg.norm(refnoise[j][:m]) + 1e-12) >= 0.995:
                hit = j
                break
        if hit is None:
            refnoise.append(t); src.append({cid}); meta.append((cid, s0, s1))   # where template k came from
        else:
            src[hit].add(cid)
    print(f"refnoise: {len(raw)} labelled events -> {len(refnoise)} distinct recordings", flush=True)
    build_templates.meta = meta
    return refnoise, src


class Matcher:
    def __init__(self, refnoise):
        self.refnoise = refnoise
        self.lens = np.array([len(t) for t in refnoise])
        self.cache = {}

    def tf(self, N):
        if N not in self.cache:
            self.cache.clear()
            T = torch.zeros(len(self.refnoise), N, dtype=torch.float32)
            for i, t in enumerate(self.refnoise):
                T[i, :len(t)] = torch.from_numpy(t.astype(np.float32))
            self.cache[N] = torch.fft.rfft(T, n=N)
        return self.cache[N]

    def match(self, x, forbid=None, chunk=256):
        Lx = len(x)
        N = 1 << int(np.ceil(np.log2(Lx + self.lens.max())))
        Tf = self.tf(N)
        X = torch.fft.rfft(torch.from_numpy(x.astype(np.float32)), n=N)
        cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        cands = []
        for s in range(0, len(self.refnoise), chunk):
            idx = np.arange(s, min(s + chunk, len(self.refnoise)))
            xc = torch.fft.irfft(X[None, :] * torch.conj(Tf[idx]), n=N)[:, :Lx].numpy()
            for r, k in enumerate(idx):
                n = self.lens[k]
                if n > Lx or (forbid is not None and forbid[k]):
                    continue
                v = xc[r, :Lx - n + 1]
                we = np.sqrt(np.maximum(cs[n:Lx + 1] - cs[:Lx - n + 1], 1e-12))
                cor = np.abs(v) / we                          # templates are unit-norm
                for _ in range(3):                            # same noise may be pasted twice
                    j = int(np.argmax(cor))
                    if cor[j] < TH_KEEP:
                        break
                    cands.append((float(v[j] ** 2), float(cor[j]), j, int(k)))
                    cor[max(0, j - n // 2):j + n // 2 + 1] = 0
        cands.sort(reverse=True)                               # by explained energy
        picked = []
        for ee, c, j, k in cands:
            a, b = j, j + self.lens[k]
            if any(min(b, B) - max(a, A) > 0.3 * min(b - a, B - A) for A, B, *_ in picked):
                continue
            picked.append((a, b, c, ee, k))
        return [dict(onset=a / SR, offset=b / SR, cor=round(c, 4), ee=ee, k=k) for a, b, c, ee, k in picked]


def main():
    mode = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    refnoise, src = build_templates()
    M = Matcher(refnoise)
    res, t0 = {}, time.time()
    if mode == "val":
        md = [r for r in json.load(open(f"{V}/validationMetadata.json")) if r.get("syntheticData")]
        jobs = [(r["segmentFileName"][:-4], f"{V}/syntheticNoiseAudio/{r['segmentFileName']}") for r in md]
    elif mode == "valnat":      # control: known-natural clips -> how often natural audio fakes a match
        md = [r for r in json.load(open(f"{V}/validationMetadata.json")) if not r.get("syntheticData")]
        jobs = [(r["segmentFileName"][:-4], f"{V}/naturalNoisyAudio/{r['segmentFileName']}") for r in md]
    else:
        syn = json.load(open("work/eval_syn_pred_fixed.json"))
        want = (mode == "test")  # "testnat": clips the router sent to the NATURAL branch
        jobs = [(c, f"{TEST}/{c}.wav") for c, v in sorted(syn.items()) if bool(v) == want]
    for i, (cid, path) in enumerate(jobs):
        forbid = np.array([s == {cid} for s in src]) if mode == "val" else None   # leave-one-clip-out
        res[cid] = M.match(read8k(path), forbid)
        if i % 100 == 0:
            print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    tag = mode + ("_pe" if PE else "") + ("_nb" if NB else "")
    json.dump(res, open(f"{OUT}/{tag}.json", "w"))
    print(f"wrote {OUT}/{tag}.json  ({len(res)} clips, {sum(len(v) for v in res.values())} matches, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
