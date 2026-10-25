"""Track 2: remove synthetic noise by SUBTRACTING the exact bank recording.

Synthetic clips = clean speech + noise recordings pasted whole from a bank that the
validation release exposes (noise = mixture - clean). bank_match.py (whitened, "pe")
already located every bank recording in each clip. Here, at 16 kHz:
  1. recover template k's raw waveform from its validation source clip,
  2. refine the lag within +-LAG samples of the matched position,
  3. least-squares gain, subtract -- nothing else in the clip is touched.
Unmatched noise is left in place (no zeroing / attenuation: that is what sank the
earlier t2_surgical_enhance.py at 15.4).

    python src/t2_bank_subtract.py val    # LOO validation: SI-SDR vs the true clean refs
    python src/t2_bank_subtract.py test   # writes work/t2_banksub/<clip>.wav + spans json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter

MODE = sys.argv[1]
OPTS = set(sys.argv[2:])        # "white": gain + lag on pre-emphasised signals (speech is low-band)
                                # "iterN": N joint re-estimation passes (overlapping events)
                                # "thX":   subtraction threshold (default 0.5)
ITERS = next((int(o[4:]) for o in OPTS if o.startswith("iter")), 0)
TH = next((float(o[2:]) for o in OPTS if o.startswith("th")), 0.5)
TAG = "_".join(sorted(OPTS)) or "base"
NBANK = "nb" in OPTS            # bank also holds validation NATURAL labelled events (bank_match ... nb)
SUFFIX = "_pe_nb" if NBANK else "_pe"
sys.argv = [sys.argv[0], "x", "pe"] + (["nb"] if NBANK else [])   # same bank -> same k indices
sys.path.insert(0, os.path.dirname(__file__))
import bank_match as BM                     # noqa: E402

V = "data/validation"
SR = 16000
LAG = 24


def si_sdr(est, ref):
    est = est - est.mean(); ref = ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    e = est - a * ref
    return 10 * np.log10((np.dot(a * ref, a * ref) + 1e-12) / (np.dot(e, e) + 1e-12))


_raw = {}
META = {}                       # "S": synthetic-only bank meta, "N": natural-augmented bank meta
NBX = "nbx" in OPTS             # primary = synthetic-bank matches (unchanged); extra = natural-source
NTH = next((float(o[3:]) for o in OPTS if o.startswith("nth")), 0.9)   # matches only where nothing
                                                                       # primary matched, cor >= NTH


def is_natural(cid):
    return not os.path.exists(f"{V}/syntheticNoiseAudio/{cid}.wav")


def clip_matches(cid, prim, extra, th):
    ms = [dict(m, src="S") for m in prim.get(cid, [])]
    if extra is not None:
        P = [(m["onset"], m["offset"]) for m in ms if m["cor"] >= th]
        for m in extra.get(cid, []):
            if m["cor"] >= NTH and is_natural(META["N"][m["k"]][0]) and \
                    all(min(m["offset"], b) - max(m["onset"], a) <= 0 for a, b in P):
                ms.append(dict(m, src="N"))
    return ms


def raw_template(k, which="S"):
    """Raw 16 kHz noise of bank template k = (mixture - clean) of its validation source."""
    if (which, k) not in _raw:
        cid, s0, s1 = (META.get(which) or BM.build_bank.meta)[k]
        if os.path.exists(f"{V}/syntheticNoiseAudio/{cid}.wav"):
            x, _ = sf.read(f"{V}/syntheticNoiseAudio/{cid}.wav", dtype="float64")
            c, _ = sf.read(f"{V}/syntheticCleanRefAudio/{cid}.wav", dtype="float64")
            x = x.mean(1) if x.ndim > 1 else x
            c = c.mean(1) if c.ndim > 1 else c
            n = min(len(x), len(c))
            src = x[:n] - c[:n]
        else:                                     # natural source: the recording itself
            src, _ = sf.read(f"{V}/naturalNoisyAudio/{cid}.wav", dtype="float64")
            src = src.mean(1) if src.ndim > 1 else src
        _raw[(which, k)] = src[int(round(s0 * SR)):int(round(s1 * SR))]
    return _raw[(which, k)]


def _pe(x):
    return np.append(x[:1], x[1:] - 0.97 * x[:-1])


NPE = next((int(o[5:]) for o in OPTS if o.startswith("white") and o[5:].isdigit()), 1 if "white" in OPTS else 0)


def _w(x):
    for _ in range(NPE):                          # "whiteN": pre-emphasis N times -- speech (low-band)
        x = _pe(x)                                # is pushed down, so the speech cross-term in the
    return x                                      # least-squares gain shrinks


def _gain(seg, t):
    ws, wt = _w(seg), _w(t)
    return np.dot(ws, wt) / (np.dot(wt, wt) + 1e-12)


GATE = next((float(o[4:]) for o in OPTS if o.startswith("gate")), None)   # "gateX": dB, see subtract
GLS = "gls" in OPTS             # "gls": re-fit each gain by generalised least squares in the STFT domain
NF, HOP, Q = 512, 128, 1 / 32768
WIN = np.hanning(NF + 1)[:-1]


def _stft(x):
    x = np.pad(x, (NF, NF)); n = 1 + (len(x) - NF) // HOP
    return np.fft.rfft(x[np.arange(NF)[None] + HOP * np.arange(n)[:, None]] * WIN, axis=1)


def _gls_gain(seg, t, g):
    """Speech under the event is the 'noise' of the gain regression. Weight every TF bin by
    1 / (local residual power): bins where speech is silent pin the gain almost exactly
    (relative error ~1e-5 vs ~2e-4 for the whitened LS). Floor = PCM16 rounding power."""
    Y, T = _stft(seg), _stft(t)
    floor = 2 * (1 + g * g) * Q * Q / 12 * np.sum(WIN ** 2)
    for _ in range(3):
        P = uniform_filter(np.abs(Y - g * T) ** 2, size=(3, 3), mode="nearest") + floor
        g = np.sum(np.real(Y * np.conj(T)) / P) / np.sum(np.abs(T) ** 2 / P)
    return max(0.0, float(g))


def subtract(y, matches, th, cid=None):
    out = y.astype(np.float64).copy()
    placed = []                                   # (a, t, g, k, src)
    for m in sorted((m for m in matches if m["cor"] >= th), key=lambda m: -m["cor"]):
        t = raw_template(m["k"], m.get("src", "S")); L = len(t)
        p = int(round(m["onset"] * SR))
        best = None
        for d in range(-LAG, LAG + 1):
            a = p + d
            if a < 0 or a + L > len(out):
                continue
            v = abs(np.dot(_w(out[a:a + L]), _w(t)))
            if best is None or v > best[0]:
                best = (v, a)
        if best is None:
            continue
        a = best[1]
        g = _gain(out[a:a + L], t)
        if GLS:
            g = _gls_gain(out[a:a + L], t, g)
        if m["cor"] < 0.5 and GATE is not None:       # weak match: keep only if it removes real energy
            seg = out[a:a + L]                        # (true copies remove >=1 dB, false ones ~0.00 dB)
            if 10 * np.log10(np.dot(seg, seg) / (np.sum((seg - g * t) ** 2) + 1e-18)) < GATE:
                continue
        out[a:a + L] -= g * t
        placed.append([a, t, g, m["k"], m.get("src", "S")])
    for _ in range(ITERS):                        # joint LS by coordinate descent: add each
        for pl in placed:                         # event back, re-fit its gain on the residual
            a, t, g = pl[:3]; L = len(t)
            out[a:a + L] += g * t
            pl[2] = _gls_gain(out[a:a + L], t, g) if GLS else _gain(out[a:a + L], t)
            out[a:a + L] -= pl[2] * t
    spans = [[a, a + len(t)] for a, t, *_ in placed]
    if GROW:
        for (a, t, g, k, which), sp in zip(placed, spans):
            if which == "S":
                sp[0], sp[1] = _grow(out, a, t, g, k, cid)
    return out, [(a / SR, b / SR) for a, b in spans]


GROW = "grow" in OPTS           # "grow": the bank holds truncated excerpts of long recordings (copies
GROWN, SRC = {}, []             # cut to different lengths). Extend a matched template with a longer
                                # copy of the same recording and cut where the target's noise ends.


def build_grow(bank8, head=2000, min_len=4000, cor_min=0.98):
    """k -> [(j, off)] with t_k[i] == scale * t_j[i + off] (16 kHz samples), found by locating k's
    first / last 0.25 s inside every other long entry (8 kHz whitened; copies correlate ~1.0)."""
    path = f"work/bank/grow{SUFFIX}.json"
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    import torch
    torch.set_num_threads(int(os.environ.get("GROW_THREADS", "8")))
    ks = sorted((k for k, t in enumerate(bank8) if len(t) >= min_len), key=lambda k: len(bank8[k]))
    buckets = [ks[i:i + 32] for i in range(0, len(ks), 32)]
    spec = []
    for b in buckets:
        N = 1 << int(np.ceil(np.log2(max(len(bank8[j]) for j in b) + head)))
        T = torch.zeros(len(b), N, dtype=torch.float64)
        for r, j in enumerate(b):
            T[r, :len(bank8[j])] = torch.from_numpy(bank8[j])
        cs = [np.concatenate([[0.0], np.cumsum(bank8[j] ** 2)]) for j in b]
        spec.append((b, N, torch.fft.rfft(T, n=N), cs))
    out = {}
    for n_done, k in enumerate(ks):
        t = bank8[k]
        for part, h in (("head", t[:head]), ("tail", t[-head:])):
            hn = np.linalg.norm(h)
            for b, N, Tf, cs in spec:
                xc = torch.fft.irfft(Tf * torch.conj(torch.fft.rfft(torch.from_numpy(h), n=N)), n=N).numpy()
                for r, j in enumerate(b):
                    n = len(bank8[j])
                    if j == k or n < head:
                        continue
                    cor = xc[r, :n - head + 1] / (hn * np.sqrt(np.maximum(cs[r][head:] - cs[r][:-head], 1e-18)))
                    o = int(np.argmax(cor))
                    if cor[o] >= cor_min:
                        off = o if part == "head" else o - (len(t) - head)
                        if off > 0 or off + len(t) < n:          # j reaches past k on some side
                            out.setdefault(k, {})[j] = 2 * off
        if n_done % 100 == 0:
            print(f"  grow index {n_done}/{len(ks)}", flush=True)
    out = {k: sorted(v.items()) for k, v in out.items()}
    json.dump(out, open(path, "w"))
    print(f"grow index: {len(out)} of {len(ks)} long templates have a longer copy", flush=True)
    return out


def _cut(r, e, g):
    """Length of the prefix of e whose subtraction lowers the residual energy the most."""
    d = r * r - (r - g * e) ** 2
    if not len(d):
        return 0
    C = np.cumsum(d); i = int(np.argmax(C))
    return i + 1 if C[i] > 0 else 0


def _grow(out, a, t, g, k, cid):
    L, n = len(t), len(out)
    best = {"after": None, "before": None}
    for j, off in GROWN.get(k, []):
        if cid is not None and cid in SRC[j]:           # leave-one-out: never use the target's own copy
            continue
        tj = raw_template(j)
        cands = []
        for o in range(off - 3, off + 4):              # 8 kHz offset -> exact 16 kHz sample
            i0, i1 = max(0, -o), min(L, len(tj) - o)
            if i1 - i0 < 1600:
                continue
            u, v = t[i0:i1], tj[i0 + o:i1 + o]
            cands.append((abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-18), o, i0, i1))
        if not cands:
            continue
        c, o, i0, i1 = max(cands)
        if c < 0.99:
            continue
        u, v = t[i0:i1], tj[i0 + o:i1 + o]
        sc = np.dot(u, v) / (np.dot(v, v) + 1e-18)
        aft, bef = len(tj) - o - L, o
        if aft > 0 and (best["after"] is None or aft > best["after"][0]):
            best["after"] = (aft, sc * tj[o + L:])
        if bef > 0 and (best["before"] is None or bef > best["before"][0]):
            best["before"] = (bef, sc * tj[:o])
    lo, hi = a, a + L
    if best["after"] is not None:
        e = best["after"][1][:max(0, n - hi)]
        c = _cut(out[hi:hi + len(e)], e, g)
        out[hi:hi + c] -= g * e[:c]; hi += c
    if best["before"] is not None:
        e = best["before"][1][::-1][:a]                 # walk left from the template start
        c = _cut(out[a - len(e):a][::-1], e, g)
        out[a - c:a] -= g * e[:c][::-1]; lo -= c
    return lo, hi


def load_banks(split):
    """-> (primary matches, extra natural-source matches or None)."""
    if NBX:
        BM.NB = False; BM.build_bank(); META["S"] = BM.build_bank.meta
        BM.NB = True; BM.build_bank(); META["N"] = BM.build_bank.meta
        return (json.load(open(f"work/bank/{split}_pe.json")),
                json.load(open(f"work/bank/{split}_pe_nb.json")))
    bank8, src = BM.build_bank()
    if GROW:
        SRC[:] = src
        GROWN.update(build_grow(bank8))
    return json.load(open(f"work/bank/{split}{SUFFIX}.json")), None


def main():
    if MODE == "val":
        prim, extra = load_banks("val")
        bank = {c: clip_matches(c, prim, extra, TH) for c in prim}
        md = [r for r in json.load(open(f"{V}/validationMetadata.json")) if r.get("syntheticData")]
        print(f"variant: {TAG}")
        for th in ((0.4, 0.5, 0.7, 0.9) if not any(o.startswith("th") for o in OPTS) else (TH,)):
            s_in, s_out, full = [], [], 0
            for r in md:
                cid = r["segmentFileName"][:-4]
                x, _ = sf.read(f"{V}/syntheticNoiseAudio/{cid}.wav", dtype="float64")
                c, _ = sf.read(f"{V}/syntheticCleanRefAudio/{cid}.wav", dtype="float64")
                x = x.mean(1) if x.ndim > 1 else x
                c = c.mean(1) if c.ndim > 1 else c
                n = min(len(x), len(c)); x, c = x[:n], c[:n]
                y, _ = subtract(x, bank.get(cid, []), th, cid)
                s_in.append(si_sdr(x, c)); s_out.append(si_sdr(y, c))
            s_out = np.array(s_out)
            print(f"  th {th}: noisy {np.mean(s_in):6.2f} dB -> subtracted mean {s_out.mean():6.2f} dB "
                  f"(median {np.median(s_out):6.2f}; >=40 dB {100 * np.mean(s_out >= 40):5.1f}% of clips, "
                  f"<20 dB {100 * np.mean(s_out < 20):5.1f}%)", flush=True)
    else:
        th = TH
        prim, extra = load_banks("test")
        bank = {c: clip_matches(c, prim, extra, th) for c in prim}
        odir = "work/t2_banksub" if TAG == "base" else f"work/t2_banksub_{TAG}"
        os.makedirs(odir, exist_ok=True)
        spans = {}
        si, sn = (int(v) for v in os.environ.get("T2_SHARD", "0/1").split("/"))   # "i/n": clips i::n
        items = sorted(bank.items())[si::sn]
        for i, (cid, v) in enumerate(items):
            x, sr = sf.read(f"{BM.TEST}/{cid}.wav", dtype="float64")
            x = x.mean(1) if x.ndim > 1 else x
            y, done = subtract(x, v, th, cid)
            sf.write(f"{odir}/{cid}.wav", y.astype(np.float32), sr, subtype="FLOAT")
            spans[cid] = done
            if i % 200 == 0:
                print(f"  {i}/{len(bank)}", flush=True)
        json.dump(spans, open(f"{odir}/subtracted_spans" + (f"_{si}" if sn > 1 else "") + ".json", "w"))
        print(f"wrote {len(spans)} clips to {odir}/ (threshold {th}, variant {TAG})")


if __name__ == "__main__":
    main()
