"""Classify whether a clip is SYNTHETIC (clean speech + injected noise) or NATURAL.

Why this exists: routing clips to a regime-specialist model is worth +0.042
Combined with a perfect oracle, but a single noise-floor threshold gets only
73.5% accuracy and the routed result comes out *worse* than not routing at all
(1.3770 vs 1.3790) -- misrouted clips lose exactly what correct routing wins.

The official validation release carries a ground-truth `syntheticData` flag for
2,501 clips, which is a proper supervised training set for this.

Features target what "clean speech + injected noise" leaves behind: an
unnaturally quiet and *spectrally flat* background between events, little
high-frequency content in quiet frames, low variability of the noise floor over
time, and abrupt frame-to-frame energy jumps where a snippet was spliced in.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

SR = 16000
HOP = 160
NFFT = 512


def clip_features(path, sr=SR):
    y, in_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if in_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    n = len(y) // HOP
    if n < 20:
        return np.zeros(24, dtype=np.float32)
    fr = y[:n * HOP].reshape(n, HOP)
    e = np.sqrt((fr ** 2).mean(1)) + 1e-9
    edb = 20 * np.log10(e)

    # spectra, frame-wise
    w = np.hanning(HOP)
    S = np.abs(np.fft.rfft(fr * w, n=NFFT, axis=1)) + 1e-9
    P = S ** 2
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)
    tot = P.sum(1) + 1e-9
    centroid = (P * freqs).sum(1) / tot
    hf = P[:, freqs > 4000].sum(1) / tot                     # HF share
    flat = np.exp(np.log(P + 1e-12).mean(1)) / (P.mean(1) + 1e-12)   # spectral flatness

    q = np.percentile(edb, [5, 10, 25, 50, 75, 95])
    quiet = edb <= np.percentile(edb, 25)                    # the "background" frames
    d = np.diff(edb)

    f = [
        *q,                                   # 0-5   energy percentiles (dB)
        q[5] - q[0], q[4] - q[1],             # 6-7   dynamic range
        edb.std(), np.median(np.abs(d)),      # 8-9   variability / frame-to-frame jump
        np.percentile(np.abs(d), 95), (np.abs(d) > 6).mean(),   # 10-11 splice-like jumps
        centroid[quiet].mean(), centroid.mean(),                # 12-13 spectral centroid
        hf[quiet].mean(), hf.mean(),                            # 14-15 HF share
        flat[quiet].mean(), flat.mean(), flat[quiet].std(),     # 16-18 flatness
        edb[quiet].std(),                                       # 19    floor stability
        float(np.mean(edb < -60)), float(np.mean(edb < -70)),   # 20-21 near-digital-silence
        float(len(y) / sr), float(np.abs(y).max()),             # 22-23 duration, peak
    ]
    return np.nan_to_num(np.array(f, dtype=np.float32))


def build_matrix(paths, workers=16):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.stack(list(ex.map(clip_features, paths, chunksize=16)))
