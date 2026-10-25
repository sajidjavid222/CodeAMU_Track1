"""Synthesise training clips with machine-exact noise-event boundaries.

Why: every measurement says boundary precision is the bottleneck, and we only
have 10,000 human-annotated gold clips. Mixing harvested noise into verified
clean speech gives unlimited clips whose onsets/offsets are exact by
construction -- and 2,080 of the 5,517 eval clips (38%) are built the same way
("clean speech with noise added synthetically").

Banks are harvested from GOLD clips only, since those timestamps are verified:
  * noise bank  -- the annotated event spans themselves
  * clean bank  -- the gaps between events, where annotators confirmed silence
                   of events (>= --min-clean seconds)

Placement deliberately over-samples SHORT inter-event gaps: 29% of real gold
gaps are under 0.3 s, and merging across those gaps is our single most common
failure (v1 predicted 1.4 events/clip against 1.78 reference).

    python src/make_synthetic.py --manifest work/train/train_split.jsonl \
        --root work/train --out work/synth --n 20000
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import soundfile as sf

SR = 16000


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def fade(x, n=240):                      # 15 ms cosine edges
    if len(x) < 2 * n or n <= 0:
        return x
    w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y = x.copy()
    y[:n] *= w
    y[-n:] *= w[::-1]
    return y


def harvest(records, root, min_clean=2.5, min_noise=0.15, max_clips=0):
    """-> (noise_bank[(audio, category)], clean_bank[audio])"""
    noise, clean = [], []
    gold = [r for r in records if r.get("tier") == "gold" and r.get("events")]
    if max_clips:
        gold = gold[:max_clips]
    for i, r in enumerate(gold):
        path = r["path"] if os.path.isabs(r["path"]) else os.path.join(root, r["path"])
        try:
            y, sr = sf.read(path, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        evs = sorted(r["events"], key=lambda e: e["onset"])
        for e in evs:                                    # noise spans
            a, b = int(e["onset"] * sr), int(e["offset"] * sr)
            a, b = max(0, a), min(len(y), b)
            if (b - a) / sr >= min_noise:
                noise.append((y[a:b].copy(), e.get("category")))
        cur = 0.0                                        # verified-clean gaps
        for e in evs + [{"onset": len(y) / sr, "offset": len(y) / sr}]:
            g0, g1 = cur, e["onset"]
            if g1 - g0 >= min_clean:
                a, b = int(g0 * sr), int(g1 * sr)
                seg = y[a:b]
                if len(seg) > 0 and rms(seg) > 1e-4:     # skip digital silence
                    clean.append(seg.copy())
            cur = max(cur, e["offset"])
        if (i + 1) % 2000 == 0:
            print(f"  harvested {i+1}/{len(gold)} gold clips -> "
                  f"{len(noise)} noise, {len(clean)} clean", flush=True)
    return noise, clean


def make_bed(clean, target_len, rng, tries=12):
    """Prefer ONE long contiguous clean region as the bed.

    Concatenating short fragments produces speech-dense audio with no natural
    pauses, and noise added to a uniformly loud bed can only ever be *louder*
    than its surroundings -- the opposite of the measured real distribution
    (real event regions sit at -2.3 dB median, in the gaps between speech).
    A single long region keeps the speaker's own pause structure intact.
    """
    for _ in range(tries):
        seg = clean[rng.randrange(len(clean))]
        if len(seg) >= target_len:
            off = rng.randrange(0, len(seg) - target_len + 1)
            return seg[off:off + target_len].astype(np.float32)
    out, total = [], 0                       # fallback: stitch a few
    while total < target_len:
        seg = clean[rng.randrange(len(clean))]
        out.append(seg)
        total += len(seg)
    return np.concatenate(out)[:target_len].astype(np.float32)


def quiet_bias_offsets(bed, win, rng, n_cand=24, alpha=1.5):
    """Candidate start offsets, biased toward quiet stretches of the bed.

    Measured on 388 real gold clips, annotated event regions are *quieter*
    than the rest of the clip (median 0.77x, -2.3 dB; only 26% louder) --
    real noise lands in the gaps between speech, not on top of it. Placing
    events uniformly would make them systematically louder than the
    surrounding speech and teach "noise == loud", which is the opposite of
    the real distribution.
    """
    if win >= len(bed):
        return [0]
    step = max(1, int(0.05 * SR))
    starts = np.arange(0, len(bed) - win, step)
    if len(starts) == 0:
        return [0]
    e = np.array([rms(bed[s:s + win]) for s in starts]) + 1e-6
    w = (1.0 / e) ** alpha
    w = w / w.sum()
    idx = np.random.choice(len(starts), size=min(n_cand, len(starts)),
                           replace=False, p=w)
    return [int(starts[i]) for i in sorted(idx)]


def synth_clip(noise, clean, rng, dur_range=(3.0, 10.0), n_events=(1, 3),
               snr_range=(-22.0, -2.0), short_gap_p=0.35):
    n = int(rng.uniform(*dur_range) * SR)
    bed = make_bed(clean, n, rng)
    y = bed.copy()
    k = rng.randint(*n_events)

    events, cursor = [], rng.uniform(0.05, 0.6)
    for _ in range(k):
        ns, cat = noise[rng.randrange(len(noise))]
        ns = fade(ns.astype(np.float32).copy())
        if len(ns) >= n:
            continue
        # honour the running cursor (keeps gaps controlled) but snap to the
        # quietest candidate at/after it, so events land in speech gaps.
        cands = [c for c in quiet_bias_offsets(bed, len(ns), rng)
                 if c >= int(cursor * SR)]
        a = cands[0] if cands else int(cursor * SR)
        b = a + len(ns)
        if b > n - int(0.05 * SR):
            break
        # level set against the WHOLE clip, not the local window: matching the
        # global inside-vs-outside ratio is what the real measurement showed.
        snr = rng.uniform(*snr_range)
        scale = rms(bed) / (rms(ns) + 1e-9) * (10 ** (snr / 20.0))
        y[a:b] += ns * scale
        events.append({"onset": round(a / SR, 4), "offset": round(b / SR, 4),
                       "category": cat, "tag": ""})
        # next gap: deliberately short 35% of the time to teach separation
        gap = rng.uniform(0.05, 0.30) if rng.random() < short_gap_p \
            else rng.uniform(0.4, 2.5)
        cursor = b / SR + gap
        if cursor * SR >= n:
            break

    peak = np.abs(y).max()
    if peak > 0.99:
        y = y / peak * 0.99
    return y.astype(np.float32), events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-clean", type=float, default=2.5)
    ap.add_argument("--harvest-clips", type=int, default=0)
    ap.add_argument("--snr-lo", type=float, default=-22.0,
                    help="speech-over-noise dB, low end (more negative = louder noise)")
    ap.add_argument("--snr-hi", type=float, default=-2.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    recs = [json.loads(l) for l in open(args.manifest)]
    print("harvesting banks from gold clips...", flush=True)
    noise, clean = harvest(recs, args.root, args.min_clean,
                           max_clips=args.harvest_clips)
    print(f"noise bank: {len(noise)} segs "
          f"({sum(len(a) for a, _ in noise)/SR/3600:.2f} h)")
    print(f"clean bank: {len(clean)} segs "
          f"({sum(len(a) for a in clean)/SR/3600:.2f} h)")
    if not noise or not clean:
        raise SystemExit("empty bank -- cannot synthesise")

    audio_dir = os.path.join(args.out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    out = []
    for i in range(args.n):
        y, evs = synth_clip(noise, clean, rng,
                            snr_range=(args.snr_lo, args.snr_hi))
        if not evs:
            continue
        cid = f"synth_{i:06d}"
        sf.write(os.path.join(audio_dir, cid + ".wav"), y, SR, subtype="PCM_16")
        out.append({"clip_id": cid, "path": f"audio/{cid}.wav",
                    "duration": round(len(y) / SR, 4), "tier": "gold",
                    "language": None, "state": None, "district": None,
                    "categories": sorted({e["category"] for e in evs if e["category"]}),
                    "events": evs, "synthetic": True})
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{args.n} clips", flush=True)

    man = os.path.join(args.out, "manifest.jsonl")
    with open(man, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    d = sum(r["duration"] for r in out)
    ne = sum(len(r["events"]) for r in out)
    print(f"\nwrote {man}: {len(out)} clips, {d/3600:.2f} h, {ne} events "
          f"({ne/max(1,len(out)):.2f}/clip)")


if __name__ == "__main__":
    main()
