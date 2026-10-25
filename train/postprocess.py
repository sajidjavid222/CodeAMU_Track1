"""Frame probabilities -> event (onset, offset) list, plus tuning of the
decoding hyper-parameters against the real leaderboard metric.

Decoding matters as much as the network here: Event-F1 demands onsets and
offsets inside max(20% x duration, 50 ms), so sloppy boundaries score zero even
when the frames are right. Steps:

  1. median filter the probability track (removes 1-2 frame flicker)
  2. hysteresis threshold -- a region is active where p > low, but only kept if
     it ever exceeds high. Catches soft event tails without gluing noise
     together, and beats single thresholding on both metrics.
  3. bridge gaps shorter than `gap_sec`
  4. drop events shorter than `min_dur`
  5. optionally pad boundaries by `expand_sec` (events are often annotated a
     little wider than the acoustic energy)

`tune` grid-searches these on validation using metrics.score_submission, i.e.
it directly optimises Combined = Event F1 + Segment Dice.
"""

from __future__ import annotations

import itertools

import numpy as np

from metrics import score_submission


def median_filter(p, k):
    if k is None or k <= 1:
        return p
    if k % 2 == 0:
        k += 1
    pad = k // 2
    q = np.pad(p, (pad, pad), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(q, k), axis=-1)


def decode_frames(prob, frame_sec, high=0.5, low=None, median_k=5,
                  gap_sec=0.10, min_dur=0.10, expand_sec=0.0, duration=None):
    """Frame probabilities (1-D) -> list of (onset, offset) in seconds."""
    if low is None:
        low = high
    low = min(low, high)

    p = median_filter(np.asarray(prob, dtype=np.float64), median_k)
    active = p > low
    if not active.any():
        return []

    # contiguous runs above `low`, kept only if they peak above `high`
    idx = np.flatnonzero(np.diff(np.concatenate(([0], active.view(np.int8), [0]))))
    runs = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]
    runs = [(s, e) for s, e in runs if p[s:e].max() > high]
    if not runs:
        return []

    events = [[s * frame_sec, e * frame_sec] for s, e in runs]

    merged = [events[0]]                                   # bridge short gaps
    for on, off in events[1:]:
        if on - merged[-1][1] <= gap_sec:
            merged[-1][1] = off
        else:
            merged.append([on, off])

    out = []
    for on, off in merged:
        on = max(0.0, on - expand_sec)
        off = off + expand_sec
        if duration is not None:
            off = min(off, duration)
            on = min(on, duration)
        if off - on >= min_dur:
            out.append((round(float(on), 3), round(float(off), 3)))
    return out


def decode_all(prob_by_clip, frame_sec, durations=None, **kw):
    return {cid: decode_frames(p, frame_sec,
                               duration=(durations or {}).get(cid), **kw)
            for cid, p in prob_by_clip.items()}


DEFAULT_GRID = {
    "high": [0.3, 0.4, 0.5, 0.6, 0.7],
    "low": [0.15, 0.25, 0.35, 0.5],
    "median_k": [1, 5, 9, 15],
    "gap_sec": [0.0, 0.1, 0.2, 0.3],
    "min_dur": [0.05, 0.1, 0.2],
    "expand_sec": [0.0, 0.02, 0.05],
}


def tune(prob_by_clip, ref_by_clip, frame_sec, durations=None, grid=None,
         verbose=True, coarse_to_fine=True):
    """Grid-search decoding params to maximise Combined on validation.

    Returns (best_params, best_score_dict). A full product of DEFAULT_GRID is
    2880 evaluations; `coarse_to_fine` instead optimises one axis at a time
    from a sane starting point, repeated twice, which is ~10x cheaper and in
    practice lands on the same optimum.
    """
    grid = grid or DEFAULT_GRID

    def evaluate(params):
        preds = decode_all(prob_by_clip, frame_sec, durations, **params)
        return score_submission(preds, ref_by_clip)

    if coarse_to_fine:
        best = {"high": 0.5, "low": 0.35, "median_k": 5, "gap_sec": 0.1,
                "min_dur": 0.1, "expand_sec": 0.0}
        best = {k: v for k, v in best.items() if k in grid}
        best_score = evaluate(best)
        for _ in range(2):
            for key, values in grid.items():
                for v in values:
                    if v == best[key]:
                        continue
                    cand = dict(best, **{key: v})
                    if cand.get("low", 0) > cand.get("high", 1):
                        continue
                    sc = evaluate(cand)
                    if sc["combined"] > best_score["combined"]:
                        best, best_score = cand, sc
        if verbose:
            print(f"  tuned: {best} -> combined={best_score['combined']:.4f} "
                  f"(F1={best_score['event_f1']:.4f} "
                  f"Dice={best_score['segment_dice']:.4f})")
        return best, best_score

    keys = list(grid)
    best, best_score = None, {"combined": -1}
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        if params.get("low", 0) > params.get("high", 1):
            continue
        sc = evaluate(params)
        if sc["combined"] > best_score["combined"]:
            best, best_score = params, sc
    if verbose:
        print(f"  tuned(full): {best} -> combined={best_score['combined']:.4f}")
    return best, best_score


if __name__ == "__main__":
    frame_sec = 0.02
    n = 250                                        # 5 s clip
    prob = np.zeros(n)
    prob[50:100] = 0.9                             # 1.0 -> 2.0 s
    prob[70] = 0.2                                 # dropout to be smoothed
    prob[160:180] = 0.85                           # 3.2 -> 3.6 s
    ev = decode_frames(prob, frame_sec, high=0.5, low=0.35, median_k=5,
                       gap_sec=0.1, min_dur=0.1)
    print("decoded:", ev)
    assert len(ev) == 2, ev
    assert abs(ev[0][0] - 1.0) < 0.05 and abs(ev[0][1] - 2.0) < 0.05, ev

    ref = {"c": [(1.0, 2.0), (3.2, 3.6)]}
    print("score:", {k: round(v, 4) for k, v in
                     score_submission({"c": ev}, ref).items() if isinstance(v, float)})
    best, sc = tune({"c": prob}, ref, frame_sec, {"c": 5.0})
    print("postprocess self-test OK")


# --------------------------------------------------------------------------
# Boundary refinement using the dedicated onset/offset heads
# --------------------------------------------------------------------------
def snap_events(events, onset_p, offset_p, frame_sec, snap_win=0.12,
                min_peak=0.15, duration=None):
    """Pull each event's edges onto the nearest onset/offset peak.

    The presence head decides *that* an event is there; these heads decide
    *where* it starts and stops. Only moves an edge when a peak clears
    `min_peak`, so a silent boundary head can never make things worse.
    """
    if onset_p is None or offset_p is None or not events:
        return events
    w = max(1, int(round(snap_win / frame_sec)))
    n = len(onset_p)
    out = []
    for on, off in events:
        i = int(round(on / frame_sec))
        j = int(round(off / frame_sec))
        a0, a1 = max(0, i - w), min(n, i + w + 1)
        if a1 > a0:
            seg = onset_p[a0:a1]
            k = int(np.argmax(seg))
            if seg[k] >= min_peak:
                on = (a0 + k) * frame_sec
        b0, b1 = max(0, j - w), min(n, j + w + 1)
        if b1 > b0:
            seg = offset_p[b0:b1]
            k = int(np.argmax(seg))
            if seg[k] >= min_peak:
                off = (b0 + k) * frame_sec
        if duration is not None:
            off = min(off, duration)
        if off > on:
            out.append((round(float(on), 3), round(float(off), 3)))
    return out


def decode_all2(prob_by_clip, frame_sec, durations=None, onset_by_clip=None,
                offset_by_clip=None, snap_win=0.12, min_peak=0.15, **kw):
    """decode_all + boundary snapping (no-op when boundary tracks are absent)."""
    out = {}
    for cid, p in prob_by_clip.items():
        dur = (durations or {}).get(cid)
        ev = decode_frames(p, frame_sec, duration=dur, **kw)
        if onset_by_clip is not None and snap_win and snap_win > 0:
            ev = snap_events(ev, onset_by_clip.get(cid),
                             (offset_by_clip or {}).get(cid), frame_sec,
                             snap_win, min_peak, dur)
        out[cid] = ev
    return out


def tune2(prob_by_clip, ref_by_clip, frame_sec, durations=None,
          onset_by_clip=None, offset_by_clip=None, grid=None, verbose=True):
    """Coarse-to-fine tuning including the snapping parameters."""
    grid = dict(grid or DEFAULT_GRID)
    if onset_by_clip is not None:
        grid = dict(grid, snap_win=[0.0, 0.06, 0.10, 0.16, 0.24],
                    min_peak=[0.1, 0.2, 0.35, 0.5])
    best = {"high": 0.5, "low": 0.35, "median_k": 5, "gap_sec": 0.1,
            "min_dur": 0.1, "expand_sec": 0.0}
    if onset_by_clip is not None:
        best.update(snap_win=0.10, min_peak=0.2)
    best = {k: v for k, v in best.items() if k in grid}

    def ev(params):
        preds = decode_all2(prob_by_clip, frame_sec, durations,
                            onset_by_clip, offset_by_clip, **params)
        return score_submission(preds, ref_by_clip, durations)

    best_score = ev(best)
    for _ in range(2):
        for key, values in grid.items():
            for v in values:
                if v == best[key]:
                    continue
                cand = dict(best, **{key: v})
                if cand.get("low", 0) > cand.get("high", 1):
                    continue
                sc = ev(cand)
                if sc["combined"] > best_score["combined"]:
                    best, best_score = cand, sc
    if verbose:
        print(f"  tuned2: {best} -> combined={best_score['combined']:.4f} "
              f"(F1={best_score['event_f1']:.4f} Dice={best_score['segment_dice']:.4f})")
    return best, best_score


# --------------------------------------------------------------------------
# Splitting merged detections at internal boundary peaks
# --------------------------------------------------------------------------
def _peaks(track, lo, hi, thr, rad=2):
    """Indices in [lo,hi) that clear `thr` and are local maxima."""
    out = []
    for t in range(max(lo, 1), min(hi, len(track) - 1)):
        if track[t] < thr:
            continue
        a, b = max(lo, t - rad), min(hi, t + rad + 1)
        if track[t] >= track[a:b].max():
            out.append(t)
    return out


def split_events(events, onset_p, offset_p, frame_sec, split_thr=0.3,
                 min_sub=0.10, max_pair_gap=0.60):
    """Cut a detection wherever an offset peak is followed by an onset peak.

    Measured on validation: 38% of missed reference events are swallowed by a
    prediction spanning several of them, and 16.6% of predictions cover more
    than one reference. Snapping fixes a segment's outer edges but can never
    turn one detection into two, so those events are unrecoverable without an
    explicit split. An interior "event ends here, another starts there" pair is
    exactly what the offset/onset heads are trained to fire on.
    """
    if onset_p is None or offset_p is None or not events:
        return events
    min_f = max(1, int(round(min_sub / frame_sec)))
    max_gap_f = max(1, int(round(max_pair_gap / frame_sec)))
    out = []
    for on, off in events:
        i0, i1 = int(round(on / frame_sec)), int(round(off / frame_sec))
        if i1 - i0 < 2 * min_f:
            out.append((on, off))
            continue
        offs = _peaks(offset_p, i0 + min_f, i1 - min_f, split_thr)
        ons = _peaks(onset_p, i0 + min_f, i1 - min_f, split_thr)
        cuts, cur = [], i0
        for t in offs:
            if t - cur < min_f:
                continue
            nxt = [u for u in ons if t < u <= t + max_gap_f and i1 - u >= min_f]
            if nxt:
                cuts.append((t, nxt[0]))
                cur = nxt[0]
        if not cuts:
            out.append((on, off))
            continue
        pos = i0
        for t, u in cuts:
            out.append((round(pos * frame_sec, 3), round(t * frame_sec, 3)))
            pos = u
        out.append((round(pos * frame_sec, 3), round(i1 * frame_sec, 3)))
    return [(a, b) for a, b in out if b - a > 0]


def decode_all3(prob_by_clip, frame_sec, durations=None, onset_by_clip=None,
                offset_by_clip=None, snap_win=0.12, min_peak=0.15,
                split_thr=0.0, min_sub=0.10, max_pair_gap=0.60,
                trim_start=0.0, trim_end=0.0, min_dur=0.05, **kw):
    """decode -> split merged detections -> snap edges (order matters:
    split first so each piece gets its own boundaries snapped)."""
    out = {}
    for cid, p in prob_by_clip.items():
        dur = (durations or {}).get(cid)
        on = (onset_by_clip or {}).get(cid)
        off = (offset_by_clip or {}).get(cid)
        ev = decode_frames(p, frame_sec, duration=dur, min_dur=min_dur, **kw)
        if on is not None and split_thr and split_thr > 0:
            ev = split_events(ev, on, off, frame_sec, split_thr, min_sub, max_pair_gap)
        if on is not None and snap_win and snap_win > 0:
            ev = snap_events(ev, on, off, frame_sec, snap_win, min_peak, dur)
        if trim_start or trim_end:
            # Measured: unmatched-but-overlapping predictions run 1.49x the
            # reference duration (71% too long), starting 45 ms early and
            # ending 97 ms late. Asymmetric trimming is the direct correction;
            # the old expand_sec grid could only ever widen events.
            t = []
            for a, b in ev:
                a2, b2 = a + trim_start, b - trim_end
                if dur is not None:
                    b2 = min(b2, dur)
                if b2 - a2 >= max(0.03, min_dur * 0.5):
                    t.append((round(float(a2), 3), round(float(b2), 3)))
            ev = t
        out[cid] = ev
    return out


def tune3(prob_by_clip, ref_by_clip, frame_sec, durations=None,
          onset_by_clip=None, offset_by_clip=None, grid=None, verbose=True,
          rounds=2):
    """Coarse-to-fine tuning over decode + snap + split parameters."""
    grid = dict(grid or DEFAULT_GRID)
    grid = dict(grid, high=[0.4, 0.5, 0.6, 0.7, 0.8], low=[0.3, 0.4, 0.5, 0.6, 0.7])
    if onset_by_clip is not None:
        grid = dict(grid, snap_win=[0.0, 0.10, 0.16, 0.24],
                    min_peak=[0.1, 0.2, 0.35],
                    split_thr=[0.0, 0.15, 0.25, 0.35, 0.5],
                    min_sub=[0.06, 0.10, 0.16],
                    max_pair_gap=[0.3, 0.6, 1.0, 1.6],
                    trim_start=[0.0, 0.03, 0.06],
                    trim_end=[0.0, 0.04, 0.08, 0.14])
    best = {"high": 0.6, "low": 0.5, "median_k": 5, "gap_sec": 0.0,
            "min_dur": 0.05, "expand_sec": 0.0}
    if onset_by_clip is not None:
        best.update(snap_win=0.24, min_peak=0.1, split_thr=0.15,
                    min_sub=0.10, max_pair_gap=0.6, trim_start=0.0, trim_end=0.0)
    best = {k: v for k, v in best.items() if k in grid}

    def ev(params):
        preds = decode_all3(prob_by_clip, frame_sec, durations,
                            onset_by_clip, offset_by_clip, **params)
        return score_submission(preds, ref_by_clip, durations)

    best_score = ev(best)
    for _ in range(rounds):
        for key, values in grid.items():
            for v in values:
                if v == best[key]:
                    continue
                cand = dict(best, **{key: v})
                if cand.get("low", 0) > cand.get("high", 1):
                    continue
                sc = ev(cand)
                if sc["combined"] > best_score["combined"]:
                    best, best_score = cand, sc
    if verbose:
        print(f"  tuned3: {best}\n    -> combined={best_score['combined']:.4f} "
              f"F1={best_score['event_f1']:.4f} Dice={best_score['segment_dice']:.4f}")
    return best, best_score


# --------------------------------------------------------------------------
# Peak-pair decoding: events built from boundary peaks, not presence runs
# --------------------------------------------------------------------------
def decode_peak_pairs(prob, onset_p, offset_p, frame_sec, thr_on=0.3,
                      thr_off=0.3, pres_thr=0.4, min_dur=0.08, max_dur=8.0,
                      duration=None):
    """Pair each onset peak with the next offset peak, gated by presence.

    Presence-run decoding puts a boundary wherever the probability track
    happens to cross a threshold, which is why 45% of our missed events are
    "detected but out of tolerance" and 38% are merged. Here every emitted
    boundary sits exactly on a peak the onset/offset heads produced, and one
    onset can only ever close on one offset -- so adjacent events separate by
    construction instead of needing to be split afterwards.
    """
    n = len(prob)
    ons = _peaks(onset_p, 0, n, thr_on)
    offs = _peaks(offset_p, 0, n, thr_off)
    if not ons or not offs:
        return []
    min_f = max(1, int(round(min_dur / frame_sec)))
    max_f = max(min_f + 1, int(round(max_dur / frame_sec)))
    out, oi, last = [], 0, -1
    for o in ons:
        if o <= last:
            continue
        cand = [f for f in offs if o + min_f <= f <= o + max_f]
        chosen = None
        for f in cand:
            if prob[o:f].mean() >= pres_thr:
                chosen = f
                break
        if chosen is None:
            continue
        a, b = o * frame_sec, chosen * frame_sec
        if duration is not None:
            b = min(b, duration)
        if b - a >= min_dur:
            out.append((round(float(a), 3), round(float(b), 3)))
            last = chosen
    return out


def decode_all_peaks(prob_by_clip, frame_sec, durations=None,
                     onset_by_clip=None, offset_by_clip=None, **kw):
    return {cid: decode_peak_pairs(p, (onset_by_clip or {}).get(cid),
                                   (offset_by_clip or {}).get(cid), frame_sec,
                                   duration=(durations or {}).get(cid), **kw)
            for cid, p in prob_by_clip.items()}


PEAK_GRID = {"thr_on": [0.15, 0.25, 0.35, 0.5], "thr_off": [0.15, 0.25, 0.35, 0.5],
             "pres_thr": [0.25, 0.4, 0.55], "min_dur": [0.05, 0.08, 0.15],
             "max_dur": [3.0, 6.0, 10.0]}


def tune_peaks(prob_by_clip, ref_by_clip, frame_sec, durations=None,
               onset_by_clip=None, offset_by_clip=None, verbose=True, rounds=2):
    best = {"thr_on": 0.25, "thr_off": 0.25, "pres_thr": 0.4,
            "min_dur": 0.08, "max_dur": 6.0}

    def ev(p):
        return score_submission(
            decode_all_peaks(prob_by_clip, frame_sec, durations,
                             onset_by_clip, offset_by_clip, **p),
            ref_by_clip, durations)

    bs = ev(best)
    for _ in range(rounds):
        for k, vals in PEAK_GRID.items():
            for v in vals:
                if v == best[k]:
                    continue
                c = dict(best, **{k: v})
                s = ev(c)
                if s["combined"] > bs["combined"]:
                    best, bs = c, s
    if verbose:
        print(f"  peak-pair: {best}\n    -> combined={bs['combined']:.4f} "
              f"F1={bs['event_f1']:.4f} Dice={bs['segment_dice']:.4f} "
              f"TP={bs['tp']} FP={bs['fp']} FN={bs['fn']}")
    return best, bs
