"""Duration-conditional boundary snapping.

Event-F1's tolerance is `max(0.20 * ref_duration, 0.05)`, so a 0.25 s event is
judged at +/-50 ms while a 2 s event gets +/-400 ms -- an 8x difference in
strictness. Every event nonetheless goes through one snap window and one peak
threshold. This applies a different (snap_win, min_peak, edge shift) to short
and long predictions, with the split point itself tuned.

Offsets also run late (measured bias +49 ms on natural validation, and offset
clears tolerance only 64.5% of the time vs onset's 74.3%), so each bucket gets
its own onset/offset shift.
"""
from __future__ import annotations
import numpy as np


def _snap_one(on, off, onset_p, offset_p, frame_sec, snap_win, min_peak,
              on_shift, off_shift, duration):
    n = len(onset_p)
    w = max(1, int(round(snap_win / frame_sec)))
    if w > 0 and snap_win > 0:
        i = int(round(on / frame_sec))
        a0, a1 = max(0, i - w), min(n, i + w + 1)
        if a1 > a0:
            seg = onset_p[a0:a1]
            k = int(np.argmax(seg))
            if seg[k] >= min_peak:
                on = (a0 + k) * frame_sec
        j = int(round(off / frame_sec))
        b0, b1 = max(0, j - w), min(n, j + w + 1)
        if b1 > b0:
            seg = offset_p[b0:b1]
            k = int(np.argmax(seg))
            if seg[k] >= min_peak:
                off = (b0 + k) * frame_sec
    on = max(0.0, on + on_shift)
    off = off + off_shift
    if duration is not None:
        off = min(off, duration)
    return on, off


def dur_snap(events, onset_p, offset_p, frame_sec, duration=None,
             split=0.4,
             snap_short=0.10, peak_short=0.2, on_short=0.0, off_short=0.0,
             snap_long=0.24, peak_long=0.1, on_long=0.0, off_long=0.0):
    if onset_p is None or offset_p is None or not events:
        return events
    out = []
    for on, off in events:
        short = (off - on) < split
        a, b = _snap_one(on, off, onset_p, offset_p, frame_sec,
                         snap_short if short else snap_long,
                         peak_short if short else peak_long,
                         on_short if short else on_long,
                         off_short if short else off_long,
                         duration)
        if b > a:
            out.append((round(float(a), 3), round(float(b), 3)))
    return out


def dur_snap_all(pred_by_clip, on_by_clip, off_by_clip, frame_sec,
                 durations=None, **kw):
    return {c: dur_snap(ev, (on_by_clip or {}).get(c), (off_by_clip or {}).get(c),
                        frame_sec, (durations or {}).get(c), **kw)
            for c, ev in pred_by_clip.items()}
