"""Boundary Regressor Module for Datathon@IndoML 2026 Track 1.

Predicts fine-grained offset trims for candidate acoustic noise events to
recover events that overlap ground truth but fail the strict tolerance test.
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

FRAME = 0.02

def extract_event_features(ev, ev_idx, all_evs, p_ens, on_ens, off_ens, dur):
    """Extracts a 19-dimensional acoustic and temporal feature vector for a candidate event."""
    p_on, p_off = ev
    p_dur = p_off - p_on
    L = len(p_ens)
    i0 = max(0, min(L - 1, int(round(p_on / FRAME))))
    i1 = max(0, min(L - 1, int(round(p_off / FRAME))))

    seg_p = p_ens[i0:max(i0 + 1, i1)]
    p_mean = float(np.mean(seg_p))
    p_max = float(np.max(seg_p))
    p_min = float(np.min(seg_p))
    p_std = float(np.std(seg_p))

    on_at_start = float(on_ens[i0])
    off_at_end = float(off_ens[i1])

    # Presence around offset
    p_m2 = float(p_ens[max(0, i1 - 2)])
    p_m4 = float(p_ens[max(0, i1 - 4)])
    p_p2 = float(p_ens[min(L - 1, i1 + 2)])
    drop_slope = (p_m4 - p_p2) / 6.0

    # Presence around onset
    p_on_m2 = float(p_ens[max(0, i0 - 2)])
    p_on_p2 = float(p_ens[min(L - 1, i0 + 2)])
    rise_slope = (p_on_p2 - p_on_m2) / 4.0

    # Peak offset search inside candidate (up to 200 ms before offset)
    w_start = max(i0, i1 - 10)
    sub_off = off_ens[w_start:i1 + 1]
    if len(sub_off) > 0:
        k = int(np.argmax(sub_off))
        peak_off_val = float(sub_off[k])
        peak_off_dist = (w_start + k - i1) * FRAME
    else:
        peak_off_val = 0.0
        peak_off_dist = 0.0

    # Peak onset search near onset
    ow_start = max(0, i0 - 4)
    ow_end = min(L, i0 + 8)
    sub_on = on_ens[ow_start:ow_end]
    if len(sub_on) > 0:
        ok = int(np.argmax(sub_on))
        peak_on_val = float(sub_on[ok])
        peak_on_dist = (ow_start + ok - i0) * FRAME
    else:
        peak_on_val = 0.0
        peak_on_dist = 0.0

    return [
        p_dur,
        p_mean,
        p_max,
        p_min,
        p_std,
        on_at_start,
        off_at_end,
        p_m2,
        p_m4,
        p_p2,
        drop_slope,
        rise_slope,
        peak_off_val,
        peak_off_dist,
        peak_on_val,
        peak_on_dist,
        p_on / max(0.1, dur) if dur else 0.0,
        p_off / max(0.1, dur) if dur else 0.0,
        len(all_evs)
    ]

def apply_boundary_trim(events, p_ens, on_ens, off_ens, dur, regressor,
                        min_shift=-0.10, max_shift=-0.02):
    """Applies learned boundary trimming to a list of candidate events."""
    if not events or regressor is None:
        return events
    feats = [extract_event_features(ev, idx, events, p_ens, on_ens, off_ens, dur)
             for idx, ev in enumerate(events)]
    pred_shifts = regressor.predict(np.array(feats))

    adjusted = []
    for (p_on, p_off), s_off in zip(events, pred_shifts):
        if min_shift <= s_off <= max_shift:
            new_off = round(float(p_off + s_off), 3)
            if new_off - p_on >= 0.08:
                adjusted.append((p_on, new_off))
            else:
                adjusted.append((p_on, p_off))
        else:
            adjusted.append((p_on, p_off))
    return adjusted
