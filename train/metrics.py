"""Exact re-implementation of the Datathon@IndoML 2026 Track-1 scoring metrics.

Spec (competition "Evaluation" page):

  1. Event-based F1 (+/-20% tolerance)
       For a reference event of duration d = offset - onset:
           tol = max(0.20 * d, 0.05) seconds
       A predicted event matches when
           |pred_onset  - ref_onset|  <= tol   AND
           |pred_offset - ref_offset| <= tol
       Events are matched greedily (closest first); micro-averaged F1.

  2. Segment-level Dice
       Timeline split into 10 ms frames, binary masks for reference and
       prediction, Dice = 2|P n G| / (|P| + |G|), macro-averaged over clips.

  3. Combined = Event F1 + Segment Dice   (leaderboard ranking, max 2.0)

Used both to score local validation and to tune post-processing.
"""

from __future__ import annotations

import numpy as np

FRAME_SEC = 0.010          # 10 ms segment resolution
MIN_TOL_SEC = 0.05         # 50 ms tolerance floor
TOL_FRAC = 0.20            # +/-20% of reference event duration


# --------------------------------------------------------------------------
# Event-based F1
# --------------------------------------------------------------------------
def match_events(pred, ref, tol_frac=TOL_FRAC, min_tol=MIN_TOL_SEC):
    """Greedily match predicted events to reference events (closest first).

    pred, ref : sequences of (onset, offset) in seconds.
    Returns (n_matched, list_of_pairs).
    """
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref):
        tol = max(tol_frac * (r_off - r_on), min_tol)
        for pi, (p_on, p_off) in enumerate(pred):
            d_on = abs(p_on - r_on)
            d_off = abs(p_off - r_off)
            if d_on <= tol and d_off <= tol:
                candidates.append((d_on + d_off, pi, ri))

    candidates.sort(key=lambda x: x[0])          # closest first
    used_p, used_r, pairs = set(), set(), []
    for _, pi, ri in candidates:
        if pi in used_p or ri in used_r:
            continue
        used_p.add(pi)
        used_r.add(ri)
        pairs.append((pi, ri))
    return len(pairs), pairs


def event_f1(pred_by_clip, ref_by_clip, tol_frac=TOL_FRAC, min_tol=MIN_TOL_SEC):
    """Micro-averaged event-based F1 over all clips."""
    tp = fp = fn = 0
    for clip_id, ref in ref_by_clip.items():
        pred = pred_by_clip.get(clip_id, [])
        n_match, _ = match_events(pred, ref, tol_frac, min_tol)
        tp += n_match
        fp += len(pred) - n_match
        fn += len(ref) - n_match

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"event_f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------
# Segment-level Dice
# --------------------------------------------------------------------------
def events_to_mask(events, n_frames, frame_sec=FRAME_SEC):
    """Binary frame mask for a list of (onset, offset) events."""
    mask = np.zeros(n_frames, dtype=bool)
    for on, off in events:
        i0 = max(0, int(np.floor(on / frame_sec)))
        i1 = min(n_frames, int(np.ceil(off / frame_sec)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def segment_dice(pred_by_clip, ref_by_clip, durations=None,
                 frame_sec=FRAME_SEC, empty_score=1.0):
    """Macro-averaged Dice across clips.

    `empty_score` is used when a clip has neither reference nor predicted
    noise frames (0/0). The organisers' script does not state this case; 1.0
    (perfect agreement on "no noise") is the common convention. Set to 0.0 to
    see the pessimistic bound -- `score_submission` reports both.
    """
    scores = []
    for clip_id, ref in ref_by_clip.items():
        pred = pred_by_clip.get(clip_id, [])
        if durations is not None and clip_id in durations:
            total = durations[clip_id]
        else:
            total = max([off for _, off in list(ref) + list(pred)] or [0.0])
        n_frames = max(1, int(np.ceil(total / frame_sec)))

        g = events_to_mask(ref, n_frames, frame_sec)
        p = events_to_mask(pred, n_frames, frame_sec)
        denom = p.sum() + g.sum()
        scores.append(empty_score if denom == 0 else 2.0 * (p & g).sum() / denom)
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------
def score_submission(pred_by_clip, ref_by_clip, durations=None):
    """Full leaderboard score: Combined = Event F1 + Segment Dice."""
    ev = event_f1(pred_by_clip, ref_by_clip)
    dice = segment_dice(pred_by_clip, ref_by_clip, durations, empty_score=1.0)
    dice_pess = segment_dice(pred_by_clip, ref_by_clip, durations, empty_score=0.0)
    out = dict(ev)
    out["segment_dice"] = dice
    out["segment_dice_empty0"] = dice_pess
    out["combined"] = ev["event_f1"] + dice
    out["n_clips"] = len(ref_by_clip)
    return out


# --------------------------------------------------------------------------
# I/O helpers for predictions.jsonl
# --------------------------------------------------------------------------
def load_jsonl_events(path):
    """Read a predictions.jsonl -> {clip_id: [(onset, offset), ...]}."""
    import json
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["clip_id"]] = [(float(e["onset"]), float(e["offset"]))
                                   for e in rec.get("events", [])]
    return out


def write_jsonl_events(path, pred_by_clip, clip_order=None):
    """Write predictions.jsonl in the exact submission format."""
    import json
    clips = clip_order if clip_order is not None else sorted(pred_by_clip)
    with open(path, "w") as fh:
        for clip_id in clips:
            events = [{"onset": round(float(on), 3), "offset": round(float(off), 3)}
                      for on, off in pred_by_clip.get(clip_id, [])]
            fh.write(json.dumps({"clip_id": clip_id, "events": events}) + "\n")
    return path


if __name__ == "__main__":
    # Self-test against the worked example in the competition spec.
    ref = {"a": [(1.0, 3.0), (5.0, 6.0)], "b": [], "c": [(0.2, 0.85)]}

    perfect = {"a": [(1.0, 3.0), (5.0, 6.0)], "b": [], "c": [(0.2, 0.85)]}
    s = score_submission(perfect, ref)
    assert abs(s["event_f1"] - 1.0) < 1e-9 and abs(s["segment_dice"] - 1.0) < 1e-9
    print("perfect ->", {k: round(v, 4) for k, v in s.items() if isinstance(v, float)})

    # 1.0->3.0 has d=2.0 so tol=0.4: onset 1.35 and offset 3.35 are inside.
    near = {"a": [(1.35, 3.35)], "b": [], "c": [(0.2, 0.85)]}
    print("near    ->", {k: round(v, 4) for k, v in score_submission(near, ref).items()
                         if isinstance(v, float)})

    # 0.2->0.85 has d=0.65 so tol=max(0.13,0.05)=0.13: offset off by 0.2 fails.
    out_of_tol = {"a": [], "b": [], "c": [(0.2, 1.05)]}
    r = score_submission(out_of_tol, ref)
    assert r["tp"] == 0, r
    print("out-tol ->", {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})

    # Short event: d=0.04 -> tol floored at 0.05 not 0.008.
    ref_s = {"x": [(1.00, 1.04)]}
    assert score_submission({"x": [(1.04, 1.08)]}, ref_s)["tp"] == 1
    assert score_submission({"x": [(1.06, 1.10)]}, ref_s)["tp"] == 0
    print("50ms floor OK")
    print("all metric self-tests passed")
